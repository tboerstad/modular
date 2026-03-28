# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Quantized matmul operations.

The main entry point is :func:`scaled_matmul`, which dispatches on the
concrete :class:`ScaledTensor` subtype to the right kernel.
"""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue

from .kernels import (
    _fused_qkv_ragged_matmul_scaled_float8,
    block_scales_interleave,
    dynamic_block_scaled_matmul_fp4,
    dynamic_scaled_matmul,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_matmul_ragged,
    mxfp4_dequant,
    quantize_dynamic_scaled_float8,
)
from .kv_cache import KVCacheParams, PagedCacheValues
from .quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from .scaled_tensors import Float8Tensor, Mxfp4Tensor, Nvfp4Tensor, ScaledTensor


# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------


def prepare_nvfp4_weight(weight: Nvfp4Tensor, device: DeviceRef) -> Nvfp4Tensor:
    """Interleave NVFP4 block scales for GPU kernel consumption.

    Converts the flat ``[M, K//16]`` scale layout loaded from disk into
    the 5-D interleaved layout required by the matmul kernel.

    Args:
        weight: An :class:`Nvfp4Tensor` with flat (non-interleaved)
            block scales.
        device: Target device for the scales tensor.

    Returns:
        A new :class:`Nvfp4Tensor` with interleaved scales on
        ``device``.
    """
    return Nvfp4Tensor(
        data=weight.data,
        scale=block_scales_interleave(weight.scale.to(device)),
        global_scale=weight.global_scale,
    )


# ---------------------------------------------------------------------------
# Dense matmul — single dispatch on weight type
# ---------------------------------------------------------------------------


def scaled_matmul(
    x: TensorValue,
    weight: ScaledTensor,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Mixed-precision matmul: ``x @ weight.T``.

    Dispatches to the appropriate kernel based on the concrete weight
    type.  The input ``x`` is expected to be bf16; the kernel handles
    any necessary quantisation internally.

    Args:
        x: Input activation tensor (typically bf16).
        weight: Quantised weight — one of :class:`Float8Tensor`,
            :class:`Nvfp4Tensor`, or :class:`Mxfp4Tensor`.
        out_type: Desired output dtype (default ``bfloat16``).

    Returns:
        Result tensor of shape ``[..., N]``.
    """
    if isinstance(weight, Nvfp4Tensor):
        return dynamic_block_scaled_matmul_fp4(
            x,
            weight.data,
            weight.scale,
            weight.global_scale,
            out_type=out_type,
        )
    elif isinstance(weight, Float8Tensor):
        return dynamic_scaled_matmul(
            x,
            weight.data,
            weight.scale,
            out_type=out_type,
        )
    elif isinstance(weight, Mxfp4Tensor):
        dequanted = mxfp4_dequant(
            weight.data, weight.scale, out_type=out_type
        )
        return x @ dequanted.T
    else:
        raise TypeError(f"Unsupported weight type: {type(weight)}")


# ---------------------------------------------------------------------------
# Helpers — derive scale specs from Float8Tensor shape
# ---------------------------------------------------------------------------


def _infer_input_scale_spec(weight: Float8Tensor) -> InputScaleSpec:
    """Derive :class:`InputScaleSpec` from a ``Float8Tensor``'s scale shape.

    Convention:
      - scalar / ``(N, 1)`` scale → per-token input (colwise)
      - 2-D ``(N/bs_n, K/bs_k)`` scale → blockwise ``(1, bs_k)`` input
    """
    scale = weight.scale
    data = weight.data

    if scale.rank == 0 or (scale.rank == 2 and int(scale.shape[1]) == 1):
        return InputScaleSpec(
            granularity=ScaleGranularity.COLWISE,
            origin=ScaleOrigin.DYNAMIC,
            dtype=scale.dtype,
        )

    if scale.rank == 2 and int(scale.shape[1]) > 1:
        bs_k = int(data.shape[1]) // int(scale.shape[1])
        return InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=scale.dtype,
            block_size=(1, bs_k),
        )

    # rank-1 (N,) → rowwise weight, colwise input
    return InputScaleSpec(
        granularity=ScaleGranularity.COLWISE,
        origin=ScaleGranularity.COLWISE,
        dtype=scale.dtype,
    )


def _infer_weight_scale_spec(weight: Float8Tensor) -> WeightScaleSpec:
    """Derive :class:`WeightScaleSpec` from a ``Float8Tensor``'s scale shape."""
    scale = weight.scale
    data = weight.data

    if scale.rank == 0:
        return WeightScaleSpec(
            granularity=ScaleGranularity.TENSOR,
            dtype=scale.dtype,
        )

    if scale.rank == 2 and int(scale.shape[1]) == 1:
        return WeightScaleSpec(
            granularity=ScaleGranularity.ROWWISE,
            dtype=scale.dtype,
        )

    if scale.rank == 2 and int(scale.shape[1]) > 1:
        bs_n = int(data.shape[0]) // int(scale.shape[0])
        bs_k = int(data.shape[1]) // int(scale.shape[1])
        return WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=scale.dtype,
            block_size=(bs_n, bs_k),
        )

    # rank-1 (N,) → rowwise
    return WeightScaleSpec(
        granularity=ScaleGranularity.ROWWISE,
        dtype=scale.dtype,
    )


def _infer_scales_granularity_mnk(
    weight: Float8Tensor,
) -> tuple[int, int, int]:
    """Compute the ``(M, N, K)`` scale granularity tuple from tensor shapes.

    This replaces the previous ``QuantConfig.scales_granularity_mnk``
    property for callers that only have a :class:`Float8Tensor`.
    """
    scale = weight.scale
    data = weight.data

    # Scalar or per-tensor
    if scale.rank == 0:
        return (-1, -1, -1)

    if scale.rank == 2 and int(scale.shape[1]) == 1:
        # (N, 1) → per-row weight, per-token input
        return (1, 1, -1)

    if scale.rank == 2 and int(scale.shape[1]) > 1:
        # Blockwise
        bs_n = int(data.shape[0]) // int(scale.shape[0])
        bs_k = int(data.shape[1]) // int(scale.shape[1])
        return (1, bs_n, bs_k)

    if scale.rank == 1:
        # (N,) → per-row weight, per-token input
        return (1, 1, -1)

    raise ValueError(f"Unexpected scale shape: {scale.shape}")


# ---------------------------------------------------------------------------
# Fused QKV matmul (FP8 only — NVFP4 callers use scaled_matmul directly)
# ---------------------------------------------------------------------------


def fused_qkv_matmul(
    kv_params: KVCacheParams,
    x: TensorValue,
    weight: Float8Tensor,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    input_row_offsets: TensorValue,
    n_heads: int,
    bias: TensorValue | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Fused QKV matmul + KV-cache store (FP8).

    The kernel handles input quantisation internally.  All scale
    information is derived from the ``weight`` tensor — no
    :class:`QuantConfig` is needed.

    Args:
        kv_params: KV cache parameters.
        x: Input tensor ``[total_seq_len, hidden_dim]`` (bf16).
        weight: :class:`Float8Tensor` holding the concatenated QKV
            weight and its scales.
        kv_collection: The paged KV cache.
        layer_idx: Current layer index.
        input_row_offsets: Batch boundary offsets.
        n_heads: Number of attention heads.
        bias: Optional bias tensor.
        _output_dim: Optional output dimension override.

    Returns:
        The query projection output tensor.
    """
    input_scale_spec = _infer_input_scale_spec(weight)
    weight_scale_spec = _infer_weight_scale_spec(weight)

    x, x_scales = quantize_dynamic_scaled_float8(
        x,
        input_scale_spec,
        weight_scale_spec,
        scales_type=weight.scale.dtype,
        out_type=weight.data.dtype,
    )

    return _fused_qkv_ragged_matmul_scaled_float8(
        kv_params,
        input=x,
        wqkv=weight.data,
        bias=bias,
        input_row_offsets=input_row_offsets,
        kv_collection=kv_collection,
        layer_idx=layer_idx,
        n_heads=n_heads,
        input_scale=x_scales.to(x.device),
        weight_scale=weight.scale.to(x.device),
        scales_granularity_mnk=_infer_scales_granularity_mnk(weight),
        _output_dim=_output_dim,
    )


# ---------------------------------------------------------------------------
# Grouped matmul (MoE)
# ---------------------------------------------------------------------------


def grouped_matmul(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    usage_stats: TensorValue,
    *,
    is_mxfp4: bool = False,
    input_scale_spec: InputScaleSpec | None = None,
    weight_scale_spec: WeightScaleSpec | None = None,
) -> TensorValue:
    """Grouped matmul for MoE layers.

    Dispatches to the MXFP4 or FP8 kernel based on ``is_mxfp4``.

    Args:
        x: Input tensor (bf16).
        weight: Expert weight tensor ``[E, ...]``.
        weight_scale: Expert weight scale tensor ``[E, ...]``.
        expert_start_indices: Starting index of each expert's tokens.
        expert_ids: Expert identifier for each token group.
        usage_stats: Per-expert usage statistics.
        is_mxfp4: Whether to use the MXFP4 dequant + matmul path.
        input_scale_spec: Input scale spec (required for FP8 path).
        weight_scale_spec: Weight scale spec (required for FP8 path).

    Returns:
        The grouped matmul output tensor (bf16).
    """
    cpu_usage_stats = usage_stats.to(DeviceRef.CPU())

    if is_mxfp4:
        dequanted = mxfp4_dequant(
            weight, weight_scale, out_type=DType.bfloat16
        )
        return grouped_matmul_ragged(
            x,
            dequanted,
            expert_start_indices,
            expert_ids,
            cpu_usage_stats,
        )
    else:
        assert input_scale_spec is not None and weight_scale_spec is not None
        assert input_scale_spec.block_size is not None
        input_block_size = input_scale_spec.block_size[1]

        weight_t = weight.transpose(1, 2)
        scale_t = weight_scale.transpose(1, 2)

        x_fp8, x_scales = quantize_dynamic_scaled_float8(
            x,
            input_scale_spec,
            weight_scale_spec,
            group_size_or_per_token=input_block_size,
            out_type=weight.dtype,
            scales_type=weight_scale_spec.dtype,
        )

        return grouped_dynamic_scaled_fp8_matmul(
            x_fp8,
            weight_t,
            x_scales,
            scale_t,
            expert_start_indices,
            expert_ids,
            cpu_usage_stats,
            input_scale_spec,
            weight_scale_spec,
        )
