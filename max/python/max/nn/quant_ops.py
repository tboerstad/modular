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
from .quant_config import QuantConfig
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
    quant_config: QuantConfig,
    bias: TensorValue | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Fused QKV matmul + KV-cache store (FP8).

    The kernel handles input quantisation internally.

    Args:
        kv_params: KV cache parameters.
        x: Input tensor ``[total_seq_len, hidden_dim]`` (bf16).
        weight: :class:`Float8Tensor` holding the concatenated QKV
            weight and its scales.
        kv_collection: The paged KV cache.
        layer_idx: Current layer index.
        input_row_offsets: Batch boundary offsets.
        n_heads: Number of attention heads.
        quant_config: Quantisation configuration (used for scale specs).
        bias: Optional bias tensor.
        _output_dim: Optional output dimension override.

    Returns:
        The query projection output tensor.
    """
    x, x_scales = quantize_dynamic_scaled_float8(
        x,
        quant_config.input_scale,
        quant_config.weight_scale,
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
        quant_config=quant_config,
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
    quant_config: QuantConfig,
) -> TensorValue:
    """Grouped matmul for MoE layers.

    Dispatches to the MXFP4 or FP8 kernel based on ``quant_config``.

    Args:
        x: Input tensor (bf16).
        weight: Expert weight tensor ``[E, ...]``.
        weight_scale: Expert weight scale tensor ``[E, ...]``.
        expert_start_indices: Starting index of each expert's tokens.
        expert_ids: Expert identifier for each token group.
        usage_stats: Per-expert usage statistics.
        quant_config: Quantisation configuration.

    Returns:
        The grouped matmul output tensor (bf16).
    """
    cpu_usage_stats = usage_stats.to(DeviceRef.CPU())

    if quant_config.is_mxfp4:
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
        assert quant_config.input_scale.block_size is not None
        input_block_size = quant_config.input_scale.block_size[1]

        weight_t = weight.transpose(1, 2)
        scale_t = weight_scale.transpose(1, 2)

        x_fp8, x_scales = quantize_dynamic_scaled_float8(
            x,
            quant_config.input_scale,
            quant_config.weight_scale,
            group_size_or_per_token=input_block_size,
            out_type=weight.dtype,
            scales_type=quant_config.weight_scale.dtype,
        )

        return grouped_dynamic_scaled_fp8_matmul(
            x_fp8,
            weight_t,
            x_scales,
            scale_t,
            expert_start_indices,
            expert_ids,
            cpu_usage_stats,
            quant_config.input_scale,
            quant_config.weight_scale,
        )
