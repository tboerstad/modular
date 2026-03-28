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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue

from .kernels import (
    _fused_qkv_ragged_matmul_scaled_float4,
    _fused_qkv_ragged_matmul_scaled_float8,
    block_scales_interleave,
    convert_weights_to_fp8_fnuz_if_needed,
    dynamic_block_scaled_matmul_fp4,
    dynamic_scaled_matmul,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_matmul_ragged,
    matmul_static_scaled_float8,
    mxfp4_dequant,
    quantize_dynamic_block_scaled_fp4,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
)
from .kv_cache import KVCacheParams, PagedCacheValues
from .nvfp4_tensor import Nvfp4Tensor
from .quant_config import QuantConfig, QuantFormat


def quantize_to_nvfp4(
    x: TensorValue,
    input_scale: TensorValue,
) -> Nvfp4Tensor:
    """Dynamically quantize a bf16 tensor to NVFP4.

    Args:
        x: The input tensor in bf16 with shape ``[M, K]``.
        input_scale: Scalar f32 input scale factor.

    Returns:
        An :class:`Nvfp4Tensor` holding the packed data, per-block
        scales (already interleaved), and ``input_scale`` as
        ``global_scale``.
    """
    data, scale = quantize_dynamic_block_scaled_fp4(
        x,
        tensor_sf=1.0 / input_scale,
        scales_type=DType.float8_e4m3fn,
        out_type=DType.uint8,  # fp4-e2m1fnX2
    )
    return Nvfp4Tensor(data=data, scale=scale, global_scale=input_scale)


def nvfp4_matmul(
    a: Nvfp4Tensor,
    b: Nvfp4Tensor,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Perform a matmul between two NVFP4 tensors.

    Computes ``a.data @ b.data.T`` using per-block and global scales
    from both operands.

    Args:
        a: The activation :class:`Nvfp4Tensor` (typically dynamically
            quantized).
        b: The weight :class:`Nvfp4Tensor` (block scales must already
            be interleaved).
        out_type: Output dtype (default ``bfloat16``).

    Returns:
        The result tensor of shape ``[M, N]`` in ``out_type``.
    """
    return dynamic_block_scaled_matmul_fp4(
        a.data,
        b.data,
        a.scale,
        b.scale,
        tensor_sf=b.global_scale * a.global_scale,
        out_type=out_type,
    )


def _matmul_float4(
    x: TensorValue,
    weight: Nvfp4Tensor,
    input_scale: TensorValue,
) -> TensorValue:
    """Computes x @ weight.T with modelopt NVFP4 quantization.

    Args:
        x: The input tensor in bf16.
        weight: The :class:`Nvfp4Tensor` containing the quantized weight,
            per-block scales (flat, pre-interleave), and ``global_scale``
            (``weight_scale_2``).
        input_scale: The input scale factor in f32.

    Returns:
        The output tensor in bf16.
    """
    a = quantize_to_nvfp4(x, input_scale)

    interleaved_weight_scale = block_scales_interleave(
        weight.scale.to(a.data.device),
    )
    b = Nvfp4Tensor(
        data=weight.data,
        scale=interleaved_weight_scale,
        global_scale=weight.global_scale,
    )

    return nvfp4_matmul(a, b)


def _matmul_float8(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue | None,
    quant_config: QuantConfig,
) -> TensorValue:
    """Computes x @ weight.T with float8 quantization.

    Args:
        x: The input tensor.
        weight: The weight tensor.
        weight_scale: The weight scale tensor.
        input_scale: The input scale tensor (only required for static
            fp8 quantization).
        quant_config: The quantization configuration.

    Returns:
        The output tensor.
    """
    weight, weight_scale = convert_weights_to_fp8_fnuz_if_needed(
        weight, weight_scale
    )

    if input_scale is not None:
        x = quantize_static_scaled_float8(x, input_scale, out_type=weight.dtype)

        return matmul_static_scaled_float8(x, weight, input_scale, weight_scale)
    else:
        x, x_scales = quantize_dynamic_scaled_float8(
            x,
            quant_config.input_scale,
            quant_config.weight_scale,
            scales_type=weight_scale.dtype,
            out_type=weight.dtype,
        )
        weight_scale = weight_scale.to(x.device)

        return dynamic_scaled_matmul(
            x,
            weight,
            x_scales,
            weight_scale,
            quant_config.input_scale,
            quant_config.weight_scale,
            out_type=DType.bfloat16,
        )


def quantized_matmul(
    x: TensorValue,
    quant_config: QuantConfig,
    input_scale: TensorValue | None = None,
    nvfp4_weight: Nvfp4Tensor | None = None,
    weight: TensorValue | None = None,
    weight_scale: TensorValue | None = None,
) -> TensorValue:
    """Single entry point for all quantized dense matmuls.

    Dispatches to the appropriate kernel based on the quantization format
    in ``quant_config``.

    Args:
        x: The input tensor.
        quant_config: The quantization configuration.
        input_scale: The input scale tensor (required for NVFP4 and
            static FP8).
        nvfp4_weight: The :class:`Nvfp4Tensor` for the weight
            (required for NVFP4).
        weight: The weight tensor (required for FP8).
        weight_scale: The weight scale tensor (required for FP8).

    Returns:
        The output tensor.
    """
    match quant_config.format:
        case QuantFormat.NVFP4:
            assert input_scale is not None
            assert nvfp4_weight is not None
            return _matmul_float4(
                x,
                nvfp4_weight,
                input_scale,
            )
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
            assert weight is not None
            assert weight_scale is not None
            return _matmul_float8(
                x,
                weight,
                weight_scale,
                input_scale,
                quant_config,
            )
        case _:
            raise ValueError(
                f"Unsupported quantization format for dense matmul: {quant_config.format}"
            )


def quantized_fused_qkv_matmul(
    kv_params: KVCacheParams,
    x: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    input_row_offsets: TensorValue,
    n_heads: int,
    quant_config: QuantConfig,
    input_scale: TensorValue | None = None,
    nvfp4_weight: Nvfp4Tensor | None = None,
    wqkv: TensorValue | None = None,
    weight_scale: TensorValue | None = None,
    bias: TensorValue | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Single entry point for quantized fused QKV matmuls.

    Dispatches to the NVFP4 or FP8 fused QKV kernel based on the
    quantization format in ``quant_config``.

    Args:
        kv_params: KV cache parameters.
        x: The input tensor of shape ``[total_seq_len, hidden_dim]``.
        kv_collection: The paged KV cache.
        layer_idx: The current layer index.
        input_row_offsets: Batch boundary offsets.
        n_heads: Number of attention heads.
        quant_config: The quantization configuration.
        input_scale: The input scale tensor.
        nvfp4_weight: The :class:`Nvfp4Tensor` for the QKV weight
            (required for NVFP4).
        wqkv: The concatenated QKV weight tensor (required for FP8).
        weight_scale: The weight scale tensor (required for FP8).
        bias: Optional bias tensor (FP8 only).
        _output_dim: Optional output dimension override for the FP8
            kernel. If not provided, defaults to
            ``n_heads * head_dim``.

    Returns:
        The query projection output tensor.
    """
    match quant_config.format:
        case QuantFormat.NVFP4:
            assert input_scale is not None
            assert nvfp4_weight is not None

            a = quantize_to_nvfp4(x, input_scale)

            interleaved_weight_scale = block_scales_interleave(
                nvfp4_weight.scale.to(a.data.device),
            )

            return _fused_qkv_ragged_matmul_scaled_float4(
                kv_params,
                input=a.data,
                input_row_offsets=input_row_offsets,
                wqkv=nvfp4_weight.data,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=n_heads,
                input_scale=a.scale.to(a.data.device),
                weight_scale=interleaved_weight_scale,
                tensor_sf=a.global_scale * nvfp4_weight.global_scale,
            )
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
            assert wqkv is not None
            assert weight_scale is not None
            # FP8 path (static or dynamic)
            if quant_config.is_static:
                assert input_scale is not None
                x = quantize_static_scaled_float8(
                    x, input_scale.to(DeviceRef.CPU())
                )
                x_scales = input_scale
            else:
                x, x_scales = quantize_dynamic_scaled_float8(
                    x,
                    quant_config.input_scale,
                    quant_config.weight_scale,
                    scales_type=weight_scale.dtype,
                    out_type=wqkv.dtype,
                )

            return _fused_qkv_ragged_matmul_scaled_float8(
                kv_params,
                input=x,
                wqkv=wqkv,
                bias=bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale.to(x.device),
                quant_config=quant_config,
                _output_dim=_output_dim,
            )
        case _:
            raise ValueError(
                f"Unsupported quantization format for fused QKV matmul: {quant_config.format}"
            )


def quantized_grouped_matmul(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    usage_stats: TensorValue,
    quant_config: QuantConfig,
) -> TensorValue:
    """Single entry point for quantized grouped matmuls (MoE).

    Dispatches to the appropriate kernel based on the quantization format.
    Handles weight dequant (MXFP4) or input quantization + transpose (FP8)
    internally.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in storage layout
            (MXFP4: ``[E, out, in//2]``, FP8: ``[E, in, out]``).
        weight_scale: The weight scale tensor in storage layout.
        expert_start_indices: Starting index of each expert's token group.
        expert_ids: Expert identifier for each token group.
        usage_stats: Per-expert usage statistics (will be moved to CPU).
        quant_config: The quantization configuration.

    Returns:
        The grouped matmul output tensor in bf16.
    """
    cpu_usage_stats = usage_stats.to(DeviceRef.CPU())

    match quant_config.format:
        case QuantFormat.MXFP4:
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
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
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
        case _:
            raise ValueError(
                f"Unsupported quantization format for grouped matmul: {quant_config.format}"
            )
