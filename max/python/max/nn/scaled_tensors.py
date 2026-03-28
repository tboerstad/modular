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
"""Scaled tensor representations for quantized inference.

Type hierarchy::

    ScaledTensor
    ├── Float8Tensor      # FP8 (static or dynamic, per-tensor/row/block)
    ├── Nvfp4Tensor       # NVFP4 (packed uint8 + block scales + global scale)
    └── Mxfp4Tensor       # MXFP4 (packed uint8 + block scales)
"""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue


@dataclass
class ScaledTensor:
    """A tensor paired with a per-block or per-tensor scale.

    This is the abstract base representation for quantized weights.
    Use one of the concrete subclasses (:class:`Float8Tensor`,
    :class:`Nvfp4Tensor`, :class:`Mxfp4Tensor`) to clearly express
    the quantization format.

    Attributes:
        data: The quantized weight tensor.
        scale: Scaling factors whose granularity depends on the
            quantization scheme (scalar, row-wise, or block-wise).
    """

    data: TensorValue
    scale: TensorValue


@dataclass
class Float8Tensor(ScaledTensor):
    """An FP8 quantized tensor with per-tensor, per-row, or per-block scales.

    Covers the ``COMPRESSED_TENSORS_FP8``, ``FBGEMM_FP8``, and
    ``BLOCKSCALED_FP8`` quantization formats.

    Attributes:
        data: The weight tensor in ``float8_e4m3fn`` (or
            ``float8_e4m3fnuz`` after conversion for AMD GPUs).
        scale: Scaling factors in ``float32`` (or ``float8_e4m3fn``
            for block-scaled).  Granularity is per-tensor (scalar),
            per-row ``[N, 1]``, or per-block ``[N, K // block_k]``.
    """

    pass


@dataclass
class Nvfp4Tensor(ScaledTensor):
    """A quantized NVFP4 tensor with an additional tensor-wide scale.

    Extends :class:`ScaledTensor` with a ``global_scale`` field that
    carries the second-level scaling factor required by the NVFP4
    format.

    Attributes:
        data: Packed uint8 tensor where each byte stores two
            ``fp4-e2m1fn`` values.  Shape is ``[M, K//2]`` for a
            logical ``[M, K]`` tensor.
        scale: Per-block scaling factors in ``float8_e4m3fn``.  Stored
            either in the 5-D interleaved layout
            ``[ceildiv(M,128), ceildiv(K,64), 32, 4, 4]`` expected by
            the GPU kernels, or in the flat ``[M, K//16]`` layout
            before interleaving (for weights loaded from disk).
        global_scale: A scalar ``float32`` tensor representing the
            tensor-wide scale.  For weights this corresponds to
            ``weight_scale_2``; for dynamically quantized activations
            it is the ``input_scale``.
    """

    global_scale: TensorValue


@dataclass
class Mxfp4Tensor(ScaledTensor):
    """An MXFP4 (Microscaling FP4) quantized tensor.

    Used in MoE (Mixture of Experts) layers where weights are stored
    in a packed format with per-block scales.

    Attributes:
        data: Packed uint8 tensor where each byte stores two FP4
            values.  Shape is ``[N, K//2]`` (dense) or
            ``[E, N, K//2]`` (MoE with E experts).
        scale: Per-block scaling factors in ``float8_e8m0fnu``.
            Shape is ``[N, K//32]`` (dense) or
            ``[E, N, K//32]`` (MoE).
    """

    pass
