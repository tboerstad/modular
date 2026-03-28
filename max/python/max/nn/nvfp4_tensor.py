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
"""Scaled tensor representations for quantized inference."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue


@dataclass
class ScaledTensor:
    """A tensor paired with a per-block or per-tensor scale.

    This is the base representation for quantized weights.  For FP8
    quantization the two fields are sufficient; NVFP4 extends this with
    an additional global scale via :class:`Nvfp4Tensor`.

    Attributes:
        data: The quantized weight tensor (e.g. ``float8_e4m3fn`` for
            FP8, ``uint8`` for packed NVFP4).
        scale: Scaling factors whose granularity depends on the
            quantization scheme (scalar, row-wise, or block-wise).
    """

    data: TensorValue
    scale: TensorValue


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
