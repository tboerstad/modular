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
"""NVFP4 quantized tensor representation."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue


@dataclass
class Nvfp4Tensor:
    """A quantized NVFP4 tensor bundling data with its scaling factors.

    This class groups the packed FP4 data, per-block scales, and the
    tensor-wide global scale into a single object so that callers do not
    need to pass them individually.

    Attributes:
        data: Packed uint8 tensor where each byte stores two
            ``fp4-e2m1fn`` values.  Shape is ``[M, K//2]`` for a logical
            ``[M, K]`` tensor.
        scale: Per-block scaling factors in ``float8_e4m3fn``.  The
            tensor is stored in the 5-D interleaved layout
            ``[ceildiv(M,128), ceildiv(K,64), 32, 4, 4]`` expected by
            the GPU kernels, **or** in the flat ``[M, K//16]`` layout
            before interleaving (for weights loaded from disk).
        global_scale: A scalar ``float32`` tensor representing the
            tensor-wide scale.  For weights this corresponds to
            ``weight_scale_2``; for dynamically quantized activations it
            is the ``input_scale``.
    """

    data: TensorValue
    scale: TensorValue
    global_scale: TensorValue
