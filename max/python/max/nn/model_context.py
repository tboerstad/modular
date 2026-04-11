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

"""Model context for bundling common layer configuration parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationEncoding
from max.nn.quant_config import QuantConfig


@dataclass(frozen=True)
class ModelContext:
    """Bundles common model configuration passed to layer constructors.

    Inspired by MLX's approach of setting device and dtype context once,
    ``ModelContext`` lets you define these values in one place and pass them
    through the model hierarchy instead of repeating ``dtype``, ``device``,
    ``quantization_encoding``, and ``quant_config`` on every layer.

    Individual constructor parameters still override context values when
    both are provided, so existing code works unchanged.

    Example:

    .. code-block:: python

        from max.nn import ModelContext, Linear, MLP

        ctx = ModelContext(
            dtype=DType.bfloat16,
            device=DeviceRef.GPU(),
        )

        # Instead of: Linear(256, 128, dtype=DType.bfloat16, device=DeviceRef.GPU())
        layer = Linear(256, 128, ctx=ctx)

    Args:
        dtype: The data type for weights and computation.
        device: The primary device for computation.
        devices: Optional sequence of devices for distributed/sharded layers.
            When set, ``device`` is typically ``devices[0]``.
        quantization_encoding: Optional quantization encoding for weights.
        quant_config: Optional quantization configuration for scaled quantization.
    """

    dtype: DType
    device: DeviceRef
    devices: Sequence[DeviceRef] | None = None
    quantization_encoding: QuantizationEncoding | None = None
    quant_config: QuantConfig | None = None

    @property
    def primary_device(self) -> DeviceRef:
        """Returns the first device from ``devices``, falling back to ``device``."""
        if self.devices:
            return self.devices[0]
        return self.device

    @property
    def all_devices(self) -> Sequence[DeviceRef]:
        """Returns ``devices`` if set, otherwise a single-element sequence of ``device``."""
        if self.devices:
            return self.devices
        return [self.device]
