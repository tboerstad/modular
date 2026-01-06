# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Utilities for distributed/multi-GPU computation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from max.graph import DeviceRef, TensorValue


def distribute_value(
    v: TensorValue, devices: Sequence[DeviceRef]
) -> list[TensorValue]:
    """Distributes a tensor value across multiple devices.

    Creates a copy of the tensor on each device in the provided sequence.
    This is commonly used to replicate values like position embeddings
    or other shared tensors across devices in tensor-parallel setups.

    Args:
        v: The tensor value to distribute.
        devices: The sequence of devices to distribute the tensor across.

    Returns:
        A list of tensor values, one per device.
    """
    return [v.to(device) for device in devices]


def forward_sharded_layers(
    layers: Sequence[Callable[[TensorValue], TensorValue]],
    xs: Sequence[TensorValue],
) -> list[TensorValue]:
    """Forward pass through sharded layers.

    Applies each layer to its corresponding input tensor. This is useful
    for tensor-parallel execution where each device has a shard of the
    layer and processes its local input.

    Args:
        layers: Sequence of callable layers that return TensorValue.
        xs: Input tensors, one per layer.

    Returns:
        List of output tensors from each layer.

    Raises:
        AssertionError: If the number of layers and input tensors don't match.
    """
    assert len(xs) == len(layers), (
        f"Number of layers ({len(layers)}) must match number of inputs ({len(xs)})"
    )
    return [layer(x) for layer, x in zip(layers, xs, strict=True)]
