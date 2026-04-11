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

"""GptOss Attention Layer — thin subclass adding learnable sink weights."""

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, Weight
from max.nn.attention import AttentionWithRope


class GptOssAttention(AttentionWithRope):
    """AttentionWithRope extended with learnable attention sink weights.

    Sink attention adds an extra logit column per head that acts as an
    attention sink, improving attention quality for long sequences.  Only the
    ``sinks`` weight and its sharding are handled here; everything else
    (projections, QKV matmul, rope, flash-attention, sharding boilerplate)
    is inherited from :class:`AttentionWithRope`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sinks = Weight(
            name="sinks",
            dtype=kwargs.get("dtype", DType.float32),
            shape=[self.n_heads],
            device=self.devices[0],
        )

    @AttentionWithRope.sharding_strategy.setter  # type: ignore[attr-defined]
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        # Delegate all standard weight sharding to the base class.
        AttentionWithRope.sharding_strategy.fset(self, strategy)  # type: ignore[union-attr]
        # Additionally shard the sinks weight.
        if strategy.is_tensor_parallel:
            self.sinks.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
        elif strategy.is_replicate:
            self.sinks.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[AttentionWithRope]:
        """Shard and propagate sink weights to each device shard."""
        devices = list(devices)
        shards = super().shard(devices)
        sinks_shards = self.sinks.shard(devices)
        for i, shard in enumerate(shards):
            shard.sinks = sinks_shards[i]
        return shards
