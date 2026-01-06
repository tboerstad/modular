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
"""Gemma3 vision attention layers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue
from max.nn.attention.multihead_attention import MultiheadAttention


class Gemma3VisionAttention(MultiheadAttention):
    """Gemma3 vision multi-head attention layer.

    Multi-headed attention from 'Attention Is All You Need' paper,
    adapted for SigLIP vision encoder component.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        has_bias: bool = True,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.bfloat16,
    ) -> None:
        head_dim = hidden_size // num_attention_heads
        devices_list = list(devices) if devices else []

        super().__init__(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            devices=devices_list if devices_list else None,
            dtype=dtype,
            scale=head_dim ** (-0.5),
            qkv_has_bias=has_bias,
            o_proj_has_bias=has_bias,
            stacked_qkv=False,
        )

    def __call__(self, x: TensorValue, **kwargs) -> TensorValue:
        return self._forward_single(x)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.q_proj.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.q_proj.sharding_strategy = strategy
        self.k_proj.sharding_strategy = strategy
        self.v_proj.sharding_strategy = strategy
        self.o_proj.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Gemma3VisionAttention]:
        devices_list = list(devices)
        q_shards = self.q_proj.shard(devices_list)
        k_shards = self.k_proj.shard(devices_list)
        v_shards = self.v_proj.shard(devices_list)
        o_shards = self.o_proj.shard(devices_list)

        shards = []
        for device, q, k, v, o in zip(
            devices_list, q_shards, k_shards, v_shards, o_shards, strict=True
        ):
            s = Gemma3VisionAttention(
                self.embed_dim, self.num_heads, self.qkv_has_bias, [device], q.weight.dtype
            )
            s.q_proj, s.k_proj, s.v_proj, s.o_proj = q, k, v, o
            shards.append(s)
        return shards
