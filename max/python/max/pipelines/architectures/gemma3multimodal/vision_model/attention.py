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
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue
from max.nn.attention.multihead_attention import MultiheadAttention
from max.nn.linear import Linear


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
        """Initialize Gemma3 vision attention layer.

        Args:
            hidden_size: The dimension of the hidden states (embed_dim).
            num_attention_heads: The number of attention heads.
            has_bias: Whether to use bias in QKV and output projections.
            devices: Device(s) to place the weights and run the computation.
                If multiple devices provided, uses distributed computation.
            dtype: DType of the QKV and output projection weights.

        Raises:
            ValueError: If hidden_size is not divisible by num_attention_heads.
        """
        # Validate that embed_dim is divisible by num_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads "
                f"(got `hidden_size`: {hidden_size} and `num_attention_heads`: "
                f"{num_attention_heads})."
            )

        head_dim = hidden_size // num_attention_heads
        # Scale factor for attention: 1/sqrt(head_dim)
        scale = head_dim ** (-0.5)

        devices_list = list(devices) if devices else []

        super().__init__(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            devices=devices_list if devices_list else None,
            dtype=dtype,
            scale=scale,
            qkv_has_bias=has_bias,
            o_proj_has_bias=has_bias,
            stacked_qkv=False,
        )

        self.is_causal = False  # Vision attention is not causal

        # Override the output projection with PyTorch-compatible naming
        self._init_pytorch_compatible_weights(dtype)

    def _init_pytorch_compatible_weights(self, dtype: DType) -> None:
        """Initialize output projection with PyTorch-compatible naming."""
        # Replace o_proj with out_proj to match PyTorch naming
        self.out_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            has_bias=self.o_proj_has_bias,
            dtype=dtype,
            device=self.devices[0],
        )
        # Remove the original o_proj to avoid conflicts
        if hasattr(self, "o_proj"):
            delattr(self, "o_proj")

    def _forward_single(self, x: TensorValue, **kwargs) -> TensorValue:
        """Single-device forward pass with PyTorch-compatible naming.

        Override to use out_proj instead of o_proj.
        """
        # Compute QKV
        q, k, v = self._compute_qkv(x)

        # Apply attention
        attn_out = self._apply_attention(q, k, v, **kwargs)

        # Output projection using PyTorch-compatible naming
        return self.out_proj(attn_out)

    def __call__(  # type: ignore[override]
        self,
        x: TensorValue,
        signal_buffers: None = None,
        **kwargs: Any,
    ) -> TensorValue:
        """Forward pass of Gemma3 vision attention.

        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size].
            signal_buffers: Not used in vision attention (set to None).
            **kwargs: Additional arguments, including optional attention_mask.

        Returns:
            attention_output: Output tensor of shape [batch_size, seq_length, hidden_size].
        """
        # Extract attention_mask from kwargs if provided
        attention_mask = kwargs.get("attention_mask")

        # Use our custom _forward_single method for the core attention computation
        attn_output = self._forward_single(x, attention_mask=attention_mask)

        return attn_output

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.q_proj.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_replicate:
            raise ValueError(
                "only replicate is currently supported for Gemma3VisionAttention"
            )

        self.q_proj.sharding_strategy = strategy
        self.k_proj.sharding_strategy = strategy
        self.v_proj.sharding_strategy = strategy
        self.out_proj.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma3VisionAttention]:
        """Shard the attention module across multiple devices.

        Args:
            devices: Iterable of devices to shard across.

        Returns:
            List of Gemma3VisionAttention instances, one per device.
        """
        assert self.sharding_strategy

        devices_list = list(devices)
        q_proj_shards = self.q_proj.shard(devices_list)
        k_proj_shards = self.k_proj.shard(devices_list)
        v_proj_shards = self.v_proj.shard(devices_list)
        out_proj_shards = self.out_proj.shard(devices_list)

        shards = []
        for device, q_shard, k_shard, v_shard, out_shard in zip(
            devices_list,
            q_proj_shards,
            k_proj_shards,
            v_proj_shards,
            out_proj_shards,
            strict=True,
        ):
            sharded = Gemma3VisionAttention(
                hidden_size=self.embed_dim,
                num_attention_heads=self.num_heads,
                has_bias=self.qkv_has_bias,
                devices=[device],
                dtype=q_shard.weight.dtype,
            )

            sharded.q_proj = q_shard
            sharded.k_proj = k_shard
            sharded.v_proj = v_shard
            sharded.out_proj = out_shard

            shards.append(sharded)

        return shards
