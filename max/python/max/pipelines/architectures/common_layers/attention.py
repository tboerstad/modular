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

"""An opaque KV Cache optimized attention mechanism with RoPE (ModuleV3)."""

from __future__ import annotations

import math

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues, uses_opaque
from max.tensor import Tensor

from .functional_kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from .rotary_embedding import RotaryEmbedding


class AttentionWithRope(Module[..., Tensor]):
    """Implementation of attention that uses Rotary Position Embedding (RoPE).

    This is a ModuleV3 port of the legacy AttentionWithRope class. It supports
    both separate and stacked QKV projections, optional clip_qkv clamping,
    optional QK normalization via RMSNorm, configurable attention masks, sliding
    window attention, and attention sinks.

    Subclasses can override the following hook methods to customize behavior:

    - ``_init_qk_norm``: Controls how QK norm weights/modules are created.
    - ``_apply_q_norm``: Controls how Q normalization is applied.
    - ``_apply_k_norm_to_cache``: Controls how K normalization is applied
      to the KV cache.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        scale: float | None = None,
        has_bias: bool = False,
        stacked_qkv: bool = False,
        clip_qkv: float | None = None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
        local_window_size: int = 0,
        use_sinks: bool = False,
        o_proj_has_bias: bool = False,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head
                dim, and dtype.
            layer_idx: The layer number associated with this Attention block.
            scale: Optional attention scale; defaults to sqrt(1/head_dim).
            has_bias: Whether Q/K/V have bias (stacked_qkv forbids bias).
            stacked_qkv: Whether Q/K/V weights are stacked in a single weight.
            clip_qkv: If provided, clamp Q/K/V weights to
                ``[-clip_qkv, clip_qkv]``.
            use_qk_norm: Whether to use RMSNorm on Q/K.
            rms_norm_eps: Value to use for numerical stability in RMSNorm.
            mask_variant: Attention mask type (causal, sliding window, etc.).
            local_window_size: Size of the sliding window for local attention.
                Only used when mask_variant is SLIDING_WINDOW_CAUSAL_MASK.
                A value of 0 means no window limit.
            use_sinks: Whether to use attention sinks. When True, creates a
                learnable sinks parameter that adds an extra logit column
                acting as an attention sink.
            o_proj_has_bias: Whether the output projection has a bias term.
        """
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.clip_qkv = clip_qkv
        self.stacked_qkv = stacked_qkv
        self.use_qk_norm = use_qk_norm
        self.rms_norm_eps = rms_norm_eps
        self.mask_variant = mask_variant
        self.local_window_size = local_window_size
        self.use_sinks = use_sinks

        if stacked_qkv and clip_qkv:
            raise ValueError(
                "`clip_qkv` not yet supported when `stacked_qkv=True`."
            )

        if stacked_qkv and has_bias:
            raise ValueError("Bias is not supported with stacked_qkv.")

        if not uses_opaque(self.kv_params.cache_strategy):
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy is not"
                " supported in the Attention layer."
            )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_weight_dim
        self.kv_weight_dim = kv_weight_dim

        if stacked_qkv:
            self.qkv_proj = Tensor.zeros(
                [q_weight_dim + 2 * kv_weight_dim, hidden_size]
            )
        else:
            self.q_proj = Linear(
                in_dim=hidden_size,
                out_dim=q_weight_dim,
                bias=has_bias,
            )
            self.k_proj = Linear(
                in_dim=hidden_size,
                out_dim=kv_weight_dim,
                bias=has_bias,
            )
            self.v_proj = Linear(
                in_dim=hidden_size,
                out_dim=kv_weight_dim,
                bias=has_bias,
            )

        self.o_proj = Linear(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            bias=o_proj_has_bias,
        )

        if self.use_qk_norm:
            self._init_qk_norm()

        if self.use_sinks:
            self.sinks = Tensor.zeros([num_attention_heads])

    def _init_qk_norm(self) -> None:
        """Initialize QK normalization weights.

        Override this method to use a different normalization scheme (e.g.,
        full-dimension RMSNorm instead of per-head RMSNorm).
        """
        self.q_norm_weight = Tensor.ones([self.kv_params.head_dim])
        self.k_norm_weight = Tensor.ones([self.kv_params.head_dim])

    def _apply_q_norm(self, xq: Tensor) -> Tensor:
        """Apply normalization to Q.

        Args:
            xq: Query tensor of shape [total_seq_len, n_heads, head_dim].

        Returns:
            Normalized query tensor of the same shape.
        """
        q_gamma = F.cast(self.q_norm_weight.to(xq.device), xq.dtype)
        eps_q = F.constant(self.rms_norm_eps, xq.dtype, device=xq.device)
        inv_rms = F.rsqrt(F.mean(xq * xq, axis=-1) + eps_q)
        return (xq * inv_rms) * q_gamma

    def _apply_k_norm_to_cache(
        self,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        total_seq_len: int,
        input_row_offsets: Tensor,
        device,
    ) -> None:
        """Apply normalization to K entries in the KV cache (in-place).

        Args:
            kv_collection: The paged KV cache.
            layer_idx: Layer index constant.
            total_seq_len: Total sequence length across the batch.
            input_row_offsets: Ragged offsets for batched sequences.
            device: Device to place the gamma tensor on.
        """
        rms_norm_key_cache(
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm_weight.cast(self.kv_params.dtype).to(device),
            epsilon=self.rms_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
        )

    @property
    def wqkv(self) -> Tensor:
        """The concatenation of q, k, and v weight vectors."""
        if self.stacked_qkv:
            return self.qkv_proj
        else:
            wq: Tensor = self.q_proj.weight
            wk: Tensor = self.k_proj.weight
            wv: Tensor = self.v_proj.weight
            if self.clip_qkv:
                wq = F.min(F.max(wq, -self.clip_qkv), self.clip_qkv)
                wk = F.min(F.max(wk, -self.clip_qkv), self.clip_qkv)
                wv = F.min(F.max(wv, -self.clip_qkv), self.clip_qkv)
            return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None
        assert not self.stacked_qkv

        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]

        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if self.use_qk_norm:
            self._apply_k_norm_to_cache(
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                total_seq_len=total_seq_len,
                input_row_offsets=kwargs["input_row_offsets"],
                device=xq.device,
            )
            xq = self._apply_q_norm(xq)

        freqs_cis = F.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
        )

        flash_kwargs: dict = {}
        if self.local_window_size > 0:
            flash_kwargs["local_window_size"] = self.local_window_size
        if self.use_sinks:
            flash_kwargs["sink_weights"] = self.sinks

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=self.mask_variant,
            scale=self.scale,
            **flash_kwargs,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
