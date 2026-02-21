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

"""Olmo3 attention with full-dimension QK normalization."""

from __future__ import annotations

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import PagedCacheValues
from max.tensor import Tensor

from ...common_layers.attention import AttentionWithRope
from ...common_layers.functional_kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from ...common_layers.rotary_embedding import RotaryEmbedding


class Olmo3Attention(AttentionWithRope):
    """AttentionWithRope variant using full-dimension QK normalization.

    OLMo3 normalizes Q and K across the full projection dimension
    (num_heads * head_dim) rather than per-head, and supports mixed
    attention (full causal + sliding window) across layers.
    """

    def __init__(
        self,
        *,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
        local_window_size: int = 4096,
        qk_norm_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        # Initialize base without QK norm — we manage norm weights ourselves.
        super().__init__(**kwargs, use_qk_norm=False)
        self.mask_variant = mask_variant
        self.local_window_size = local_window_size
        self.qk_norm_eps = qk_norm_eps

        # Full-dimension QK norm: gamma spans all heads.
        self.q_norm_weight = Tensor.ones([self.q_weight_dim])
        kv_weight_dim = self.kv_params.head_dim * self.num_key_value_heads
        self.k_norm_weight = Tensor.ones([kv_weight_dim])

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        input_row_offsets = kwargs["input_row_offsets"]

        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Full-dimension QK norm: normalize Q across all heads together.
        xq = xq.reshape((-1, self.n_heads * self.kv_params.head_dim))
        q_gamma = F.cast(self.q_norm_weight.to(xq.device), xq.dtype)
        eps = F.constant(self.qk_norm_eps, xq.dtype, device=xq.device)
        inv_rms = F.rsqrt(F.mean(xq * xq, axis=-1) + eps)
        xq = (xq * inv_rms) * q_gamma
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Normalize K in the KV cache across the full dimension.
        rms_norm_key_cache(
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm_weight.cast(self.kv_params.dtype).to(
                xq.device
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
            per_head_norm=False,
        )

        freqs_cis = F.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
        )

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=self.mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, self.q_weight_dim])
        return self.o_proj(attn_out)
