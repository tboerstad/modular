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

"""Olmo3 Attention Layer."""

from __future__ import annotations

from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.tensor import Tensor

from ...common_layers.attention import AttentionWithRope
from ...common_layers.functional_kernels import rms_norm_key_cache
from ...common_layers.rotary_embedding import RotaryEmbedding
from .rms_norm import Olmo3RMSNorm


class Olmo3Attention(AttentionWithRope):
    """Implementation of the attention layer for the Olmo3 text model.

    Depending on the layer type, the attention layer can be either a full
    attention layer or a sliding window attention layer.

    Olmo3 includes Q and K normalization after the Q and K projections,
    using full-dimension RMSNorm (not per-head).
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
        mask_variant: MHAMaskVariant,
        has_bias: bool = False,
        local_window_size: int = 4096,
        use_qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: Rotary embedding used for the attention layer. Basic RoPE
                for sliding attention, YARN RoPE for full attention.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: The number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params.
            layer_idx: The layer number associated with this Attention block.
            mask_variant: The mask variant for attention
                (causal or sliding window).
            has_bias: Whether to use an attention bias.
            local_window_size: Size of the sliding window.
            use_qk_norm: Whether to use Q and K normalization.
            qk_norm_eps: Epsilon value for Q and K normalization.
        """
        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            layer_idx=layer_idx,
            has_bias=has_bias,
            mask_variant=mask_variant,
            local_window_size=local_window_size,
            use_qk_norm=use_qk_norm,
            rms_norm_eps=qk_norm_eps,
            o_proj_has_bias=has_bias,
        )

    def _init_qk_norm(self) -> None:
        """Olmo3 uses full-dimension RMSNorm (not per-head)."""
        self.q_norm = Olmo3RMSNorm(self.q_weight_dim, self.rms_norm_eps)
        self.k_norm = Olmo3RMSNorm(self.kv_weight_dim, self.rms_norm_eps)

    def _apply_q_norm(self, xq: Tensor) -> Tensor:
        """Apply full-dimension RMSNorm to Q, then reshape back to per-head."""
        total_seq_len = xq.shape[0]
        q = xq.reshape(
            (total_seq_len, self.n_heads * self.kv_params.head_dim)
        )
        q = self.q_norm(q)
        return q.reshape(
            (total_seq_len, self.n_heads, self.kv_params.head_dim)
        )

    def _apply_k_norm_to_cache(
        self,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        total_seq_len: int,
        input_row_offsets: Tensor,
        device,
    ) -> None:
        """Apply full-dimension RMSNorm to K in the KV cache."""
        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(device),
            epsilon=self.rms_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
            per_head_norm=False,
        )
