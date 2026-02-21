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

"""GptOss Attention Layer."""

from __future__ import annotations

from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams

from ...common_layers.attention import AttentionWithRope
from ...common_layers.rotary_embedding import YarnRotaryEmbedding


class GptOssAttention(AttentionWithRope):
    """Implementation of the attention layer for the GptOss text model.

    Depending on the layer type, the attention layer can be either a full
    attention layer or a sliding window attention layer. This layer generates
    the attention mask based on the layer type.

    This layer also supports sink attention, which is a technique to improve
    the attention mechanism by adding an extra logit column that acts as an
    attention sink.
    """

    def __init__(
        self,
        *,
        rope: YarnRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        mask_variant: MHAMaskVariant,
        scale: float | None = None,
        has_bias: bool = False,
        local_window_size: int = 1024,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: Rotary embedding used for all attention layers
                (full + sliding window).
            num_attention_heads: The number of attention heads.
            num_key_value_heads: The number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the
                head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            mask_variant: The mask variant for attention
                (causal or sliding window).
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias. Defaults to False.
            local_window_size: Size of the sliding window. Defaults to 1024.
        """
        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            layer_idx=layer_idx,
            scale=scale,
            has_bias=has_bias,
            mask_variant=mask_variant,
            local_window_size=local_window_size,
            use_sinks=True,
            o_proj_has_bias=has_bias,
        )
