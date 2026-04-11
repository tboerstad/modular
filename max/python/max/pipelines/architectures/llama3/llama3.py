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
"""Build a Llama3 model that uses continuous or paged kv-caching"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.graph.quantization import QuantizationEncoding
from max.nn.attention import (
    AttentionWithRope,
    GGUFQAttentionWithRope,
    GPTQAttentionWithRope,
)
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.layer import Module
from max.nn.linear import MLP, GPTQLinear, Linear
from max.nn.lora import AttentionWithRopeAndLoRA
from max.nn.model_context import ModelContext
from max.nn.norm import ConstantLayerNorm, RMSNorm
from max.nn.transformer import Transformer, TransformerBlock
from max.pipelines.lib.lora import LoRAManager

from .model_config import Llama3Config, create_rope_embedding


class StackedMLP(Module):
    def __init__(
        self,
        dtype: DType | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
        hidden_dim: int = 0,
        feed_forward_length: int = 0,
        devices: Sequence[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        has_scale: bool = False,
        ctx: ModelContext | None = None,
    ) -> None:
        super().__init__()

        if ctx is not None:
            dtype = dtype or ctx.dtype
            devices = devices or list(ctx.all_devices)
            if quantization_encoding is None:
                quantization_encoding = ctx.quantization_encoding
        if dtype is None:
            raise TypeError(
                "dtype must be provided either directly or via ctx"
            )
        if devices is None:
            raise TypeError(
                "devices must be provided either directly or via ctx"
            )

        self.gate_up_proj = linear_cls(
            in_dim=hidden_dim,
            out_dim=feed_forward_length * 2,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
        )
        self.down_proj = linear_cls(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        up_states = self.gate_up_proj(x)

        gate = up_states[:, : up_states.shape.static_dims[0] // 2]
        up_states = up_states[:, up_states.shape.static_dims[0] // 2 :]

        return self.down_proj(ops.silu(gate) * up_states)


class Llama3(Transformer):
    def __init__(self, config: Llama3Config) -> None:
        assert len(config.devices) == 1
        self.config = config
        rope = create_rope_embedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved_rope_weights=config.interleaved_rope_weights,
            rope_scaling_params=config.rope_scaling_params,
            longrope_scaling_params=config.longrope_scaling_params,
            device=config.devices[0],
        )

        # Select norm layer class.
        create_norm: Callable[..., Module]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNorm,
                config.hidden_size,
                config.norm_dtype or config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,  # disable Gemma3-style scaling
            )
        else:
            create_norm = functools.partial(
                ConstantLayerNorm,
                config.hidden_size,
                config.devices[0],
                config.norm_dtype or config.dtype,
            )

        # Bundle common model parameters into a single context, inspired by
        # MLX's approach of setting dtype/device once for the model hierarchy.
        ctx = ModelContext(
            dtype=config.dtype,
            device=config.devices[0],
            devices=config.devices,
            quantization_encoding=config.model_quantization_encoding,
            quant_config=config.quant_config,
        )

        # Select linear layer class.
        linear_cls: Callable[..., Linear]
        if config.quantization_config:
            linear_cls = functools.partial(
                GPTQLinear, quantization_config=config.quantization_config
            )
        else:
            linear_cls = functools.partial(
                Linear, quant_config=config.quant_config
            )
        if config.stacked_mlp and config.quant_config:
            raise ValueError(
                "StackedMLP and scaled quantization are not compatible"
            )
        # MLP class selection: ctx carries dtype, devices, quantization, and
        # quant_config so the partial wrapper is no longer needed.
        mlp_cls = StackedMLP if config.stacked_mlp else MLP

        attention_cls: Callable[..., AttentionWithRope]
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            assert config.quantization_config is not None
            assert not config.attention_bias, (
                "Attention bias is not supported for GPTQAttentionWithRope."
            )
            attention_cls = functools.partial(
                GPTQAttentionWithRope,
                quantization_config=config.quantization_config,
                scale=config.attention_multiplier,
            )
        elif config.model_quantization_encoding is not None:
            assert not config.attention_bias, (
                "Attention bias is not supported for GGUFQAttentionWithRope."
            )
            attention_cls = functools.partial(
                GGUFQAttentionWithRope,
                quantization_encoding=config.model_quantization_encoding,
                scale=config.attention_multiplier,
            )
        elif config.lora_config is not None:
            attention_cls = functools.partial(
                AttentionWithRopeAndLoRA,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                clip_qkv=config.clip_qkv,
                has_bias=config.attention_bias,
                max_num_loras=config.lora_config.max_num_loras,
                max_lora_rank=config.lora_config.max_lora_rank,
                quant_config=config.quant_config,
            )
        else:
            # ctx carries quant_config, so it can be omitted from the partial.
            attention_cls = functools.partial(
                AttentionWithRope,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                clip_qkv=config.clip_qkv,
                has_bias=config.attention_bias,
                ctx=ctx,
            )

        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    linear_cls=linear_cls,
                    ctx=ctx,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_output_dtype = config.quant_config.embedding_output_dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            rope=rope,
            return_logits=config.return_logits,
            return_hidden_states=config.return_hidden_states,
            embedding_multiplier=config.embedding_multiplier,
            logits_scaling=config.logits_scaling,
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        lora_manager: LoRAManager | None,
        needs_hidden_state_input: bool = False,
    ) -> tuple[TensorType | BufferType, ...]:
        # TODO: Move input symbol computation from the manager classes.
        # It should be possible to compute the input symbols from the model
        # config.
        device_ref = self.config.devices[0]

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = kv_params.get_symbolic_inputs()

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        if lora_manager is not None:
            (
                lora_ids,
                lora_ranks,
                lora_grouped_offsets,
                num_active_loras,
                lora_end_idx,
                batch_seq_len,
                lora_ids_kv,
                lora_grouped_offsets_kv,
            ) = lora_manager.get_symbolic_inputs(device_ref)
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                lora_ids,
                lora_ranks,
                lora_grouped_offsets,
                num_active_loras,
                lora_end_idx,
                batch_seq_len,
                lora_ids_kv,
                lora_grouped_offsets_kv,
                *kv_inputs[0],
            )
        # hidden state input is for EAGLE-like spec decoding draft models
        if needs_hidden_state_input:
            hidden_states_type = TensorType(
                self.config.dtype,
                shape=["total_seq_len", self.config.hidden_size],
                device=device_ref,
            )
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                hidden_states_type,
                *kv_inputs[0],
            )

        return (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *kv_inputs[0],
        )
