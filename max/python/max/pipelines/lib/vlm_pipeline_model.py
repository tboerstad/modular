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

"""Base class for two-stage Vision Language Model (VLM) pipeline models.

This module provides ``VLMPipelineModelBase``, an abstract base class that
extracts the common two-stage (vision + language) execution pattern shared
by VLM architectures such as Idefics3, InternVL, Gemma3, Qwen2.5VL, and
Qwen3VL.

Subclasses implement the architecture-specific hooks for graph construction,
vision/language execution, and input preparation, while the base class
provides shared boilerplate for model loading, output parsing, row-offset
pre-allocation, and the two-stage ``execute()`` template.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import DLPackArray, Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextAndVisionContext

from .config import KVCacheConfig, PipelineConfig
from .interfaces import ModelInputs, ModelOutputs, PipelineModelWithKVCache

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class VLMModelInputs(ModelInputs):
    """Base class for two-stage VLM model inputs.

    Subclasses add architecture-specific vision fields (pixel_values,
    attention_mask, position_ids, etc.). The ``has_vision_inputs`` property
    must be implemented by subclasses.
    """

    @property
    def has_vision_inputs(self) -> bool:
        """Returns True when vision data is present in this batch."""
        raise NotImplementedError


class VLMPipelineModelBase(
    PipelineModelWithKVCache[TextAndVisionContext],
):
    """Abstract base for two-stage VLM pipeline models.

    Provides:
    - ``_parse_language_outputs()``: shared output parsing
    - ``_preallocate_row_offsets()``: shared row-offset pre-allocation
    - ``load_model()``: template for building & compiling vision + language graphs

    Subclasses must implement the abstract hooks listed below. The class
    intentionally does NOT enforce a particular tensor type (``Buffer`` vs
    ``list[Buffer]``) for multi-device support — subclasses control argument
    passing entirely through their hook implementations.
    """

    vision_model: Model
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

    # ------------------------------------------------------------------
    # Shared concrete methods
    # ------------------------------------------------------------------

    def _preallocate_row_offsets_single(self) -> Buffer:
        """Pre-allocate a row-offset buffer for multi-step execution (single device).

        Returns:
            A pre-allocated buffer of shape ``[max_batch_size + 1]`` on ``devices[0]``.
        """
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        return Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

    def _preallocate_row_offsets_multi(self) -> list[Buffer]:
        """Pre-allocate row-offset buffers for multi-step execution (all devices).

        Returns:
            A list of pre-allocated buffers, one per device.
        """
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        host = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        )
        return [host.to(dev) for dev in self.devices]

    @staticmethod
    def _parse_language_outputs(
        language_outputs: tuple[Any, ...],
    ) -> ModelOutputs:
        """Convert raw language model outputs to ``ModelOutputs``.

        This implements the shared output-parsing logic that is identical
        across all two-stage VLM architectures:

        - 3 outputs → (next_token_logits, logits, logit_offsets)
        - 1 output  → logits == next_token_logits

        Args:
            language_outputs: Raw outputs from the language model execution.

        Returns:
            A ``ModelOutputs`` instance.
        """
        if len(language_outputs) == 3:
            assert isinstance(language_outputs[0], Buffer)
            assert isinstance(language_outputs[1], Buffer)
            assert isinstance(language_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=language_outputs[0],
                logits=language_outputs[1],
                logit_offsets=language_outputs[2],
            )
        else:
            assert isinstance(language_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=language_outputs[0],
                logits=language_outputs[0],
            )

    def _prepare_text_buffers(
        self,
        context_batch: Sequence[TextAndVisionContext],
        return_n_logits: int = 1,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Build the common text input buffers from a context batch.

        Creates ragged token IDs, cumulative row offsets, and the
        ``return_n_logits`` scalar — the text-input assembly that is
        duplicated verbatim across all VLM ``prepare_initial_token_inputs``
        methods.

        Args:
            context_batch: The batch of text-and-vision contexts.
            return_n_logits: Number of logits to return per sequence.

        Returns:
            A tuple of ``(input_ids, input_row_offsets, return_n_logits_buf)``
            all placed on ``self.devices[0]``.
        """
        device = self.devices[0]

        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(device)

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
        input_ids = Buffer.from_numpy(tokens).to(device)

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        return input_ids, input_row_offsets, return_n_logits_buf

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_vision_graph(
        self, **kwargs: Any
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build and return the vision model graph and its state dict."""
        ...

    @abstractmethod
    def _build_language_graph(
        self, **kwargs: Any
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build and return the language model graph and its state dict."""
        ...
