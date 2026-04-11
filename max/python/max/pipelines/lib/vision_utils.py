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

"""Shared utilities for Vision Language Model (VLM) pipelines."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.graph import TensorValue, ops
from max.nn.kernels import scatter_nd_skip_oob_indices

logger = logging.getLogger("max.pipelines")


class VisionStacker:
    """Helper class for efficient parallel stacking of vision patches.

    Uses ThreadPoolExecutor for thread management and bulk numpy operations
    for optimal memory bandwidth utilization.
    """

    def __init__(self, max_workers: int = 24) -> None:
        """Initialize the vision stacker with a thread pool.

        Args:
            max_workers: Maximum number of worker threads (default: 24).
        """
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def stack(
        self, images: list[npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Stack images using parallel bulk copy operations.

        Args:
            images: List of numpy arrays to stack.

        Returns:
            Stacked numpy array.
        """
        n = len(images)
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        # Pre-allocate output.
        out = np.empty((n, *images[0].shape), dtype=images[0].dtype)

        # Divide work evenly among threads.
        # ThreadPoolExecutor will handle cases where n < workers.
        workers = self._pool._max_workers
        step = math.ceil(n / workers)
        slices = [slice(i, min(i + step, n)) for i in range(0, n, step)]

        # Launch parallel bulk copy tasks.
        futures = [
            self._pool.submit(self._copy_block, out, images, sl)
            for sl in slices
        ]

        # Wait for completion and propagate any exceptions.
        for f in as_completed(futures):
            f.result()

        return out

    @staticmethod
    def _copy_block(
        out: npt.NDArray[np.floating[Any]],
        images: list[npt.NDArray[np.floating[Any]]],
        sl: slice,
    ) -> None:
        """Copy a block of images using bulk numpy operations.

        This method performs a C-level bulk copy that releases the GIL,
        allowing true parallel execution.
        """
        # Convert slice of list to temporary array view and bulk copy.
        np.copyto(out[sl], np.asarray(images[sl], dtype=images[0].dtype))


def assert_image_embeddings_invariant(
    image_embeddings: Sequence[Buffer],
    image_token_indices: Sequence[Buffer],
) -> None:
    """Validates that image embeddings count matches image token indices count.

    This prevents out-of-bounds access during scatter operations where image
    embeddings are placed at specific token positions. Supports multi-device
    setups where each device has its own embeddings and indices.

    Args:
        image_embeddings: Per-device tensors of image embeddings.
        image_token_indices: Per-device tensors of image token indices.

    Raises:
        AssertionError: If embedding count doesn't match indices count on
            any device.
    """
    for i, (embed, indices) in enumerate(
        zip(image_embeddings, image_token_indices, strict=True)
    ):
        embed_count = embed.shape[0]
        indices_count = indices.shape[0]
        if embed_count != indices_count:
            logger.error(
                f"[CRITICAL] Device {i}: Vision embedding count ({embed_count}) "
                f"!= image token indices count ({indices_count})."
            )
        assert embed_count == indices_count, (
            f"Vision embedding shape mismatch on device {i}: {embed_count} embeddings "
            f"but {indices_count} indices."
        )


def merge_multimodal_embeddings(
    inputs_embeds: TensorValue,
    multimodal_embeddings: TensorValue,
    image_token_indices: TensorValue,
) -> TensorValue:
    """Merges multimodal embeddings into text embeddings at pre-computed indices.

    This is the MAX Graph API implementation of the embedding merge operation.
    It returns an updated copy of inputs_embeds with multimodal embeddings
    at positions specified by the indices.

    Indices may be oob (out of bounds), in which case the corresponding
    update will be skipped.

    Args:
        inputs_embeds: Text embeddings with shape [num_tokens, hidden_size].
        multimodal_embeddings: Vision embeddings to insert with shape
            [num_multimodal_tokens, hidden_size].
        image_token_indices: Pre-computed indices where to insert multimodal
            embeddings, with shape [num_multimodal_tokens].

    Returns:
        Copy of the inputs_embeds tensor with multimodal embeddings merged in.
    """
    # Use scatter_nd_skip_oob_indices to directly place embeddings at specified indices.
    # Expand indices to 2D for scatter_nd_skip_oob_indices: [num_tokens, 1]
    indices_2d = ops.unsqueeze(image_token_indices, -1)

    if multimodal_embeddings.dtype != inputs_embeds.dtype:
        multimodal_embeddings = ops.cast(
            multimodal_embeddings, dtype=inputs_embeds.dtype
        )

    # Scatter the multimodal embeddings into inputs_embeds at the specified
    # indices. Any negative values in the indices means that the corresponding
    # update will be skipped.
    return scatter_nd_skip_oob_indices(
        input=inputs_embeds,
        updates=multimodal_embeddings,
        indices=indices_2d,
    )
