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

"""Multi-model precompilation with shared kernel optimization.

When multiple models are compiled together, they can share compiled kernel
objects through the compilation engine's internal cache. This applies not just
to models with the same architecture, but also across different architectures
that use common kernel types (matmul, attention, normalization, etc.).

For example, a Llama model and a Gemma model both use attention, matmul, and
normalization kernels. Compiling them in the same session allows the second
model to reuse kernel compilations from the first.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import PipelineConfig
from .registry import PIPELINE_REGISTRY

logger = logging.getLogger("max.pipelines")

# ---------------------------------------------------------------------------
# Kernel category classification
# ---------------------------------------------------------------------------
# Every compiled model is built from a set of kernel types (matmul, attention,
# normalization, …).  Two models that share a kernel *category* will reuse the
# same compiled kernel objects when they target the same device and dtype,
# regardless of whether they belong to the same architecture.

_TRANSFORMER_DECODER_KERNELS = frozenset(
    {
        "matmul",
        "attention",
        "normalization",
        "activation",
        "embedding",
        "rope",
        "kv_cache",
        "sampling",
        "softmax",
    }
)

_TRANSFORMER_ENCODER_KERNELS = frozenset(
    {
        "matmul",
        "attention",
        "normalization",
        "activation",
        "embedding",
        "softmax",
        "pooling",
    }
)

_DIFFUSION_KERNELS = frozenset(
    {
        "matmul",
        "normalization",
        "activation",
        "convolution",
        "attention",
        "embedding",
    }
)

_ENCODER_ARCH_PATTERNS = ("Bert", "MPNet", "Roberta", "Albert", "XLM")
_DIFFUSION_ARCH_PATTERNS = ("Flux", "StableDiffusion", "UNet")


def _get_kernel_categories(arch_name: str | None) -> frozenset[str]:
    """Determine the kernel categories an architecture uses.

    This is a heuristic based on the architecture name.  All transformer-based
    architectures share core kernel types (matmul, attention, normalization).
    """
    if not arch_name or arch_name == "unknown":
        return frozenset()

    for pattern in _DIFFUSION_ARCH_PATTERNS:
        if pattern in arch_name:
            return _DIFFUSION_KERNELS

    for pattern in _ENCODER_ARCH_PATTERNS:
        if pattern in arch_name:
            return _TRANSFORMER_ENCODER_KERNELS

    # Default: transformer decoder (causal LM) — the most common case.
    return _TRANSFORMER_DECODER_KERNELS


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Information about a model for compilation planning."""

    model_path: str
    config: PipelineConfig
    arch_name: str
    kernel_categories: frozenset[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def precompile_models(
    primary_config: PipelineConfig,
    additional_model_paths: tuple[str, ...],
    config_kwargs: dict[str, Any],
) -> None:
    """Precompile multiple models, taking advantage of shared kernels.

    Models are analysed for kernel sharing potential — both within the same
    architecture (full sharing) and across different architectures (partial
    sharing of common kernel types like matmul, attention, normalization).

    The compilation engine caches compiled kernel objects, so compiling models
    together in the same process avoids redundant kernel compilation regardless
    of whether models share the same architecture.

    Args:
        primary_config: The fully resolved PipelineConfig for the primary model.
        additional_model_paths: Paths to additional models to precompile.
        config_kwargs: The original CLI kwargs, used as a base for creating
            configs for additional models.
    """
    all_configs = _build_all_configs(
        primary_config, additional_model_paths, config_kwargs
    )

    model_infos = _build_model_infos(all_configs)
    _log_shared_kernel_info(model_infos)

    ordered = _order_for_max_sharing(model_infos)
    _compile_models(ordered)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _build_all_configs(
    primary_config: PipelineConfig,
    additional_model_paths: tuple[str, ...],
    config_kwargs: dict[str, Any],
) -> list[tuple[str, PipelineConfig]]:
    """Build PipelineConfig for each model.

    Additional models inherit device and target settings from the primary
    config but get their own model path and auto-detected quantization
    encoding.
    """
    configs: list[tuple[str, PipelineConfig]] = [
        (primary_config.model.model_path, primary_config)
    ]

    for model_path in additional_model_paths:
        model_kwargs = config_kwargs.copy()
        model_kwargs["model_path"] = model_path
        # Reset model-specific settings so they are auto-detected for each
        # additional model rather than inheriting the primary model's values.
        model_kwargs.pop("quantization_encoding", None)
        model_kwargs.pop("weight_path", None)

        try:
            config = PipelineConfig(**model_kwargs)
            configs.append((model_path, config))
        except Exception as e:
            logger.error(
                f"Failed to create config for model '{model_path}': {e}"
            )
            raise

    return configs


def _build_model_infos(
    configs: list[tuple[str, PipelineConfig]],
) -> list[ModelInfo]:
    """Build ModelInfo for each model, including kernel analysis."""
    infos: list[ModelInfo] = []
    for model_path, config in configs:
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=config.model.huggingface_model_repo,
            use_legacy_module=config.use_legacy_module,
        )
        arch_name = arch.name if arch else "unknown"
        kernel_categories = _get_kernel_categories(arch_name)
        infos.append(
            ModelInfo(
                model_path=model_path,
                config=config,
                arch_name=arch_name,
                kernel_categories=kernel_categories,
            )
        )
    return infos


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_shared_kernel_info(model_infos: list[ModelInfo]) -> None:
    """Log information about shared kernels between models."""
    total = len(model_infos)

    # Group by architecture for per-arch summary.
    arch_groups: dict[str, list[ModelInfo]] = defaultdict(list)
    for info in model_infos:
        arch_groups[info.arch_name].append(info)

    logger.info("")
    logger.info(
        f"Multi-model precompilation: {total} model(s), "
        f"{len(arch_groups)} architecture(s)"
    )
    logger.info("=" * 60)

    # Per-architecture listing.
    for arch_name, infos in arch_groups.items():
        model_paths = [i.model_path for i in infos]
        if len(infos) > 1:
            logger.info(
                f"  {arch_name}: {len(infos)} models (shared kernels)"
            )
            for path in model_paths:
                logger.info(f"    - {path}")
        else:
            logger.info(f"  {arch_name}: {model_paths[0]}")

    # Cross-architecture kernel sharing.
    if len(arch_groups) > 1:
        arch_names = list(arch_groups.keys())
        logger.info("")
        logger.info("  Cross-architecture shared kernels:")
        for i in range(len(arch_names)):
            for j in range(i + 1, len(arch_names)):
                a_cats = arch_groups[arch_names[i]][0].kernel_categories
                b_cats = arch_groups[arch_names[j]][0].kernel_categories
                shared = a_cats & b_cats
                if shared:
                    all_cats = a_cats | b_cats
                    pct = len(shared) / len(all_cats) * 100 if all_cats else 0
                    logger.info(
                        f"    {arch_names[i]} <-> {arch_names[j]}: "
                        f"{len(shared)} shared kernel types "
                        f"({', '.join(sorted(shared))}) "
                        f"[{pct:.0f}% overlap]"
                    )

        # Kernels common to every architecture in this batch.
        all_arch_cats = [
            infos[0].kernel_categories for infos in arch_groups.values()
        ]
        common_across_all = (
            frozenset.intersection(*all_arch_cats) if all_arch_cats else frozenset()
        )
        if common_across_all:
            logger.info(
                f"    All architectures share: "
                f"{', '.join(sorted(common_across_all))}"
            )

    logger.info("=" * 60)
    logger.info("")


# ---------------------------------------------------------------------------
# Compilation ordering
# ---------------------------------------------------------------------------


def _order_for_max_sharing(model_infos: list[ModelInfo]) -> list[ModelInfo]:
    """Order models to maximise cumulative kernel cache hits.

    Models with the most kernel categories are compiled first so their
    kernels warm the cache for subsequent models.  Among models with the
    same number of kernel categories, models with the same architecture
    are grouped together (identical architecture = 100 % kernel reuse).
    """
    return sorted(
        model_infos,
        key=lambda info: (-len(info.kernel_categories), info.arch_name),
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _compile_models(model_infos: list[ModelInfo]) -> None:
    """Compile all models in the given order."""
    total_start = time.perf_counter()
    total = len(model_infos)
    compiled_count = 0

    # Track which kernel categories have already been compiled.
    compiled_kernel_categories: set[str] = set()
    prev_arch: str | None = None

    for info in model_infos:
        model_start = time.perf_counter()
        compiled_count += 1

        # On architecture transitions, report expected kernel reuse.
        if prev_arch is not None and info.arch_name != prev_arch:
            reusable = info.kernel_categories & compiled_kernel_categories
            if reusable:
                logger.info(
                    f"  Reusing {len(reusable)} compiled kernel types from "
                    f"previous models: {', '.join(sorted(reusable))}"
                )
        prev_arch = info.arch_name

        logger.info(
            f"[{compiled_count}/{total}] Compiling: {info.model_path} "
            f"({info.arch_name})"
        )

        try:
            _ = PIPELINE_REGISTRY.retrieve(info.config)
            elapsed = time.perf_counter() - model_start
            compiled_kernel_categories |= info.kernel_categories
            logger.info(
                f"[{compiled_count}/{total}] "
                f"Compiled {info.model_path} in {elapsed:.1f}s"
            )
        except Exception as e:
            logger.error(
                f"Failed to compile model '{info.model_path}': {e}"
            )
            raise

    total_elapsed = time.perf_counter() - total_start
    logger.info("")
    logger.info(
        f"Multi-model precompilation complete: "
        f"{compiled_count} model(s) in {total_elapsed:.1f}s"
    )
