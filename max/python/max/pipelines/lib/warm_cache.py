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

When multiple models are compiled in the same process, the compilation engine's
internal kernel cache automatically shares compiled kernel objects between them.
This applies both within the same architecture (full sharing of identical
operation graphs) and across different architectures that use common operations
(e.g., matmul, attention, normalization).

Models with the same architecture are grouped together during compilation to
maximize cache reuse, since identical architectures produce identical operation
graphs and therefore achieve 100% kernel reuse.
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
# ModelInfo
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Information about a model for compilation planning."""

    model_path: str
    config: PipelineConfig
    arch_name: str


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def precompile_models(
    primary_config: PipelineConfig,
    additional_model_paths: tuple[str, ...],
    config_kwargs: dict[str, Any],
) -> None:
    """Precompile multiple models, taking advantage of shared kernels.

    The compilation engine caches compiled kernel objects internally, so
    compiling models in the same process avoids redundant kernel compilation.
    This works both within the same architecture (identical operation graphs)
    and across architectures that share common operations.

    Models are grouped by architecture so that identical architectures are
    compiled adjacently, maximizing kernel cache hits.

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
    _log_compilation_plan(model_infos)

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
    """Build ModelInfo for each model, resolving the architecture name."""
    infos: list[ModelInfo] = []
    for model_path, config in configs:
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=config.model.huggingface_model_repo,
            use_legacy_module=config.use_legacy_module,
        )
        arch_name = arch.name if arch else "unknown"
        infos.append(
            ModelInfo(
                model_path=model_path,
                config=config,
                arch_name=arch_name,
            )
        )
    return infos


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_compilation_plan(model_infos: list[ModelInfo]) -> None:
    """Log the multi-model compilation plan."""
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

    # Cross-architecture note.
    if len(arch_groups) > 1:
        logger.info("")
        logger.info(
            "  Cross-architecture kernel sharing enabled via compiler cache"
        )

    logger.info("=" * 60)
    logger.info("")


# ---------------------------------------------------------------------------
# Compilation ordering
# ---------------------------------------------------------------------------


def _order_for_max_sharing(model_infos: list[ModelInfo]) -> list[ModelInfo]:
    """Order models to maximise cumulative kernel cache hits.

    Models with the same architecture are grouped together so that identical
    operation graphs achieve 100% kernel reuse.  The compiler's internal
    cache handles cross-architecture sharing of common operations (e.g.,
    matmul, attention, normalization) automatically.
    """
    return sorted(model_infos, key=lambda info: info.arch_name)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _compile_models(model_infos: list[ModelInfo]) -> None:
    """Compile all models in the given order.

    The compilation engine's kernel cache is process-scoped, so compiling
    multiple models in the same session shares compiled kernel objects
    automatically — both within and across architectures.
    """
    total_start = time.perf_counter()
    total = len(model_infos)
    compiled_count = 0
    prev_arch: str | None = None

    for info in model_infos:
        model_start = time.perf_counter()
        compiled_count += 1

        # On architecture transitions, note cross-arch kernel reuse.
        if prev_arch is not None and info.arch_name != prev_arch:
            logger.info(
                "  Switching architecture — compiler cache provides "
                "cross-architecture kernel reuse"
            )
        prev_arch = info.arch_name

        logger.info(
            f"[{compiled_count}/{total}] Compiling: {info.model_path} "
            f"({info.arch_name})"
        )

        try:
            _ = PIPELINE_REGISTRY.retrieve(info.config)
            elapsed = time.perf_counter() - model_start
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
