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

When multiple models share the same architecture, they use the same kernel
implementations (attention, matmul, normalization, etc.). By compiling these
models together in the same process, the compilation engine can reuse compiled
kernel objects, reducing total compilation time.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from .config import PipelineConfig
from .registry import PIPELINE_REGISTRY

logger = logging.getLogger("max.pipelines")


def precompile_models(
    primary_config: PipelineConfig,
    additional_model_paths: tuple[str, ...],
    config_kwargs: dict[str, Any],
) -> None:
    """Precompile multiple models, taking advantage of shared kernels.

    Models that share the same architecture use identical kernel implementations
    (e.g., attention, matmul, normalization kernels). By compiling these models
    together in the same process, the compilation engine reuses compiled kernel
    objects from its internal cache, reducing total compilation time compared to
    compiling each model in a separate process.

    Models are grouped by architecture and compiled in order so that models
    sharing kernels are compiled consecutively, maximizing cache reuse.

    Args:
        primary_config: The fully resolved PipelineConfig for the primary model.
        additional_model_paths: Paths to additional models to precompile.
        config_kwargs: The original CLI kwargs, used as a base for creating
            configs for additional models (device settings, target, etc. are
            inherited).
    """
    all_configs = _build_all_configs(
        primary_config, additional_model_paths, config_kwargs
    )

    arch_groups = _group_by_architecture(all_configs)
    _log_shared_kernel_info(arch_groups)
    _compile_models(arch_groups)


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


def _group_by_architecture(
    configs: list[tuple[str, PipelineConfig]],
) -> dict[str, list[tuple[str, PipelineConfig]]]:
    """Group model configs by their architecture name.

    Models with the same architecture share kernel implementations.
    Grouping allows us to compile them consecutively, maximizing the
    compilation engine's kernel cache reuse.
    """
    groups: dict[str, list[tuple[str, PipelineConfig]]] = defaultdict(list)

    for model_path, config in configs:
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=config.model.huggingface_model_repo,
            use_legacy_module=config.use_legacy_module,
        )
        arch_name = arch.name if arch else "unknown"
        groups[arch_name].append((model_path, config))

    return dict(groups)


def _log_shared_kernel_info(
    arch_groups: dict[str, list[tuple[str, PipelineConfig]]],
) -> None:
    """Log information about shared kernels between models."""
    total_models = sum(len(models) for models in arch_groups.values())

    logger.info("")
    logger.info(
        f"Multi-model precompilation: {total_models} model(s), "
        f"{len(arch_groups)} architecture(s)"
    )
    logger.info("=" * 60)

    for arch_name, models in arch_groups.items():
        model_paths = [m[0] for m in models]
        if len(models) > 1:
            logger.info(
                f"  {arch_name}: {len(models)} models (shared kernels)"
            )
            for path in model_paths:
                logger.info(f"    - {path}")
        else:
            logger.info(f"  {arch_name}: {model_paths[0]}")

    shared_count = sum(
        len(models) for models in arch_groups.values() if len(models) > 1
    )
    if shared_count > 0:
        logger.info(
            f"  {shared_count} models share kernels with at least one other "
            "model"
        )

    logger.info("=" * 60)
    logger.info("")


def _compile_models(
    arch_groups: dict[str, list[tuple[str, PipelineConfig]]],
) -> None:
    """Compile all models, grouped by architecture for kernel reuse.

    Models sharing the same architecture are compiled consecutively. The
    compilation engine internally caches compiled kernel objects, so
    compiling same-architecture models back-to-back maximizes cache hits
    and reduces redundant compilation work.
    """
    total_start = time.perf_counter()
    compiled_count = 0
    total_models = sum(len(models) for models in arch_groups.values())

    for arch_name, models in arch_groups.items():
        if len(models) > 1:
            logger.info(
                f"Compiling {len(models)} models with shared "
                f"'{arch_name}' kernels..."
            )

        for model_path, config in models:
            model_start = time.perf_counter()
            compiled_count += 1
            logger.info(
                f"[{compiled_count}/{total_models}] Compiling: {model_path}"
            )

            try:
                _ = PIPELINE_REGISTRY.retrieve(config)
                elapsed = time.perf_counter() - model_start
                logger.info(
                    f"[{compiled_count}/{total_models}] "
                    f"Compiled {model_path} in {elapsed:.1f}s"
                )
            except Exception as e:
                logger.error(
                    f"Failed to compile model '{model_path}': {e}"
                )
                raise

    total_elapsed = time.perf_counter() - total_start
    logger.info("")
    logger.info(
        f"Multi-model precompilation complete: "
        f"{compiled_count} model(s) in {total_elapsed:.1f}s"
    )
