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
"""Tests for kernel profiling parsing logic in smoke_test.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from smoke_tests.smoke_test import (
    KernelHit,
    KernelProfile,
    _lookup_kernel_source,
    build_kernel_profile,
    parse_kernel_hits,
    write_kernel_profile,
)


SAMPLE_STDERR = """\
INFO: Loading model weights...
[OP] LAUNCH elementwise(mo.add) [id=0] shape=1x2048x8192;vector_width=16
[OP] COMPLETE elementwise(mo.add) [id=0] shape=1x2048x8192;vector_width=16
[OP] LAUNCH matmul(mo.matmul) [id=1] A=1x2048x8192xbfloat16;B=8192x8192xbfloat16;C=1x2048x8192xbfloat16;transpose_a=false;transpose_b=false
[OP] COMPLETE matmul(mo.matmul) [id=1] A=1x2048x8192xbfloat16;B=8192x8192xbfloat16;C=1x2048x8192xbfloat16;transpose_a=false;transpose_b=false
[OP] LAUNCH elementwise(mo.add) [id=2] shape=1x2048x8192;vector_width=16
[OP] COMPLETE elementwise(mo.add) [id=2] shape=1x2048x8192;vector_width=16
[OP] LAUNCH elementwise(mo.mul) [id=3] shape=1x2048x8192;vector_width=16
[OP] COMPLETE elementwise(mo.mul) [id=3] shape=1x2048x8192;vector_width=16
some random server log line
[OP] LAUNCH rms_norm [id=4] target=gpu:0
[OP] COMPLETE rms_norm [id=4] target=gpu:0
""".strip()

# Realistic LLM inference stderr with low-level GPU kernel names
SAMPLE_LLM_STDERR = """\
[OP] LAUNCH flash_attention [id=0] q=1x32x128;k=1x32x2048x128;v=1x32x2048x128;output=1x32x128
[OP] COMPLETE flash_attention [id=0] q=1x32x128;k=1x32x2048x128;v=1x32x2048x128;output=1x32x128
[OP] LAUNCH flare_mla_decoding [id=1] q=1x128x192;output=1x128x192
[OP] COMPLETE flare_mla_decoding [id=1] q=1x128x192;output=1x128x192
[OP] LAUNCH rms_norm_fused_residual_add [id=2] target=gpu:0
[OP] COMPLETE rms_norm_fused_residual_add [id=2] target=gpu:0
[OP] LAUNCH _cublasLt_matmul [id=3]
[OP] COMPLETE _cublasLt_matmul [id=3]
[OP] LAUNCH ep.fused_silu.fp8 [id=4]
[OP] COMPLETE ep.fused_silu.fp8 [id=4]
[OP] LAUNCH mo.moe.create_indices [id=5]
[OP] COMPLETE mo.moe.create_indices [id=5]
[OP] LAUNCH mo.mla.graph.prefill.decode.paged [id=6]
[OP] COMPLETE mo.mla.graph.prefill.decode.paged [id=6]
""".strip()


def test_parse_kernel_hits_basic() -> None:
    lines = SAMPLE_STDERR.splitlines()
    hits = parse_kernel_hits(lines)

    assert len(hits) == 5
    assert hits[0].kernel == "elementwise(mo.add)"
    assert hits[0].detail == "shape=1x2048x8192;vector_width=16"
    assert hits[1].kernel == "matmul(mo.matmul)"
    assert "A=1x2048x8192xbfloat16" in hits[1].detail
    assert hits[4].kernel == "rms_norm"
    assert hits[4].detail == "target=gpu:0"


def test_parse_kernel_hits_empty() -> None:
    hits = parse_kernel_hits(["no op lines here", "just regular logs"])
    assert hits == []


def test_parse_kernel_hits_only_launch_not_complete() -> None:
    """Verify we only capture LAUNCH, not COMPLETE lines."""
    lines = [
        "[OP] LAUNCH foo [id=0] detail",
        "[OP] COMPLETE foo [id=0] detail",
    ]
    hits = parse_kernel_hits(lines)
    assert len(hits) == 1
    assert hits[0].kernel == "foo"


def test_kernel_hit_shape_key() -> None:
    hit = KernelHit(kernel="matmul(mo.matmul)", detail="shape=128x64")
    assert hit.shape_key == "matmul(mo.matmul) | shape=128x64"

    hit_no_detail = KernelHit(kernel="some_op", detail="")
    assert hit_no_detail.shape_key == "some_op"


def test_build_kernel_profile_aggregation() -> None:
    lines = SAMPLE_STDERR.splitlines()
    hits = parse_kernel_hits(lines)

    with patch(
        "smoke_tests.smoke_test.get_gpu_name_and_count",
        return_value=("Test GPU", 1),
    ):
        profile = build_kernel_profile("test/model", hits)

    assert profile.model == "test/model"
    assert profile.gpu_name == "Test GPU"
    assert profile.gpu_count == 1
    assert profile.total_kernel_launches == 5

    # elementwise(mo.add) appears twice with same detail -> grouped
    assert profile.unique_kernels == 4

    # Most common first
    assert profile.kernels[0]["kernel"] == "elementwise(mo.add)"
    assert profile.kernels[0]["count"] == 2
    # Detail fields are parsed into top-level keys
    assert profile.kernels[0]["shape"] == "1x2048x8192"
    assert profile.kernels[0]["vector_width"] == "16"


def test_build_kernel_profile_detail_parsing() -> None:
    hits = [
        KernelHit(
            kernel="matmul(mo.matmul)",
            detail="A=128x64xfloat32;B=64x128xfloat32;transpose_b=true",
        ),
    ]
    with patch(
        "smoke_tests.smoke_test.get_gpu_name_and_count",
        return_value=("H100", 2),
    ):
        profile = build_kernel_profile("meta-llama/llama-3.2-1b", hits)

    entry = profile.kernels[0]
    assert entry["A"] == "128x64xfloat32"
    assert entry["B"] == "64x128xfloat32"
    assert entry["transpose_b"] == "true"


def test_build_kernel_profile_includes_source() -> None:
    """Verify that source mapping is included for known kernels."""
    hits = [
        KernelHit(kernel="flash_attention", detail="q=1x32x128"),
        KernelHit(kernel="flare_mla_decoding", detail="q=1x128x192"),
        KernelHit(kernel="rms_norm", detail=""),
        KernelHit(kernel="_cublasLt_matmul", detail=""),
        KernelHit(kernel="ep.fused_silu.fp8", detail=""),
        KernelHit(kernel="unknown_kernel_xyz", detail=""),
    ]
    with patch(
        "smoke_tests.smoke_test.get_gpu_name_and_count",
        return_value=("H100", 1),
    ):
        profile = build_kernel_profile("test/model", hits)

    by_name = {e["kernel"]: e for e in profile.kernels}

    assert "mha.mojo" in by_name["flash_attention"]["source"]
    assert "mla.mojo" in by_name["flare_mla_decoding"]["source"]
    assert "normalization.mojo" in by_name["rms_norm"]["source"]
    assert "cuBLASLt" in by_name["_cublasLt_matmul"]["source"]
    assert "ep_api.mojo" in by_name["ep.fused_silu.fp8"]["source"]
    # Unknown kernels should have no source key
    assert "source" not in by_name["unknown_kernel_xyz"]


def test_lookup_kernel_source_prefix_matching() -> None:
    """Verify prefix matching picks the longest match."""
    # Exact match
    assert _lookup_kernel_source("flash_attention") is not None
    assert "mha.mojo" in _lookup_kernel_source("flash_attention")

    # Prefix match: "flash_attention_split_kv" should match that entry first
    src = _lookup_kernel_source("flash_attention_split_kv")
    assert src is not None
    assert "flash_attention.mojo" in src

    # mo.mla.graph.prefill.decode.paged should match before mo.mla.graph.prefill.paged
    src = _lookup_kernel_source("mo.mla.graph.prefill.decode.paged")
    assert src is not None
    assert "mla_graph" in src

    # Unknown kernel
    assert _lookup_kernel_source("totally_unknown_kernel") is None


def test_parse_llm_kernels() -> None:
    """Test parsing of realistic LLM inference kernel traces."""
    lines = SAMPLE_LLM_STDERR.splitlines()
    hits = parse_kernel_hits(lines)

    assert len(hits) == 7
    kernel_names = [h.kernel for h in hits]
    assert "flash_attention" in kernel_names
    assert "flare_mla_decoding" in kernel_names
    assert "rms_norm_fused_residual_add" in kernel_names
    assert "_cublasLt_matmul" in kernel_names
    assert "ep.fused_silu.fp8" in kernel_names
    assert "mo.moe.create_indices" in kernel_names
    assert "mo.mla.graph.prefill.decode.paged" in kernel_names


def test_write_kernel_profile_json() -> None:
    profile = KernelProfile(
        model="test/model",
        gpu_name="H100",
        gpu_count=1,
        total_kernel_launches=3,
        unique_kernels=2,
        kernels=[
            {"kernel": "matmul(mo.matmul)", "count": 2, "source": "linalg/matmul/__init__.mojo", "detail": "shapes"},
            {"kernel": "elementwise(mo.add)", "count": 1, "source": "algorithm/functional.mojo"},
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = write_kernel_profile(Path(tmpdir), profile)
        assert fp.exists()

        data = json.loads(fp.read_text())
        assert data["model"] == "test/model"
        assert data["total_kernel_launches"] == 3
        assert len(data["kernels"]) == 2
        assert data["kernels"][0]["kernel"] == "matmul(mo.matmul)"
        assert data["kernels"][0]["source"] == "linalg/matmul/__init__.mojo"
