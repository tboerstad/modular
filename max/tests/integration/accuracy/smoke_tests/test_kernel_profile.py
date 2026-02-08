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


def test_write_kernel_profile_json() -> None:
    profile = KernelProfile(
        model="test/model",
        gpu_name="H100",
        gpu_count=1,
        total_kernel_launches=3,
        unique_kernels=2,
        kernels=[
            {"kernel": "matmul(mo.matmul)", "count": 2, "detail": "shapes"},
            {"kernel": "elementwise(mo.add)", "count": 1},
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
