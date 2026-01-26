#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""
Git bisect helper for smoke_test.py regressions.

Examples:
    # Find commit that broke the smoke test
    ./bisect_smoke_test.py --model modularai/Llama-3.1-8B-Instruct-GGUF \
        --good-commit abc123 --bad-commit def456

    # Find commit where text accuracy dropped below 0.7
    ./bisect_smoke_test.py --model modularai/Llama-3.1-8B-Instruct-GGUF \
        --good-commit abc123 --bad-commit def456 --text-threshold 0.7
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click

BISECT_GOOD = 0
BISECT_BAD = 1
BISECT_SKIP = 125


def find_repo_root() -> Path:
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


def run_smoke_test(model: str, output_path: Path) -> bool | None:
    """Returns True on success, False on failure, None on timeout."""
    repo_root = find_repo_root()
    cmd = [
        str(repo_root / "bazelw"),
        "run",
        "smoke-test",
        "--",
        model,
        "--framework",
        "max-ci",
        "--output-path",
        str(output_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, cwd=repo_root, timeout=1800).returncode == 0
    except subprocess.TimeoutExpired:
        print("Smoke test timed out after 30 minutes")
        return None


def parse_results(output_path: Path, model: str) -> dict[str, float] | None:
    model_dir = output_path / model.lower().strip().replace("/", "__")
    metrics_file = model_dir / "eval_metrics.json"

    if not metrics_file.exists():
        return None

    metrics = json.loads(metrics_file.read_text())
    results = {}
    for entry in metrics:
        task = entry.get("eval_task", "")
        accuracy = entry.get("accuracy")
        if accuracy is not None:
            if "chartqa" in task:
                results["vision"] = accuracy
            elif "gsm8k" in task:
                results["text"] = accuracy
    return results


def check_thresholds(
    results: dict[str, float],
    text_threshold: float | None,
    vision_threshold: float | None,
) -> bool:
    if text_threshold is not None:
        if results.get("text", 0) < text_threshold:
            print(f"Text accuracy {results.get('text')} < {text_threshold}")
            return False
    if vision_threshold is not None:
        if results.get("vision", 0) < vision_threshold:
            print(f"Vision accuracy {results.get('vision')} < {vision_threshold}")
            return False
    return True


def test_commit(
    model: str,
    text_threshold: float | None,
    vision_threshold: float | None,
) -> int:
    has_thresholds = text_threshold is not None or vision_threshold is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        success = run_smoke_test(model, output_path)

        if success is None:
            return BISECT_SKIP
        if not success:
            return BISECT_SKIP if has_thresholds else BISECT_BAD

        if not has_thresholds:
            return BISECT_GOOD

        results = parse_results(output_path, model)
        if results is None:
            return BISECT_SKIP

        return (
            BISECT_GOOD
            if check_thresholds(results, text_threshold, vision_threshold)
            else BISECT_BAD
        )


def start_bisect(
    script_path: Path,
    model: str,
    good_commit: str,
    bad_commit: str,
    text_threshold: float | None,
    vision_threshold: float | None,
) -> int:
    repo_root = find_repo_root()

    test_cmd = [
        sys.executable,
        str(script_path),
        "--test",
        "--model",
        model,
    ]
    if text_threshold is not None:
        test_cmd.extend(["--text-threshold", str(text_threshold)])
    if vision_threshold is not None:
        test_cmd.extend(["--vision-threshold", str(vision_threshold)])

    subprocess.run(["git", "bisect", "start"], cwd=repo_root, check=True)
    subprocess.run(["git", "bisect", "bad", bad_commit], cwd=repo_root, check=True)
    subprocess.run(["git", "bisect", "good", good_commit], cwd=repo_root, check=True)

    print(f"Running bisect: {' '.join(test_cmd)}")
    result = subprocess.run(["git", "bisect", "run"] + test_cmd, cwd=repo_root)
    subprocess.run(["git", "bisect", "log"], cwd=repo_root)
    return result.returncode


@click.command()
@click.option("--model", required=True, help="HuggingFace model path")
@click.option("--good-commit", help="Known good commit")
@click.option("--bad-commit", help="Known bad commit")
@click.option("--text-threshold", type=float, help="Min text accuracy")
@click.option("--vision-threshold", type=float, help="Min vision accuracy")
@click.option("--test", is_flag=True, help="Test current commit (used by git bisect)")
def main(
    model: str,
    good_commit: str | None,
    bad_commit: str | None,
    text_threshold: float | None,
    vision_threshold: float | None,
    test: bool,
) -> None:
    if test:
        sys.exit(test_commit(model, text_threshold, vision_threshold))

    if not good_commit or not bad_commit:
        raise click.UsageError("--good-commit and --bad-commit are required")

    # Copy script outside repo so git checkout doesn't overwrite it
    with tempfile.TemporaryDirectory(prefix="bisect_script_") as tmpdir:
        script_path = Path(tmpdir) / "bisect_smoke_test.py"
        shutil.copy2(Path(__file__).resolve(), script_path)
        sys.exit(
            start_bisect(
                script_path,
                model,
                good_commit,
                bad_commit,
                text_threshold,
                vision_threshold,
            )
        )


if __name__ == "__main__":
    main()
