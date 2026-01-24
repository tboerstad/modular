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
Git bisect helper script for smoke_test.py.

This script automates git bisect to find commits that caused regressions
in smoke test accuracy or introduced failures.

Usage modes:
    1. Start bisect (default): Initiates git bisect with the given commits
       ./bisect_smoke_test.py --model meta-llama/llama-3.1-8b-instruct \
           --good-commit abc123 --bad-commit def456

    2. Test mode (used internally by git bisect):
       ./bisect_smoke_test.py --test --model meta-llama/llama-3.1-8b-instruct

Threshold modes:
    - No thresholds: Test passes if smoke_test.py runs successfully
    - With thresholds: Test passes if accuracy meets or exceeds thresholds

Examples:
    # Find commit that broke the smoke test (any failure)
    ./bisect_smoke_test.py --model meta-llama/llama-3.1-8b-instruct \
        --good-commit abc123 --bad-commit def456

    # Find commit where text accuracy dropped below 0.7
    ./bisect_smoke_test.py --model meta-llama/llama-3.1-8b-instruct \
        --good-commit abc123 --bad-commit def456 --text-threshold 0.7

    # Find commit where vision accuracy dropped below 0.5
    ./bisect_smoke_test.py --model meta-llama/llama-3.1-8b-instruct \
        --good-commit abc123 --bad-commit def456 --vision-threshold 0.5

    # Find commit where either accuracy dropped
    ./bisect_smoke_test.py --model meta-llama/llama-3.1-8b-instruct \
        --good-commit abc123 --bad-commit def456 \
        --text-threshold 0.7 --vision-threshold 0.5
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Git bisect exit codes
BISECT_GOOD = 0  # Commit is good
BISECT_BAD = 1  # Commit is bad
BISECT_SKIP = 125  # Cannot test this commit (build failure, etc.)


def find_repo_root() -> Path:
    """Find the repository root by looking for .git directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no .git directory found)")


def safe_model_name(model: str) -> str:
    """Convert model path to safe directory name."""
    return model.replace("/", "__")


def run_smoke_test(model: str, output_path: Path, num_questions: int) -> bool:
    """
    Run the smoke test via bazel.

    Returns True if the test ran successfully, False otherwise.
    """
    repo_root = find_repo_root()
    bazel = repo_root / "bazelw"

    cmd = [
        str(bazel),
        "run",
        "smoke-test",
        "--",
        model,
        "--framework",
        "max-ci",
        "--output-path",
        str(output_path),
        "--num-questions",
        str(num_questions),
    ]

    logger.info(f"Running smoke test: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            timeout=3600,  # 1 hour timeout
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Smoke test failed with exit code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return False

        logger.info("Smoke test completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Smoke test timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"Error running smoke test: {e}")
        return False


def parse_results(
    output_path: Path,
    model: str,
) -> dict[str, float] | None:
    """
    Parse the eval_metrics.json file and return accuracy by task type.

    Returns a dict mapping task type ('text' or 'vision') to accuracy,
    or None if results cannot be parsed.
    """
    model_dir = output_path / safe_model_name(model.lower().strip())
    metrics_file = model_dir / "eval_metrics.json"

    if not metrics_file.exists():
        logger.error(f"Results file not found: {metrics_file}")
        return None

    try:
        with open(metrics_file) as f:
            metrics = json.load(f)

        results = {}
        for entry in metrics:
            task = entry.get("eval_task", "")
            accuracy = entry.get("accuracy")

            if accuracy is None:
                continue

            # Map task names to threshold types
            if "chartqa" in task:
                results["vision"] = accuracy
            elif "gsm8k" in task:
                results["text"] = accuracy

        logger.info(f"Parsed results: {results}")
        return results

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse results JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading results: {e}")
        return None


def check_thresholds(
    results: dict[str, float],
    text_threshold: float | None,
    vision_threshold: float | None,
) -> bool:
    """
    Check if results meet the specified thresholds.

    Returns True if all specified thresholds are met, False otherwise.
    """
    if text_threshold is not None:
        text_accuracy = results.get("text")
        if text_accuracy is None:
            logger.error("Text threshold specified but no text results found")
            return False
        if text_accuracy < text_threshold:
            logger.info(
                f"Text accuracy {text_accuracy:.4f} below threshold {text_threshold}"
            )
            return False
        logger.info(
            f"Text accuracy {text_accuracy:.4f} meets threshold {text_threshold}"
        )

    if vision_threshold is not None:
        vision_accuracy = results.get("vision")
        if vision_accuracy is None:
            logger.error("Vision threshold specified but no vision results found")
            return False
        if vision_accuracy < vision_threshold:
            logger.info(
                f"Vision accuracy {vision_accuracy:.4f} below threshold "
                f"{vision_threshold}"
            )
            return False
        logger.info(
            f"Vision accuracy {vision_accuracy:.4f} meets threshold {vision_threshold}"
        )

    return True


def test_commit(
    model: str,
    text_threshold: float | None,
    vision_threshold: float | None,
    num_questions: int,
) -> int:
    """
    Test the current commit.

    Returns appropriate git bisect exit code.
    """
    has_thresholds = text_threshold is not None or vision_threshold is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)

        # Run the smoke test
        success = run_smoke_test(model, output_path, num_questions)

        if not success:
            if has_thresholds:
                # Build/test failure when we're checking thresholds - skip
                logger.info("Test failed to run, marking as skip")
                return BISECT_SKIP
            else:
                # No thresholds - we're looking for failures, this is bad
                logger.info("Test failed, marking as bad commit")
                return BISECT_BAD

        # Test ran successfully
        if not has_thresholds:
            # No thresholds - successful run means good commit
            logger.info("Test passed, marking as good commit")
            return BISECT_GOOD

        # Parse results and check thresholds
        results = parse_results(output_path, model)
        if results is None:
            logger.info("Could not parse results, marking as skip")
            return BISECT_SKIP

        if check_thresholds(results, text_threshold, vision_threshold):
            logger.info("All thresholds met, marking as good commit")
            return BISECT_GOOD
        else:
            logger.info("Thresholds not met, marking as bad commit")
            return BISECT_BAD


def start_bisect(
    model: str,
    good_commit: str,
    bad_commit: str,
    text_threshold: float | None,
    vision_threshold: float | None,
    num_questions: int,
) -> int:
    """
    Start git bisect with the given parameters.

    Returns the exit code from git bisect run.
    """
    repo_root = find_repo_root()

    # Copy script to temp location outside repo - git bisect checkouts would
    # otherwise overwrite/delete the script mid-bisect
    script_tmpdir = tempfile.mkdtemp(prefix="bisect_script_")
    script_copy = Path(script_tmpdir) / "bisect_smoke_test.py"
    shutil.copy2(Path(__file__).resolve(), script_copy)

    # Build the test command
    test_cmd = [
        sys.executable,
        str(script_copy),
        "--test",
        "--model",
        model,
        "--num-questions",
        str(num_questions),
    ]

    if text_threshold is not None:
        test_cmd.extend(["--text-threshold", str(text_threshold)])
    if vision_threshold is not None:
        test_cmd.extend(["--vision-threshold", str(vision_threshold)])

    try:
        # Initialize bisect
        logger.info(f"Starting git bisect: good={good_commit}, bad={bad_commit}")

        subprocess.run(
            ["git", "bisect", "start"],
            cwd=repo_root,
            check=True,
        )

        subprocess.run(
            ["git", "bisect", "bad", bad_commit],
            cwd=repo_root,
            check=True,
        )

        subprocess.run(
            ["git", "bisect", "good", good_commit],
            cwd=repo_root,
            check=True,
        )

        # Run bisect
        logger.info(f"Running bisect with command: {' '.join(test_cmd)}")
        result = subprocess.run(
            ["git", "bisect", "run"] + test_cmd,
            cwd=repo_root,
        )

        # Show the result
        subprocess.run(["git", "bisect", "log"], cwd=repo_root)

        return result.returncode

    except subprocess.CalledProcessError as e:
        logger.error(f"Git bisect failed: {e}")
        return 1
    finally:
        # Always reset bisect state
        logger.info("Resetting git bisect state")
        subprocess.run(["git", "bisect", "reset"], cwd=repo_root)
        shutil.rmtree(script_tmpdir, ignore_errors=True)


@click.command()
@click.option(
    "--model",
    type=str,
    required=True,
    help="HuggingFace model path (e.g., meta-llama/llama-3.1-8b-instruct)",
)
@click.option(
    "--good-commit",
    type=str,
    default=None,
    help="Known good commit (where test passes)",
)
@click.option(
    "--bad-commit",
    type=str,
    default=None,
    help="Known bad commit (where test fails)",
)
@click.option(
    "--text-threshold",
    type=float,
    default=None,
    help="Minimum text (gsm8k) accuracy threshold (0.0-1.0)",
)
@click.option(
    "--vision-threshold",
    type=float,
    default=None,
    help="Minimum vision (chartqa) accuracy threshold (0.0-1.0)",
)
@click.option(
    "--num-questions",
    type=int,
    default=320,
    help="Number of questions to ask per task (default: 320)",
)
@click.option(
    "--test",
    is_flag=True,
    default=False,
    help="Test mode: test current commit and return bisect exit code",
)
def main(
    model: str,
    good_commit: str | None,
    bad_commit: str | None,
    text_threshold: float | None,
    vision_threshold: float | None,
    num_questions: int,
    test: bool,
) -> None:
    """
    Git bisect helper for smoke test regressions.

    In normal mode, starts git bisect between good and bad commits.
    In test mode (--test), tests the current commit and returns
    appropriate exit code for git bisect.
    """
    if test:
        # Test mode: test current commit
        exit_code = test_commit(model, text_threshold, vision_threshold, num_questions)
        sys.exit(exit_code)
    else:
        # Bisect mode: need good and bad commits
        if good_commit is None or bad_commit is None:
            raise click.UsageError(
                "Both --good-commit and --bad-commit are required "
                "when not in --test mode"
            )

        exit_code = start_bisect(
            model,
            good_commit,
            bad_commit,
            text_threshold,
            vision_threshold,
            num_questions,
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
