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
from __future__ import annotations

"""
This script is used for the CI "Max Serve Smoke Test" workflow.
It does two things:
    1. Starts the MAX/SGLang/VLLM inference server for the given model
    2. Runs a tiny evaluation task using against the chat/completions API

Currently there is a hard dependency that two virtualenvs are already created:
    - .venv-serve (not needed for max-ci, which uses bazel)
    - .venv-eval

Where the serve environment should already have either MAX/VLLM/SGLang installed.
The eval environment should already have lm-eval installed.
These dependencies are to be removed once this script
has been integrated into bazel.

Note that if you're running this script inside bazel, only available for max-ci,
then the virtualenvs are not needed.
"""

import json
import logging
import os
import re
import signal
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from pprint import pformat
from subprocess import Popen, TimeoutExpired, check_call, check_output
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import click
import requests

DUMMY_2X2_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP8zwACTGCSAQANHQEDgslx/wAAAABJRU5ErkJggg=="
)
IMAGE_PROMPT = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Say 'hello image'"},
        {"type": "image_url", "image_url": {"url": DUMMY_2X2_IMAGE}},
    ],
}
TEXT_PROMPT = {"role": "user", "content": "Say: 'hello world'"}
URL = "http://127.0.0.1:8000/v1/chat/completions"

TEXT_TASK = "gsm8k_cot_llama"
VISION_TASK = "chartqa"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EvalResults = dict[str, Any]
EvalSamples = list[dict[str, Any]]

# Matches: [OP] LAUNCH kernel_name [id=N] detail_string
_OP_LAUNCH_RE = re.compile(
    r"\[OP\]\s+LAUNCH\s+(?P<kernel>\S+)\s+\[id=\d+\]\s*(?P<detail>.*)"
)


@dataclass
class KernelHit:
    """A single observed kernel invocation from op logging."""

    kernel: str
    detail: str

    @property
    def shape_key(self) -> str:
        """Return a deduplication key: kernel name + shapes."""
        return f"{self.kernel} | {self.detail}" if self.detail else self.kernel


@dataclass
class KernelProfile:
    """Aggregated kernel profile for a model."""

    model: str
    gpu_name: str
    gpu_count: int
    total_kernel_launches: int
    unique_kernels: int
    kernels: list[dict[str, Any]] = field(default_factory=list)


def parse_kernel_hits(stderr_lines: list[str]) -> list[KernelHit]:
    """Parse [OP] LAUNCH lines from server stderr into KernelHit objects."""
    hits: list[KernelHit] = []
    for line in stderr_lines:
        m = _OP_LAUNCH_RE.search(line)
        if m:
            hits.append(KernelHit(kernel=m.group("kernel"), detail=m.group("detail").strip()))
    return hits


def build_kernel_profile(
    model: str, hits: list[KernelHit]
) -> KernelProfile:
    """Aggregate kernel hits into a profile summary."""
    gpu_name, gpu_count = get_gpu_name_and_count()

    # Count by (kernel, detail) to group identical invocations
    counter: Counter[str] = Counter()
    detail_map: dict[str, str] = {}
    kernel_name_map: dict[str, str] = {}
    for hit in hits:
        key = hit.shape_key
        counter[key] += 1
        detail_map[key] = hit.detail
        kernel_name_map[key] = hit.kernel

    kernels = []
    for key, count in counter.most_common():
        entry: dict[str, Any] = {
            "kernel": kernel_name_map[key],
            "count": count,
        }
        detail = detail_map[key]
        if detail:
            entry["detail"] = detail
            # Parse semicolon-delimited key=value pairs from detail
            for part in detail.split(";"):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    entry[k.strip()] = v.strip()
        kernels.append(entry)

    return KernelProfile(
        model=model,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        total_kernel_launches=len(hits),
        unique_kernels=len(counter),
        kernels=kernels,
    )


def write_kernel_profile(path: Path, profile: KernelProfile) -> Path:
    """Write kernel profile to a JSON file and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    fp = path / "kernel_profile.json"
    fp.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    logger.info(f"Kernel profile written to {fp}")
    return fp


def is_deepseek(model: str) -> bool:
    """Temporary workaround for large DeepSeek models."""
    return "deepseek" in model and "lite" not in model


def validate_hf_token() -> None:
    if os.getenv("HF_TOKEN") is None:
        raise ValueError(
            "Environment variable `HF_TOKEN` must be set. "
            "See https://www.notion.so/modularai/HuggingFace-Access-Token-29d1044d37bb809fbe70e37428faf9da"
        )


def _inside_bazel() -> bool:
    return os.getenv("BUILD_WORKSPACE_DIRECTORY") is not None


def test_single_request(model: str, task: str) -> None:
    is_vision = task == VISION_TASK
    m = [IMAGE_PROMPT if is_vision else TEXT_PROMPT]

    r = requests.post(
        URL,
        json={"model": model, "messages": m, "max_tokens": 8},
        timeout=(30, 180),  # Initial req can be slow for huge models
    )
    r.raise_for_status()
    resp = r.json()["choices"][0]["message"]["content"]
    logger.info(f"Test single request OK. Response: {resp}")


@cache
def get_gpu_name_and_count() -> tuple[str, int]:
    """Returns the name and number of the available GPUs, e.g. ('MI300', 2)"""
    amd = ["amd-smi", "static", "--json"]
    nv = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    try:
        result = check_output(amd, text=True)
        data = json.loads(result.strip())["gpu_data"]
        return data[0]["asic"]["market_name"], len(data)
    except:
        try:
            lines = check_output(nv, text=True).strip().split("\n")
            return lines[0].strip(), len(lines)
        except:
            logger.warning("nvidia-smi and amd-smi both failed")
            return "N/A", 0


def server_is_ready() -> bool:
    health_url = "http://127.0.0.1:8000/health"
    try:
        return requests.get(health_url, timeout=1).status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_server_cmd(framework: str, model: str) -> list[str]:
    gpu_model, gpu_count = get_gpu_name_and_count()
    sglang_backend = "triton" if "b200" in gpu_model.lower() else "fa3"
    SGLANG = f"sglang.launch_server --attention-backend {sglang_backend} --mem-fraction-static 0.8"
    # limit-mm-per-prompt.video is for InternVL3 on B200
    VLLM = "vllm.entrypoints.openai.api_server --max-model-len 16384 --limit-mm-per-prompt.video 0"
    MAX = "max.entrypoints.pipelines serve"

    is_huge_model = is_deepseek(model)
    if is_huge_model and framework != "sglang":
        MAX += f" --device-memory-utilization 0.8 --devices gpu:{','.join(str(i) for i in range(gpu_count))} --ep-size {gpu_count} --data-parallel-degree {gpu_count} --max-batch-input-tokens 1024"
        VLLM += f" --enable-chunked-prefill --gpu-memory-utilization 0.8 --data-parallel-size={gpu_count} --enable-expert-parallel"
        # Have not been successful in getting SGLang to work with R1 yet
    elif gpu_count > 1:
        MAX += f" --devices gpu:{','.join(str(i) for i in range(gpu_count))}"
        VLLM += f" --tensor-parallel-size={gpu_count}"
        SGLANG += f" --tp-size={gpu_count}"

    if _inside_bazel():
        assert framework == "max-ci", "bazel invocation only supports max-ci"
        cmd = [sys.executable, "-m", *MAX.split()]
    else:
        assert framework != "max-ci", "max-ci must be run through bazel"
        interpreter = [".venv-serve/bin/python", "-m"]
        commands = {
            "sglang": [*interpreter, *SGLANG.split()],
            "vllm": [*interpreter, *VLLM.split()],
            "max": [*interpreter, *MAX.split()],
        }
        cmd = commands[framework]

    cmd = cmd + ["--port", "8000", "--trust-remote-code", "--model", model]

    # GPT-OSS uses repetition_penalty in lm_eval to prevent reasoning loops,
    # so we need to enable penalties on the server
    if "gpt-oss" in model and framework in ["max-ci", "max"]:
        cmd += ["--enable-penalties"]

    return cmd


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def call_eval(
    model: str, task: str, *, max_concurrent: int, num_questions: int
) -> tuple[EvalResults, EvalSamples]:
    extra_gen_kwargs = ""
    is_reasoning_model = any(
        kw in model
        for kw in (
            "academic-ds",
            "deepseek-r1",
            "gpt-oss",
            "internvl3_5",
            "qwen3",
        )
    )
    # Reasoning models needs extra tokens for .. reasoning
    if is_reasoning_model:
        extra_gen_kwargs = ",max_gen_toks=4096"

    # GPT-OSS sometimes gets stuck in a reasoning loop. To ensure consistency
    # in CI, we add a repetition penalty which helps prevent the loop
    if "gpt-oss" in model:
        extra_gen_kwargs = extra_gen_kwargs + ",repetition_penalty=1.1"

    interpreter = sys.executable if _inside_bazel() else ".venv-eval/bin/python"

    include_path = str(Path(__file__).parent.resolve() / "tasks")
    with TemporaryDirectory() as tempdir:
        eval_cmd = [
            "lm_eval",
            f"--tasks={task}",
            "--model=local-chat-completions",
            "--log_samples",
            f"--model_args=model={model},base_url={URL},num_concurrent={max_concurrent},max_retries=1",
            "--apply_chat_template",
            f"--output_path={tempdir}",
            f"--limit={num_questions}",
            "--seed=42",
            f"--gen_kwargs=seed=42,temperature=0{extra_gen_kwargs}",
            f"--include_path={include_path}",
            "--fewshot_as_multiturn",
        ]

        args = [interpreter, "-m", *eval_cmd]
        logger.info(f"Running eval with:\n {' '.join(args)}")
        check_call(args, timeout=600)

        return parse_eval_results(Path(tempdir))


def parse_eval_results(loc: Path) -> tuple[EvalResults, EvalSamples]:
    samples = []
    for line in open(next(loc.glob("**/samples*.jsonl")), encoding="utf-8"):
        samples.append(json.loads(line))

    results = json.loads(next(loc.glob("**/results*.json")).read_text("utf-8"))

    return results, samples


def write_github_output(key: str, value: str) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def gracefully_stop_process(process: Popen[bytes]) -> None:
    start_time = time.time()
    process.send_signal(signal.SIGINT)
    try:
        process.wait(25)
        shutdown_seconds = int(time.time() - start_time)
        logger.info(f"Server shutdown took {shutdown_seconds} seconds")
    except TimeoutExpired:
        logger.warning("Server did not stop after ctrl-c, trying SIGTERM")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(5)
        except ProcessLookupError:
            pass
        except TimeoutExpired:
            logger.warning("Process did not terminate gracefully, forcing kill")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(5)


@dataclass
class EvalSummary:
    gpu_name: str
    gpu_count: int
    startup_time_seconds: float
    eval_task: str
    task_type: str
    accuracy: float
    accuracy_stderr: float
    total_evaluation_time_seconds: float
    task_hash: str


def build_eval_summary(
    results: Sequence[Mapping[str, Any]],
    startup_time_seconds: float,
) -> list[EvalSummary]:
    """
    Extract the metrics from the eval results and build a summary for each task.
    """
    summaries = []

    for result in results:
        task = next(iter(result["results"].keys()))
        metrics = result["results"][task]
        total_secs = float(result["total_evaluation_time_seconds"])

        if VISION_TASK in task:
            accuracy = metrics["relaxed_accuracy,none"]
            accuracy_stderr = metrics["relaxed_accuracy_stderr,none"]
            task_type = "vision"
        elif task == TEXT_TASK:
            accuracy = metrics["exact_match,flexible-extract"]
            accuracy_stderr = metrics["exact_match_stderr,flexible-extract"]
            task_type = "text"
        else:
            raise ValueError(f"Unknown task: {task}")

        gpu_name, gpu_count = get_gpu_name_and_count()
        summaries.append(
            EvalSummary(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                startup_time_seconds=round(startup_time_seconds, 2),
                eval_task=task,
                task_type=task_type,
                accuracy=accuracy,
                accuracy_stderr=accuracy_stderr,
                total_evaluation_time_seconds=total_secs,
                task_hash=result["task_hashes"][task],
            )
        )

    return summaries


def print_samples(samples: EvalSamples, print_cot: bool) -> None:
    """
    Print question and the model's responses to each sample
    Assumes 'resps' is [[str]] (one decode) and GSM8K uses 'question',
    ChartQA uses 'query'.
    """
    for item in samples:
        doc = item.get("doc", {})
        question = doc.get("question") or doc.get("query")

        filt = item["filtered_resps"]
        extracted = filt[0] if isinstance(filt, list) and filt else str(filt)

        status = "✅" if item["exact_match"] == 1.0 else "❌"
        prefix_q = "🧮" if "question" in doc else "📊"

        logger.info(f"{prefix_q} {question}")
        if print_cot:
            logger.info(f"🤖💭 {item['resps'][0][0]}")
        logger.info(f"{status} {extracted}")


def start_server(
    cmd: list[str],
    timeout: int,
    *,
    kernel_profile: bool = False,
    stderr_log: Path | None = None,
) -> tuple[Popen[bytes], float]:
    env = os.environ.copy()

    if kernel_profile:
        env["MOJO_LOGGING_LEVEL"] = "trace"

    if not _inside_bazel():
        # SGLang depends on ninja which is in the serve environment
        env["PYTHONSAFEPATH"] = "1"  # Avoids root dir `max` shadowing
        venv_bin = os.path.abspath(".venv-serve/bin")
        prev_path = env.get("PATH")
        env["PATH"] = f"{venv_bin}:{prev_path}" if prev_path else venv_bin

    # When profiling kernels, tee stderr to a file so we can parse [OP] lines.
    stderr_dest: int | None = None
    stderr_file = None
    if stderr_log is not None:
        stderr_file = open(stderr_log, "wb")
        stderr_dest = stderr_file.fileno()

    start = time.monotonic()
    proc = Popen(cmd, start_new_session=True, env=env, stderr=stderr_dest)
    try:
        deadline = start + timeout
        while time.monotonic() < deadline:
            if server_is_ready():
                break
            if proc.poll() is not None:
                if stderr_file:
                    stderr_file.close()
                raise RuntimeError("Server process terminated unexpectedly")
            time.sleep(0.5)
        else:
            if stderr_file:
                stderr_file.close()
            raise TimeoutError(f"Server did not start in {timeout} seconds")
        return proc, time.monotonic() - start
    except:
        gracefully_stop_process(proc)
        if stderr_file:
            stderr_file.close()
        raise


def write_results(
    path: Path,
    summary: list[EvalSummary],
    results: list[EvalResults],
    all_samples: list[EvalSamples],
    tasks: list[str],
) -> None:
    summary_file = path / "eval_metrics.json"
    summary_json = json.dumps([asdict(s) for s in summary], indent=2)
    summary_file.write_text(summary_json, encoding="utf-8")
    for result, samples, task in zip(results, all_samples, tasks, strict=True):
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        result_fp = path / f"results_{task}_{timestamp}.json"
        result_fp.write_text(json.dumps(result, indent=2), encoding="utf-8")

        samples_fp = path / f"samples_{task}_{timestamp}.jsonl"
        with open(samples_fp, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


@click.command()
@click.argument(
    "hf_model_path",
    type=str,
    required=True,
)
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "max", "max-ci"]),
    default="max-ci",
    required=False,
    help="Framework to use for the smoke test. Only max-ci is supported when running in bazel.",
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help="If provided, a summary json file and the eval result are written here",
)
@click.option(
    "--print-responses",
    is_flag=True,
    default=False,
    help="Print question/response pairs from eval samples after the run finishes",
)
@click.option(
    "--print-cot",
    is_flag=True,
    default=False,
    help="Print the model's chain-of-thought reasoning for each sample. Must be used with --print-responses",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=64,
    help="Maximum concurrent requests to send to the server",
)
@click.option(
    "--num-questions",
    type=int,
    default=320,
    help="Number of questions to ask the model",
)
@click.option(
    "--kernel-profile",
    is_flag=True,
    default=False,
    help=(
        "Instead of running eval, start the server with Mojo op logging enabled, "
        "send a single request to trigger compilation, and write a kernel_profile.json "
        "listing every kernel hit with its shapes. Requires --output-path."
    ),
)
def smoke_test(
    hf_model_path: str,
    framework: str,
    output_path: Path | None,
    print_responses: bool,
    print_cot: bool,
    max_concurrent: int,
    num_questions: int,
    kernel_profile: bool,
) -> None:
    """
    Example usage: ./bazelw run smoke-test -- meta-llama/llama-3.2-1b-instruct

    This command asks 320 questions against the model behind the given hf_model_path.
    It runs the provided framework (defaulting to MAX serve) in the background,
    and fires off HTTP requests to chat/completions API.
    Note: Only models with a chat template (typically -instruct, -it, -chat, etc.) are supported.

    Accuracy is reported at the end, with higher values being better.
    A 1.0 value means 100% accuracy.

    """
    validate_hf_token()

    if print_cot and not print_responses:
        raise ValueError("--print-cot must be used with --print-responses")

    if kernel_profile and framework not in ("max", "max-ci"):
        raise click.UsageError(
            "--kernel-profile only works with MAX frameworks (max or max-ci)"
        )

    if kernel_profile and output_path is None:
        raise click.UsageError("--kernel-profile requires --output-path")

    build_workspace = os.getenv("BUILD_WORKSPACE_DIRECTORY")
    if output_path and build_workspace and not output_path.is_absolute():
        output_path = Path(build_workspace) / output_path

    model = hf_model_path.lower().strip()
    cmd = get_server_cmd(framework, model)

    # TODO Refactor this to a model list/matrix specifying type of model
    is_vision_model = any(
        kw in model
        for kw in (
            "gemma-3",
            "idefics",
            "internvl",
            "olmocr",
            "pixtral",
            "qwen2.5-vl",
            "qwen3-vl",
            "vision",
        )
    )
    # 1b is non-vision
    if "gemma-3-1b" in model:
        is_vision_model = False

    tasks = [TEXT_TASK]
    if is_vision_model:
        tasks = [VISION_TASK] + tasks

    logger.info(f"Starting server with command:\n {' '.join(cmd)}")
    timeout = 900
    if is_deepseek(model):
        timeout = 1800

    # --- Kernel profile mode ---
    if kernel_profile:
        assert output_path is not None  # Enforced above
        profile_output = output_path / safe_model_name(model)

        with NamedTemporaryFile(
            mode="w+", suffix=".log", prefix="kernel_profile_", delete=False
        ) as stderr_tmp:
            stderr_log = Path(stderr_tmp.name)

        logger.info(
            "Kernel profile mode: starting server with MOJO_LOGGING_LEVEL=trace"
        )
        server_process, startup_time = start_server(
            cmd, timeout, kernel_profile=True, stderr_log=stderr_log
        )
        try:
            logger.info(f"Server started in {startup_time:.2f} seconds")

            # Fire a single request to trigger compilation and one inference pass
            task = VISION_TASK if is_vision_model else TEXT_TASK
            test_single_request(model, task)
            logger.info("Single request completed, collecting kernel trace...")
        finally:
            try:
                gracefully_stop_process(server_process)
            except Exception:
                logger.exception(f"Failed to shutdown {framework.upper()}")

        # Parse the captured stderr for [OP] LAUNCH lines
        stderr_lines = stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()
        stderr_log.unlink(missing_ok=True)

        hits = parse_kernel_hits(stderr_lines)
        logger.info(f"Captured {len(hits)} kernel launches")

        profile = build_kernel_profile(model, hits)
        fp = write_kernel_profile(profile_output, profile)

        # Summary to stdout
        logger.info(
            f"Kernel profile: {profile.total_kernel_launches} launches, "
            f"{profile.unique_kernels} unique kernels"
        )
        for entry in profile.kernels[:20]:
            logger.info(
                f"  {entry['count']:>5}x  {entry['kernel']}"
                + (f"  ({entry.get('detail', '')})" if entry.get("detail") else "")
            )
        if len(profile.kernels) > 20:
            logger.info(f"  ... and {len(profile.kernels) - 20} more (see {fp})")
        return

    # --- Normal eval mode ---
    results = []
    all_samples = []

    server_process, startup_time = start_server(cmd, timeout)
    try:
        logger.info(f"Server started in {startup_time:.2f} seconds")
        write_github_output("startup_time", f"{startup_time:.2f}")

        for task in tasks:
            test_single_request(model, task)
            result, samples = call_eval(
                model,
                task,
                max_concurrent=max_concurrent,
                num_questions=num_questions,
            )
            if print_responses:
                print_samples(samples, print_cot)

            results.append(result)
            all_samples.append(samples)
    finally:
        try:
            gracefully_stop_process(server_process)
        except Exception:
            logger.exception(f"Failed to shutdown {framework.upper()}")

    if results:
        summary = build_eval_summary(results, startup_time_seconds=startup_time)

        if output_path is not None:
            path = output_path / safe_model_name(model)
            path.mkdir(parents=True, exist_ok=True)
            write_results(path, summary, results, all_samples, tasks)

            logger.info(pformat(summary, indent=2))


if __name__ == "__main__":
    smoke_test()
