"""Benchmark the RAG pipeline on CPU: retrieval cost, then INT4 vs FP16 inference.

Each model/quantization config runs in a fresh spawned subprocess so its RAM
figures are not polluted by the embedding model (torch) loaded for retrieval, or
by memory a previous model failed to return to the OS. RAM is process RSS
sampled in a background thread, since llama.cpp allocates natively and is
invisible to ``tracemalloc``.

Usage::

    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --models qwen2.5-1.5b llama-3.2-1b --quants q4_k_m q8_0 f16
    python benchmarks/run_benchmark.py --doc README.md --runs 5 --output benchmarks/results.md
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import shutil
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import psutil

# Allow `python benchmarks/run_benchmark.py` from the repo root. Spawned workers
# re-import this module, so this also runs in each child process.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.quant_engine import DEFAULT_MODEL, MODEL_PRESETS  # noqa: E402  (no heavy deps at import)

DEFAULT_QUERY = "How does INT4 quantization reduce memory usage on edge devices?"
SAMPLE_DOC = """\
# Quantization for Edge Inference

Quantization stores model weights at lower numeric precision. A 16-bit floating point
(FP16) weight takes two bytes, while a 4-bit integer (INT4) weight takes half a byte, so
INT4 cuts weight memory by roughly 4x compared with FP16. GGUF formats such as Q4_K_M
group weights into blocks that share a scale factor, which keeps accuracy loss small.

On CPUs, token generation is usually memory-bandwidth bound: every generated token must
stream the full set of weights from RAM. Smaller weights therefore mean more tokens per
second, not just a smaller footprint. Prompt processing (prefill) is more compute bound,
which is why time-to-first-token scales with prompt length and thread count.

## Retrieval-Augmented Generation

RAG retrieves relevant passages from a local document store and injects them into the
prompt. Here documents are chunked into 500-character windows with 50 characters of
overlap, embedded with all-MiniLM-L6-v2 into 384-dimensional vectors, and searched with a
FAISS inner-product index over normalized vectors, which is equivalent to cosine
similarity. Keeping retrieval small lets the whole pipeline run on a laptop or edge box.
"""


# --------------------------------------------------------------------------- memory


def rss_mb(proc: psutil.Process | None = None) -> float:
    """Resident set size of ``proc`` (default: this process) in MiB."""
    return (proc or psutil.Process()).memory_info().rss / 2**20


class PeakRSSSampler:
    """Context manager that samples process RSS in a background thread and keeps the peak.

    llama.cpp calls go through ctypes, which releases the GIL, so the sampler keeps
    running while native inference is busy.
    """

    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb(self._proc))
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PeakRSSSampler:
        self.peak_mb = rss_mb(self._proc)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, rss_mb(self._proc))


# --------------------------------------------------------------------------- retrieval


@dataclass
class RetrievalStats:
    """Cost of building and querying the FAISS index."""

    source: str
    num_chunks: int
    rss_before_mb: float
    rss_after_mb: float
    build_s: float
    load_s: float
    search_ms: float
    top_chunks: list[str] = field(default_factory=list)


def resolve_docs_dir(doc: Path | None, docs_dir: Path, workdir: Path) -> tuple[Path, str]:
    """Pick the corpus: an explicit file, else ``docs_dir`` if non-empty, else a built-in sample.

    Returns ``(directory_to_ingest, human_readable_label)``.
    """
    from src.retrieval.vector_store import SUPPORTED_EXTENSIONS

    target = workdir / "docs"
    target.mkdir()
    if doc is not None:
        if not doc.is_file():
            raise FileNotFoundError(f"Document not found: {doc}")
        if doc.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"{doc} must be one of {sorted(SUPPORTED_EXTENSIONS)}")
        shutil.copy(doc, target / doc.name)
        return target, str(doc)
    if docs_dir.is_dir() and any(
        p.suffix.lower() in SUPPORTED_EXTENSIONS for p in docs_dir.rglob("*") if p.is_file()
    ):
        return docs_dir, str(docs_dir)
    (target / "sample.md").write_text(SAMPLE_DOC, encoding="utf-8")
    return target, "built-in sample.md"


def benchmark_retrieval(
    docs_dir: Path, label: str, index_dir: Path, query: str, top_k: int, runs: int
) -> RetrievalStats:
    """Build, save, reload and query a FAISS index, timing each step."""
    from src.retrieval.vector_store import VectorStore

    rss_before = rss_mb()
    start = time.perf_counter()
    store = VectorStore()
    store.build(docs_dir)  # includes embedding-model load on first call
    store.save(index_dir)
    build_s = time.perf_counter() - start

    start = time.perf_counter()
    loaded = VectorStore.load(index_dir)
    loaded._model = store._model  # reuse the already-loaded embedder; we time index I/O only
    load_s = time.perf_counter() - start

    loaded.search(query, top_k)  # warm-up
    start = time.perf_counter()
    for _ in range(runs):
        results = loaded.search(query, top_k)
    search_ms = (time.perf_counter() - start) / runs * 1000

    return RetrievalStats(
        source=label,
        num_chunks=len(loaded.chunks),
        rss_before_mb=rss_before,
        rss_after_mb=rss_mb(),
        build_s=build_s,
        load_s=load_s,
        search_ms=search_ms,
        top_chunks=[r.chunk.text for r in results],
    )


# --------------------------------------------------------------------------- inference


@dataclass(frozen=True)
class InferenceConfig:
    """One model/quantization pairing to benchmark."""

    model: str
    quant: str

    @property
    def label(self) -> str:
        return f"{self.model} {self.quant}"


def download_model(cfg: InferenceConfig) -> Path:
    """Fetch the GGUF file into the HF cache so download time never counts as load time."""
    from huggingface_hub import hf_hub_download

    spec = MODEL_PRESETS[cfg.model]
    if cfg.quant not in spec.files:
        raise ValueError(f"Unknown quant {cfg.quant!r} for {cfg.model}; choose from {sorted(spec.files)}")
    return Path(hf_hub_download(repo_id=spec.repo_id, filename=spec.files[cfg.quant]))


def run_inference_benchmark(
    model_path: str,
    chunks: Sequence[str],
    query: str,
    runs: int,
    max_tokens: int,
    n_ctx: int,
    n_threads: int | None,
) -> dict[str, Any]:
    """Load one GGUF model and time ``runs`` generations. Runs inside a worker process."""
    from src.core.quant_engine import QuantizedSLMEngine

    proc = psutil.Process()
    rss_base = rss_mb(proc)

    start = time.perf_counter()
    engine = QuantizedSLMEngine(model_path=Path(model_path), n_ctx=n_ctx, n_threads=n_threads)
    load_s = time.perf_counter() - start
    rss_loaded = rss_mb(proc)

    # Warm-up pages in the mmapped weights and initializes threadpools.
    engine.llm.reset()
    engine.generate(query, chunks, max_tokens=8, temperature=0.0)

    ttft, tps, latency, tokens, cpu_s, cpu_cores = [], [], [], [], [], []
    peak = rss_loaded
    answer = ""
    for _ in range(runs):
        engine.llm.reset()  # drop the KV prefix cache so every run pays full prefill
        cpu_before = proc.cpu_times()
        with PeakRSSSampler() as sampler:
            result = engine.generate(query, chunks, max_tokens=max_tokens, temperature=0.0)
        cpu_after = proc.cpu_times()
        used = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)

        peak = max(peak, sampler.peak_mb)
        ttft.append(result.time_to_first_token_s * 1000)
        tps.append(result.tokens_per_second)
        latency.append(result.total_time_s)
        tokens.append(result.completion_tokens)
        cpu_s.append(used)
        cpu_cores.append(used / result.total_time_s if result.total_time_s > 0 else 0.0)
        answer = result.text

    return {
        "file_mb": os.path.getsize(model_path) / 2**20,
        "load_s": load_s,
        "rss_base_mb": rss_base,
        "rss_loaded_mb": rss_loaded,
        "rss_peak_mb": peak,
        "ttft_ms": ttft,
        "tokens_per_s": tps,
        "latency_s": latency,
        "completion_tokens": tokens,
        "cpu_s": cpu_s,
        "cpu_cores_busy": cpu_cores,
        "n_threads": engine.llm.n_threads,
        "answer": answer,
    }


def _worker(queue: mp.Queue, kwargs: dict[str, Any]) -> None:
    """Subprocess entrypoint: report results or the error through ``queue``."""
    try:
        queue.put(run_inference_benchmark(**kwargs))
    except Exception as exc:  # surfaced in the report rather than crashing the whole run
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def run_isolated(kwargs: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Run :func:`run_inference_benchmark` in a fresh spawned process."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, kwargs))
    proc.start()
    try:
        result = queue.get(timeout=timeout_s)  # read before join to avoid a full-pipe deadlock
    except Exception:
        result = {"error": f"worker produced no result (exit code {proc.exitcode}, timeout {timeout_s}s)"}
    proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
    return result


# --------------------------------------------------------------------------- reporting


def _mean_sd(values: Sequence[float], fmt: str) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return format(values[0], fmt)
    return f"{format(statistics.mean(values), fmt)} +/- {format(statistics.stdev(values), fmt)}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a GitHub-flavored Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def system_info() -> list[tuple[str, str]]:
    """Host facts that explain the numbers."""
    try:
        import llama_cpp

        llama_version = llama_cpp.__version__
    except ImportError:
        llama_version = "not installed"
    vm = psutil.virtual_memory()
    return [
        ("OS", f"{platform.system()} {platform.release()} ({platform.machine()})"),
        ("CPU", platform.processor() or "unknown"),
        ("Cores (physical / logical)", f"{psutil.cpu_count(logical=False)} / {psutil.cpu_count()}"),
        ("RAM (total / available)", f"{vm.total / 2**30:.1f} GiB / {vm.available / 2**30:.1f} GiB"),
        ("Python", platform.python_version()),
        ("llama-cpp-python", llama_version),
    ]


def build_report(
    args: argparse.Namespace,
    retrieval: RetrievalStats,
    results: list[tuple[InferenceConfig, dict[str, Any]]],
) -> str:
    """Assemble the full Markdown report."""
    out = ["# Light-RAG CPU Benchmark", ""]
    out += ["## System", "", md_table(["Property", "Value"], system_info()), ""]

    out += ["## Retrieval (FAISS + all-MiniLM-L6-v2)", ""]
    out.append(md_table(
        ["Corpus", "Chunks", "Build + embed (s)", "Index load (s)", f"Search top-{args.top_k} (ms)",
         "RSS before (MB)", "RSS after (MB)"],
        [[retrieval.source, retrieval.num_chunks, f"{retrieval.build_s:.2f}", f"{retrieval.load_s:.3f}",
          f"{retrieval.search_ms:.2f}", f"{retrieval.rss_before_mb:.0f}", f"{retrieval.rss_after_mb:.0f}"]],
    ))
    out.append("")

    out += [
        "## Inference",
        "",
        f"Query: *{args.query}*; {len(retrieval.top_chunks)} context chunks; "
        f"max_tokens={args.max_tokens}; temperature=0; "
        f"threads={next((r['n_threads'] for _, r in results if 'error' not in r), args.threads or 'auto')}; "
        f"{args.runs} timed run(s) after 1 warm-up; values are mean +/- sd",
        "",
    ]
    rows = []
    for cfg, r in results:
        if "error" in r:
            rows.append([cfg.label, "error: " + r["error"]] + [""] * 9)
            continue
        rows.append([
            cfg.label,
            f"{r['file_mb']:.0f}",
            f"{r['load_s']:.2f}",
            f"{r['rss_base_mb']:.0f}",
            f"{r['rss_loaded_mb']:.0f}",
            f"{r['rss_peak_mb']:.0f}",
            f"{r['rss_peak_mb'] - r['rss_base_mb']:.0f}",
            _mean_sd(r["ttft_ms"], ".0f"),
            _mean_sd(r["tokens_per_s"], ".1f"),
            _mean_sd(r["latency_s"], ".2f"),
            _mean_sd(r["cpu_cores_busy"], ".1f"),
        ])
    out.append(md_table(
        ["Config", "File (MB)", "Load (s)", "RSS base (MB)", "RSS loaded (MB)", "RSS peak (MB)",
         "Peak - base (MB)", "TTFT (ms)", "Tokens/s", "Latency (s)", "Avg cores busy"],
        rows,
    ))
    out.append("")
    out += [
        "*RSS base* is the worker before model load; *Peak - base* is the model's total footprint during "
        "inference. *Avg cores busy* = process CPU time / wall time. llama.cpp mmaps weights, so RSS "
        "counts only pages actually touched.",
        "",
    ]

    comparisons = _speedups(results)
    if comparisons:
        out += ["## Quantized vs F16", "", md_table(
            ["Model", "Quant", "Tokens/s speedup", "TTFT speedup", "Peak RAM saved"], comparisons
        ), ""]

    ok = [(cfg, r) for cfg, r in results if "error" not in r and r["answer"]]
    if ok:
        out += ["## Sample answers", ""]
        for cfg, r in ok:
            out += [f"**{cfg.label}:** {r['answer'].strip()}", ""]
    return "\n".join(out)


def _speedups(results: list[tuple[InferenceConfig, dict[str, Any]]]) -> list[list[str]]:
    """Compare each non-f16 quant against the f16 run of the same model."""
    ok = {(c.model, c.quant): r for c, r in results if "error" not in r}
    rows = []
    for (model, quant), r in ok.items():
        base = ok.get((model, "f16"))
        if quant == "f16" or base is None:
            continue
        tps, base_tps = statistics.mean(r["tokens_per_s"]), statistics.mean(base["tokens_per_s"])
        ttft, base_ttft = statistics.mean(r["ttft_ms"]), statistics.mean(base["ttft_ms"])
        mem = r["rss_peak_mb"] - r["rss_base_mb"]
        base_mem = base["rss_peak_mb"] - base["rss_base_mb"]
        rows.append([
            model,
            quant,
            f"{tps / base_tps:.2f}x" if base_tps else "-",
            f"{base_ttft / ttft:.2f}x" if ttft else "-",
            f"{(1 - mem / base_mem) * 100:.0f}%" if base_mem > 0 else "-",
        ])
    return rows


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    """Run retrieval and inference benchmarks and print a Markdown report."""
    parser = argparse.ArgumentParser(description="Benchmark Light-RAG retrieval and INT4 vs FP16 inference on CPU.")
    parser.add_argument("--doc", type=Path, help="Single .md/.txt file to index (default: data/docs or a built-in sample).")
    parser.add_argument("--docs-dir", type=Path, default=Path("data/docs"))
    parser.add_argument("-q", "--query", default=DEFAULT_QUERY)
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_PRESETS), default=[DEFAULT_MODEL])
    parser.add_argument("--quants", nargs="+", default=["q4_k_m", "f16"], help="e.g. q4_k_m q8_0 f16")
    parser.add_argument("--runs", type=int, default=3, help="Timed generations per config (after 1 warm-up).")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=None, help="llama.cpp threads (default: library choice).")
    parser.add_argument("--timeout", type=float, default=1800, help="Per-config worker timeout in seconds.")
    parser.add_argument("--output", type=Path, help="Also write the Markdown report to this file.")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    configs = [InferenceConfig(m, q) for m in args.models for q in args.quants]

    with tempfile.TemporaryDirectory(prefix="lightrag-bench-") as tmp:
        workdir = Path(tmp)
        print("[1/3] Building FAISS index...", file=sys.stderr)
        docs_dir, label = resolve_docs_dir(args.doc, args.docs_dir, workdir)
        retrieval = benchmark_retrieval(docs_dir, label, workdir / "index", args.query, args.top_k, runs=20)

    results: list[tuple[InferenceConfig, dict[str, Any]]] = []
    for i, cfg in enumerate(configs, 1):
        print(f"[2/3] ({i}/{len(configs)}) {cfg.label}: downloading if needed...", file=sys.stderr)
        try:
            model_path = download_model(cfg)
        except Exception as exc:
            results.append((cfg, {"error": f"download failed: {exc}"}))
            continue
        print(f"[2/3] ({i}/{len(configs)}) {cfg.label}: benchmarking in subprocess...", file=sys.stderr)
        results.append((cfg, run_isolated(
            dict(
                model_path=str(model_path),
                chunks=retrieval.top_chunks,
                query=args.query,
                runs=args.runs,
                max_tokens=args.max_tokens,
                n_ctx=args.n_ctx,
                n_threads=args.threads,
            ),
            timeout_s=args.timeout,
        )))

    print("[3/3] Done.\n", file=sys.stderr)
    report = build_report(args, retrieval, results)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nReport written to {args.output}", file=sys.stderr)
    return 0 if all("error" not in r for _, r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
