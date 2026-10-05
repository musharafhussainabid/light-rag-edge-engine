"""Quantized GGUF small-language-model engine for CPU-only RAG generation.

Models are pulled from the HuggingFace Hub on first use via
``llama_cpp.Llama.from_pretrained`` and cached locally. Prompts are built with the
chat template embedded in the GGUF metadata, so Qwen2.5 and Llama-3.2 both work
without hand-written template strings.

CLI usage::

    python -m src.core.quant_engine -q "What is quantization?"
    python -m src.core.quant_engine --model llama-3.2-1b --quant f16 --no-stream
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from llama_cpp import Llama

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """A HuggingFace GGUF repo and the exact filename for each supported quantization."""

    repo_id: str
    files: dict[str, str]


MODEL_PRESETS: dict[str, ModelSpec] = {
    "qwen2.5-1.5b": ModelSpec(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        files={
            "q4_k_m": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "q8_0": "qwen2.5-1.5b-instruct-q8_0.gguf",
            "f16": "qwen2.5-1.5b-instruct-fp16.gguf",
        },
    ),
    "llama-3.2-1b": ModelSpec(
        repo_id="unsloth/Llama-3.2-1B-Instruct-GGUF",
        files={
            "q4_k_m": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "q8_0": "Llama-3.2-1B-Instruct-Q8_0.gguf",
            "f16": "Llama-3.2-1B-Instruct-F16.gguf",
        },
    ),
}
DEFAULT_MODEL = "qwen2.5-1.5b"
DEFAULT_QUANT = "q4_k_m"

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant that answers questions using only the provided context. "
    "Cite the passages you use by their number, e.g. [1]. If the context does not contain "
    "the answer, say you don't know rather than guessing."
)
NO_CONTEXT_MARKER = "(no context retrieved)"


@dataclass(frozen=True)
class GenerationResult:
    """A completed generation with the timing figures used by the benchmarks."""

    text: str
    completion_tokens: int
    time_to_first_token_s: float
    total_time_s: float

    @property
    def tokens_per_second(self) -> float:
        """Decode throughput, excluding prompt processing (time to first token)."""
        decode_time = self.total_time_s - self.time_to_first_token_s
        if self.completion_tokens <= 1 or decode_time <= 0:
            return 0.0
        return (self.completion_tokens - 1) / decode_time


class QuantizedSLMEngine:
    """CPU inference over a GGUF-quantized instruct model with RAG prompt assembly."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        quant: str = DEFAULT_QUANT,
        model_path: Path | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = False,
    ) -> None:
        """Load the model.

        Args:
            model: Key into :data:`MODEL_PRESETS`. Ignored when ``model_path`` is set.
            quant: Quantization key for the preset (``q4_k_m``, ``q8_0``, ``f16``).
            model_path: Local ``.gguf`` file to use instead of downloading a preset.
            n_ctx: Context window in tokens; shared by system prompt, chunks, query and answer.
            n_threads: CPU threads for inference; ``None`` lets llama.cpp choose.
            system_prompt: Instruction prepended to every request.
            verbose: Forward llama.cpp's native logging to stderr.
        """
        self.n_ctx = n_ctx
        self.system_prompt = system_prompt
        llama_kwargs = dict(n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=0, verbose=verbose)

        if model_path is not None:
            self.model_name = Path(model_path).name
            logger.info("Loading local model %s", model_path)
            self.llm = Llama(model_path=str(model_path), **llama_kwargs)
        else:
            if model not in MODEL_PRESETS:
                raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODEL_PRESETS)}")
            spec = MODEL_PRESETS[model]
            if quant not in spec.files:
                raise ValueError(f"Unknown quant {quant!r} for {model}; choose from {sorted(spec.files)}")
            self.model_name = f"{model}-{quant}"
            logger.info("Loading %s/%s (downloads on first use)", spec.repo_id, spec.files[quant])
            self.llm = Llama.from_pretrained(
                repo_id=spec.repo_id, filename=spec.files[quant], **llama_kwargs
            )

    def build_messages(self, query: str, context_chunks: Sequence[str]) -> list[dict[str, str]]:
        """Wrap numbered context chunks and the query into chat messages."""
        if context_chunks:
            context = "\n\n".join(f"[{i}] {chunk.strip()}" for i, chunk in enumerate(context_chunks, 1))
        else:
            context = NO_CONTEXT_MARKER
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.strip()}"},
        ]

    def _fit_context(self, query: str, context_chunks: Sequence[str], max_tokens: int) -> list[str]:
        """Drop the lowest-ranked (trailing) chunks until the prompt fits in ``n_ctx``.

        Chunks are assumed to be ordered best-first, as returned by the retriever.
        The template overhead is estimated from the rendered message text plus a
        small allowance for role/special tokens.
        """
        chunks = list(context_chunks)
        template_overhead = 32
        while True:
            text = "\n".join(m["content"] for m in self.build_messages(query, chunks))
            prompt_tokens = len(self.llm.tokenize(text.encode("utf-8"), add_bos=False)) + template_overhead
            if prompt_tokens + max_tokens <= self.n_ctx:
                return chunks
            if not chunks:
                raise ValueError(
                    f"Query alone needs ~{prompt_tokens} tokens; exceeds n_ctx={self.n_ctx} "
                    f"with max_tokens={max_tokens}"
                )
            logger.warning("Prompt exceeds n_ctx=%d; dropping context chunk %d", self.n_ctx, len(chunks))
            chunks.pop()

    def stream(
        self,
        query: str,
        context_chunks: Sequence[str] = (),
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Yield the answer incrementally as text deltas."""
        messages = self.build_messages(query, self._fit_context(query, context_chunks, max_tokens))
        for event in self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True
        ):
            delta = event["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    def generate(
        self,
        query: str,
        context_chunks: Sequence[str] = (),
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> GenerationResult:
        """Run a full generation and return the text with latency/throughput figures.

        Consumes :meth:`stream` so time-to-first-token can be measured. The
        completion token count is obtained by re-tokenizing the output, which
        matches the sampled token count for practically all text.
        """
        parts: list[str] = []
        ttft: float | None = None
        start = time.perf_counter()
        for delta in self.stream(query, context_chunks, max_tokens, temperature):
            if ttft is None:
                ttft = time.perf_counter() - start
            parts.append(delta)
        total = time.perf_counter() - start

        text = "".join(parts)
        completion_tokens = len(self.llm.tokenize(text.encode("utf-8"), add_bos=False)) if text else 0
        return GenerationResult(
            text=text,
            completion_tokens=completion_tokens,
            time_to_first_token_s=ttft if ttft is not None else total,
            total_time_s=total,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: retrieve context from the local index and stream an answer."""
    from src.retrieval.vector_store import DEFAULT_DOCS_DIR, DEFAULT_INDEX_DIR, VectorStore

    parser = argparse.ArgumentParser(description="Answer a question over local docs with a quantized SLM.")
    parser.add_argument("-q", "--query", default="What is this project about?")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default=DEFAULT_MODEL)
    parser.add_argument("--quant", default=DEFAULT_QUANT, help="q4_k_m, q8_0 or f16")
    parser.add_argument("--model-path", type=Path, help="Local .gguf file (overrides --model/--quant).")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--no-context", action="store_true", help="Skip retrieval and query the model directly.")
    parser.add_argument("--no-stream", action="store_true", help="Print the full answer with timing stats.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    chunks: list[str] = []
    if not args.no_context:
        try:
            store = VectorStore.load_or_build(args.docs_dir, args.index_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s (use --no-context to skip retrieval)", exc)
            return 1
        results = store.search(args.query, args.top_k)
        chunks = [r.chunk.text for r in results]
        for i, r in enumerate(results, 1):
            logger.info("[%d] %s#chunk%d score=%.3f", i, r.chunk.source, r.chunk.chunk_id, r.score)

    engine = QuantizedSLMEngine(model=args.model, quant=args.quant, model_path=args.model_path)
    print(f"\nQ: {args.query}\nA: ", end="", flush=True)
    if args.no_stream:
        result = engine.generate(args.query, chunks, max_tokens=args.max_tokens)
        print(result.text)
        print(
            f"\n[{engine.model_name}] {result.completion_tokens} tokens, "
            f"TTFT {result.time_to_first_token_s:.2f}s, total {result.total_time_s:.2f}s, "
            f"{result.tokens_per_second:.1f} tok/s"
        )
    else:
        for delta in engine.stream(args.query, chunks, max_tokens=args.max_tokens):
            print(delta, end="", flush=True)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
