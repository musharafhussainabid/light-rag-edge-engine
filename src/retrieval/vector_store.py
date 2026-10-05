"""Local document ingestion and FAISS-backed vector retrieval.

Pipeline: read ``.md``/``.txt`` files -> fixed-size character chunks with overlap
-> embed with ``all-MiniLM-L6-v2`` -> store in a FAISS inner-product index over
L2-normalised vectors (i.e. cosine similarity).

CLI usage::

    python -m src.retrieval.vector_store --query "What is quantization?" -k 3
    python -m src.retrieval.vector_store --rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import faiss
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DOCS_DIR = Path("data/docs")
DEFAULT_INDEX_DIR = Path("data/index")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt"})

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "chunks.json"


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a source document."""

    text: str
    source: str
    chunk_id: int
    start_char: int


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with its cosine similarity to the query."""

    chunk: Chunk
    score: float


def iter_documents(docs_dir: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, text)`` for every supported file under ``docs_dir``, recursively.

    Files are visited in sorted order so index builds are deterministic.
    """
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                yield path, text


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[int, str]]:
    """Split ``text`` into fixed-size character windows with ``overlap`` chars shared.

    Returns a list of ``(start_char, chunk_text)``. Whitespace-only chunks are dropped.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunks: list[tuple[int, str]] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if piece.strip():
            chunks.append((start, piece.strip()))
        if start + chunk_size >= len(text):
            break
    return chunks


class VectorStore:
    """FAISS index over document chunks, with save/load to a local directory."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily load the embedding model (auto-downloads from HF Hub on first use)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed texts into L2-normalised float32 vectors."""
        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def build(
        self,
        docs_dir: Path = DEFAULT_DOCS_DIR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Ingest, chunk, and embed all documents in ``docs_dir`` into a fresh index."""
        chunks: list[Chunk] = []
        for path, text in iter_documents(docs_dir):
            source = path.relative_to(docs_dir).as_posix()
            for i, (start, piece) in enumerate(chunk_text(text, chunk_size, overlap)):
                chunks.append(Chunk(text=piece, source=source, chunk_id=i, start_char=start))

        if not chunks:
            raise ValueError(f"No non-empty {sorted(SUPPORTED_EXTENSIONS)} files in {docs_dir}")

        logger.info("Embedding %d chunks from %s", len(chunks), docs_dir)
        vectors = self._embed([c.text for c in chunks], show_progress=True)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index
        self.chunks = chunks

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        """Return the top-``k`` chunks most similar to ``query``."""
        if self.index is None:
            raise RuntimeError("Index is empty; call build() or load() first")
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(self._embed([query]), k)
        return [
            SearchResult(chunk=self.chunks[idx], score=float(score))
            for score, idx in zip(scores[0], ids[0])
            if idx != -1
        ]

    def save(self, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
        """Persist the FAISS index and chunk metadata to ``index_dir``."""
        if self.index is None:
            raise RuntimeError("Nothing to save; call build() first")
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / INDEX_FILENAME))
        metadata = {
            "model_name": self.model_name,
            "chunks": [asdict(c) for c in self.chunks],
        }
        (index_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d vectors to %s", self.index.ntotal, index_dir)

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR) -> VectorStore:
        """Load a store previously written by :meth:`save`.

        The embedding model recorded at build time is reused so query vectors
        live in the same space as the indexed ones.
        """
        index_path = index_dir / INDEX_FILENAME
        meta_path = index_dir / METADATA_FILENAME
        if not index_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"No saved index in {index_dir}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(model_name=metadata["model_name"])
        store.index = faiss.read_index(str(index_path))
        store.chunks = [Chunk(**c) for c in metadata["chunks"]]
        if store.index.ntotal != len(store.chunks):
            raise ValueError(
                f"Corrupt index in {index_dir}: {store.index.ntotal} vectors "
                f"vs {len(store.chunks)} chunks"
            )
        return store

    @classmethod
    def load_or_build(
        cls,
        docs_dir: Path = DEFAULT_DOCS_DIR,
        index_dir: Path = DEFAULT_INDEX_DIR,
        rebuild: bool = False,
    ) -> VectorStore:
        """Load the saved index if present, otherwise build it from ``docs_dir`` and save."""
        if not rebuild:
            try:
                return cls.load(index_dir)
            except FileNotFoundError:
                logger.info("No saved index in %s; building from %s", index_dir, docs_dir)
        store = cls()
        store.build(docs_dir)
        store.save(index_dir)
        return store


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: build/load the index and print top-k chunks for a query."""
    parser = argparse.ArgumentParser(description="Build a FAISS index over local docs and query it.")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--rebuild", action="store_true", help="Re-ingest docs even if an index exists.")
    parser.add_argument("-q", "--query", default="What is this project about?")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        store = VectorStore.load_or_build(args.docs_dir, args.index_dir, rebuild=args.rebuild)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print(f"\nQuery: {args.query!r}  ({store.index.ntotal} chunks indexed)\n")
    for rank, result in enumerate(store.search(args.query, args.top_k), start=1):
        c = result.chunk
        print(f"[{rank}] score={result.score:.4f}  {c.source}#chunk{c.chunk_id} (char {c.start_char})")
        print(f"    {c.text[:300].replace(chr(10), ' ')}{'...' if len(c.text) > 300 else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
