"""Run the Light-RAG API server with Uvicorn.

Usage::

    python app.py                       # http://127.0.0.1:8000, docs at /docs
    python app.py --host 0.0.0.0 --port 8080

Host and port can also be set with the ``HOST`` and ``PORT`` environment
variables (e.g. in Docker). Model and index settings are read from the
``LIGHTRAG_*`` variables documented in ``src/api/server.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

# Make `src` importable regardless of the directory the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: list[str] | None = None) -> None:
    """Parse CLI options and start a single-worker Uvicorn server."""
    parser = argparse.ArgumentParser(description="Serve the Light-RAG /query API.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    from src.api.server import app

    # One worker only: each worker would load its own copy of the model into RAM.
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)


if __name__ == "__main__":
    main()
