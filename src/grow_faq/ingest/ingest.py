"""Run the full ingestion pipeline: scrape, load, chunk, embed, index.

Every stage stops the run on failure. A failed scrape or load can therefore
never leave a partial collection that looks complete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from grow_faq.config import Settings, load_settings
from grow_faq.ingest import chunk as chunk_stage
from grow_faq.ingest import embed as embed_stage
from grow_faq.ingest import index as index_stage
from grow_faq.ingest import load as load_stage
from grow_faq.ingest import scrape as scrape_stage

Stage = Callable[[Settings], bool]


def _stages(client: Any | None) -> list[tuple[str, Stage]]:
    return [
        ("scrape", scrape_stage.run_scrape),
        ("load", load_stage.run_load),
        ("chunk", chunk_stage.run_chunk),
        ("embed", embed_stage.run_embed),
        ("index", lambda settings: index_stage.run_index(settings, client=client)),
    ]


def run_ingest(settings: Settings, client: Any | None = None) -> bool:
    """Run all five ingestion stages in order, stopping on the first failure."""
    for name, stage in _stages(client):
        print(f"== {name} ==")
        try:
            ok = stage(settings)
        except Exception as exc:
            print(f"[fail] {name}: {exc}", file=sys.stderr)
            return False
        if not ok:
            print(f"pipeline stopped at {name}", file=sys.stderr)
            return False
    print("pipeline complete")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--env", type=Path, default=None, help="path to .env")
    args = parser.parse_args(argv)

    settings = load_settings(config_path=args.config, env_path=args.env)
    return 0 if run_ingest(settings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
