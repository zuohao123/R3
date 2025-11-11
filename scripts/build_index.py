"""
CLI helper to build a pseudo-text retrieval index from serialized samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.indexer.build_corpus import build_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL samples with pseudo_text fields")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store corpus/index")
    args = parser.parse_args()

    samples = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    build_corpus(samples, args.output)


if __name__ == "__main__":
    main()
