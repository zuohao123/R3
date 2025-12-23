#!/usr/bin/env python3
"""
Download PaddleOCR models into a local cache directory.

Usage:
  python scripts/download_paddleocr_models.py --output_dir /path/to/cache --lang en
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PaddleOCR models.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("~/.paddleocr").expanduser(),
        help="Directory to store PaddleOCR models (PADDLEOCR_HOME).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="OCR language (e.g., en, ch).",
    )
    parser.add_argument(
        "--no_angle_cls",
        action="store_true",
        help="Disable angle classification model download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLEOCR_HOME"] = str(output_dir)

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        print("ERROR: paddleocr is not installed.", file=sys.stderr)
        print("Install with: pip install paddleocr", file=sys.stderr)
        return 1

    use_angle_cls = not args.no_angle_cls
    print(f"Downloading PaddleOCR models to: {output_dir}")
    print(f"lang={args.lang} use_angle_cls={use_angle_cls}")

    try:
        PaddleOCR(use_angle_cls=use_angle_cls, lang=args.lang)
    except Exception as exc:
        print(f"ERROR: PaddleOCR download failed: {exc}", file=sys.stderr)
        print("Check network access or pre-download on a machine with Internet.", file=sys.stderr)
        return 2

    pdmodel = list(output_dir.rglob("*.pdmodel"))
    pdiparams = list(output_dir.rglob("*.pdiparams"))
    print(f"Downloaded files: pdmodel={len(pdmodel)} pdiparams={len(pdiparams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
