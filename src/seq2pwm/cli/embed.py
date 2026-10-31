"""Build the MD5-indexed ESM-2 cache consumed by Seq2PWM."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import torch

from seq2pwm.embedding_cache import ensure_esm_cache


def get_md5(sequence: str) -> str:
    """Return the cache key used by the original training pipeline."""
    return hashlib.md5(str(sequence).encode("utf-8")).hexdigest()


def extract_cache_with_md5(
    csv_paths: Iterable[Path],
    output_path: Path,
    *,
    model_cache: Path,
    device: str,
) -> None:
    ensure_esm_cache(
        csv_paths,
        output_path,
        model_cache=model_cache,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Seq2PWM ESM cache from one or more input CSV files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        nargs="+",
        help="Input CSV file(s) containing protein1_seq and optionally protein2_seq.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(".cache/esm"),
        help="Directory used by torch.hub for the ESM-2 model files.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_cache_with_md5(
        args.input,
        args.output,
        model_cache=args.model_cache,
        device=args.device,
    )


if __name__ == "__main__":
    main()
