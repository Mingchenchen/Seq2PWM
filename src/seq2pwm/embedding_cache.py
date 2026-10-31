"""Build and validate MD5-indexed ESM caches for configured datasets."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from .esm import ESMAsSchNetEmbedding, ressymb_to_resindex


SEQUENCE_COLUMNS = ("protein1_seq", "protein2_seq")
DEFAULT_CACHE_INPUT_KEYS = (
    "train_tf_data",
    "valid_data",
    "test_data"
)


def sequence_md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cache_input_paths(data_cfg: Mapping[str, Any]) -> list[Path]:
    """Resolve the configured CSVs whose sequences must be cached."""

    configured_keys = data_cfg.get("esm_cache_inputs")
    if configured_keys is None:
        configured_keys = [
            key for key in DEFAULT_CACHE_INPUT_KEYS if data_cfg.get(key)
        ]
    if not isinstance(configured_keys, (list, tuple)) or not configured_keys:
        raise ValueError("data.esm_cache_inputs must be a non-empty list")

    paths: list[Path] = []
    seen: set[Path] = set()
    for key in configured_keys:
        if not isinstance(key, str) or key not in data_cfg:
            raise ValueError(
                f"Unknown data.esm_cache_inputs entry {key!r}; each entry must "
                "name a path field in the data configuration"
            )
        value = data_cfg[key]
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            path = Path(item)
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def collect_sequences(csv_paths: Iterable[Path]) -> dict[str, str]:
    """Return unique sequences keyed exactly as ``Seq2PWMDataset`` expects."""

    sequences: dict[str, str] = {}
    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Configured ESM-cache input is missing: {path}")
        frame = pd.read_csv(path)
        available = [column for column in SEQUENCE_COLUMNS if column in frame]
        if "protein1_seq" not in available:
            raise ValueError(
                f"{path} must contain protein1_seq to build the ESM cache"
            )
        for column in available:
            for value in frame[column].dropna():
                sequence = str(value)
                if not sequence.strip():
                    continue
                sequences.setdefault(sequence_md5(sequence), sequence)
    if not sequences:
        raise ValueError("No protein sequences were found in the configured inputs")
    return sequences


def _valid_embedding(value: Any, sequence_length: int) -> bool:
    return (
        torch.is_tensor(value)
        and value.ndim == 2
        and value.size(0) == sequence_length
        and value.size(1) == 1280
    )


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest(
    cache_path: Path,
    csv_paths: list[Path],
    expected_count: int,
    cache_entry_count: int,
    generated_count: int,
) -> None:
    manifest = {
        "format": "Seq2PWM MD5-indexed ESM-2 residue embeddings",
        "esm_model": "esm2_t33_650M_UR50D",
        "embedding_dimension": 1280,
        "cache_file": cache_path.name,
        "expected_sequence_count": expected_count,
        "cache_entry_count": cache_entry_count,
        "generated_or_repaired_count": generated_count,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in csv_paths
        ],
    }
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}"
    )
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_esm_cache(
    csv_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    model_cache: str | Path,
    device: str | torch.device,
) -> Path:
    """Create or repair one cache covering every configured input sequence."""

    input_paths = [Path(path) for path in csv_paths]
    cache_path = Path(output_path)
    sequences = collect_sequences(input_paths)

    cache: dict[str, torch.Tensor] = {}
    if cache_path.is_file():
        loaded = torch.load(
            cache_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if not isinstance(loaded, Mapping):
            raise TypeError(f"ESM cache must contain a mapping: {cache_path}")
        cache.update(loaded)

    missing = [
        (key, sequence)
        for key, sequence in sequences.items()
        if not _valid_embedding(cache.get(key), len(sequence))
    ]

    if missing:
        print(
            f"Building ESM cache at {cache_path}: "
            f"{len(missing)} of {len(sequences)} sequences require embeddings"
        )
        model = ESMAsSchNetEmbedding(str(model_cache)).to(device).eval()
        with torch.inference_mode():
            for key, sequence in tqdm(missing, desc="ESM-2 embeddings"):
                sequence_index = torch.tensor(
                    [[
                        ressymb_to_resindex.get(amino_acid, 0)
                        for amino_acid in sequence
                    ]],
                    dtype=torch.long,
                    device=device,
                )
                embedding = model(sequence_index).squeeze(0).detach().cpu()
                if not _valid_embedding(embedding, len(sequence)):
                    raise ValueError(
                        f"Unexpected ESM embedding shape for sequence MD5 {key}: "
                        f"{tuple(embedding.shape)}"
                    )
                cache[key] = embedding
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _atomic_torch_save(cache, cache_path)
    else:
        print(
            f"Using complete ESM cache at {cache_path} "
            f"({len(sequences)} configured sequences)"
        )

    _write_manifest(
        cache_path,
        input_paths,
        expected_count=len(sequences),
        cache_entry_count=len(cache),
        generated_count=len(missing),
    )
    return cache_path
