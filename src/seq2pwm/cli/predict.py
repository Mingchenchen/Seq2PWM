"""User-facing Seq2PWM prediction from an unlabeled CSV or FASTA file."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from seq2pwm.weights import (
    infer_model_family,
    load_weights_strict,
    read_weights,
)
from seq2pwm.esm import ESMAsSchNetEmbedding, ressymb_to_resindex
from seq2pwm.logo import save_motif_logo
from seq2pwm.model import PWMGenerator
from seq2pwm.model_ablation import PWMGenerator as AblationPWMGenerator


@dataclass(frozen=True)
class Sample:
    sample_id: str
    protein1_seq: str
    protein2_seq: str | None = None


ESM_MAX_SEQUENCE_LENGTH = 1024


@dataclass(frozen=True)
class ChainWindow:
    chain: int
    sequence: str
    start: int
    end: int
    concatenated_offset: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CropPlan:
    total_length: int
    start: int
    end: int
    windows: tuple[ChainWindow, ...]

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def applied(self) -> bool:
        return self.length < self.total_length


@dataclass(frozen=True)
class PredictionResult:
    ppm: torch.Tensor
    residue_importance: torch.Tensor
    dbd_probability: float
    crop: CropPlan


def plan_random_crop(
    sample: Sample,
    max_length: int,
    rng: random.Random,
) -> CropPlan:
    """Choose the same contiguous random crop used by the training collator."""

    if max_length <= 0:
        raise ValueError("--max-length must be a positive integer")

    chains = [(1, sample.protein1_seq)]
    if sample.protein2_seq is not None:
        chains.append((2, sample.protein2_seq))

    total_length = sum(len(sequence) for _, sequence in chains)
    if total_length > max_length:
        start = rng.randint(0, total_length - max_length)
        end = start + max_length
    else:
        start, end = 0, total_length

    windows = []
    concatenated_offset = 0
    for chain, sequence in chains:
        chain_end = concatenated_offset + len(sequence)
        overlap_start = max(start, concatenated_offset)
        overlap_end = min(end, chain_end)
        if overlap_start < overlap_end:
            windows.append(
                ChainWindow(
                    chain=chain,
                    sequence=sequence,
                    start=overlap_start - concatenated_offset,
                    end=overlap_end - concatenated_offset,
                    concatenated_offset=concatenated_offset,
                )
            )
        concatenated_offset = chain_end

    return CropPlan(
        total_length=total_length,
        start=start,
        end=end,
        windows=tuple(windows),
    )


def normalize_sequence(value: object, *, field: str, sample_id: str) -> str:
    sequence = re.sub(r"\s+", "", str(value)).upper()
    if not sequence:
        raise ValueError(f"{sample_id}: {field} is empty")
    if not re.fullmatch(r"[A-Z]+", sequence):
        raise ValueError(
            f"{sample_id}: {field} must contain amino-acid letters only"
        )
    return sequence


def read_csv(path: Path) -> list[Sample]:
    frame = pd.read_csv(path)
    id_column = "index" if "index" in frame.columns else "id"
    required = {id_column, "protein1_seq"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    samples = []
    for _, row in frame.iterrows():
        sample_id = str(row[id_column]).strip()
        if not sample_id:
            raise ValueError("Every input row must have a non-empty id/index")
        sequence1 = normalize_sequence(
            row["protein1_seq"], field="protein1_seq", sample_id=sample_id
        )
        sequence2 = None
        if "protein2_seq" in frame.columns and not pd.isna(row["protein2_seq"]):
            raw_sequence2 = str(row["protein2_seq"]).strip()
            if raw_sequence2:
                sequence2 = normalize_sequence(
                    raw_sequence2,
                    field="protein2_seq",
                    sample_id=sample_id,
                )
        samples.append(Sample(sample_id, sequence1, sequence2))
    return samples


def read_fasta(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    header: str | None = None
    sequence_lines: list[str] = []

    def append_record() -> None:
        if header is None:
            return
        sample_id = header.split()[0]
        sequence = normalize_sequence(
            "".join(sequence_lines), field="sequence", sample_id=sample_id
        )
        samples.append(Sample(sample_id, sequence))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            append_record()
            header = line[1:].strip()
            if not header:
                raise ValueError("FASTA header cannot be empty")
            sequence_lines = []
        else:
            if header is None:
                raise ValueError("FASTA sequence appeared before the first header")
            sequence_lines.append(line)
    append_record()
    return samples


def read_samples(path: Path) -> list[Sample]:
    if path.suffix.lower() in {".fa", ".faa", ".fasta", ".fas"}:
        samples = read_fasta(path)
    else:
        samples = read_csv(path)
    if not samples:
        raise ValueError(f"No sequences were found in {path}")
    identifiers = [sample.sample_id for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Input sample identifiers must be unique")
    return samples


def sequence_md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EmbeddingProvider:
    def __init__(
        self,
        *,
        device: torch.device,
        esm_model_cache: Path,
        cache_path: Path | None,
        cache_only: bool,
    ) -> None:
        self.device = device
        self.esm_model_cache = esm_model_cache
        self.cache_only = cache_only
        self.cache: dict[str, torch.Tensor] = {}
        self.model: ESMAsSchNetEmbedding | None = None
        self.new_embeddings = 0
        if cache_path is not None:
            if not cache_path.is_file():
                raise FileNotFoundError(f"ESM cache not found: {cache_path}")
            loaded = torch.load(
                cache_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            if not isinstance(loaded, dict):
                raise TypeError("ESM cache must be a dictionary keyed by sequence MD5")
            self.cache.update(loaded)

    def _load_model(self) -> ESMAsSchNetEmbedding:
        if self.model is None:
            self.model = ESMAsSchNetEmbedding(str(self.esm_model_cache))
            self.model = self.model.to(self.device).eval()
        return self.model

    def get(self, sequence: str) -> torch.Tensor:
        key = sequence_md5(sequence)
        if key in self.cache:
            return torch.as_tensor(self.cache[key], dtype=torch.float32)
        if self.cache_only:
            raise KeyError(
                f"Sequence MD5 {key} is absent from the supplied ESM cache"
            )

        model = self._load_model()
        indices = torch.tensor(
            [[ressymb_to_resindex.get(amino_acid, 0) for amino_acid in sequence]],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            embedding = model(indices).squeeze(0).detach().cpu()
        self.cache[key] = embedding
        self.new_embeddings += 1
        return embedding

    def get_window(self, sequence: str, start: int, end: int) -> torch.Tensor:
        """Return a residue window, preferring full-sequence cached context."""

        if not 0 <= start < end <= len(sequence):
            raise ValueError(
                f"Invalid sequence window [{start}, {end}) for length {len(sequence)}"
            )

        full_key = sequence_md5(sequence)
        if full_key in self.cache:
            full_embedding = torch.as_tensor(
                self.cache[full_key], dtype=torch.float32
            )
            if full_embedding.size(0) != len(sequence):
                raise ValueError(
                    "Cached ESM embedding length does not match its sequence: "
                    f"{full_embedding.size(0)} != {len(sequence)}"
                )
            return full_embedding[start:end]

        if start == 0 and end == len(sequence):
            return self.get(sequence)

        cropped_sequence = sequence[start:end]
        cropped_key = sequence_md5(cropped_sequence)
        if cropped_key in self.cache:
            return torch.as_tensor(self.cache[cropped_key], dtype=torch.float32)

        # When a complete chain fits ESM-2, embed it before slicing so the
        # residue representation retains the same within-chain context used by
        # the original precomputed-cache pipeline. Longer individual chains
        # must be cropped before ESM-2 embedding.
        if len(sequence) <= ESM_MAX_SEQUENCE_LENGTH:
            return self.get(sequence)[start:end]
        return self.get(cropped_sequence)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cache, path)


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "sample"


def serialize_ppm(ppm: torch.Tensor) -> str:
    return ";".join(
        ",".join(f"{float(value):.10g}" for value in row)
        for row in ppm
    )


def predict_sample(
    model: torch.nn.Module,
    sample: Sample,
    embeddings: EmbeddingProvider,
    device: torch.device,
    max_length: int,
    rng: random.Random,
) -> PredictionResult:
    crop = plan_random_crop(sample, max_length, rng)
    inputs = []
    asym_ids = []
    for window in crop.windows:
        embedding = embeddings.get_window(
            window.sequence,
            window.start,
            window.end,
        )
        inputs.append(embedding)
        asym_ids.append(
            torch.full((window.length,), window.chain, dtype=torch.long)
        )

    protein_embedding = torch.cat(inputs, dim=0)
    asym_id = torch.cat(asym_ids, dim=0)
    length = protein_embedding.size(0)
    if length != crop.length:
        raise ValueError(
            f"{sample.sample_id}: ESM embedding length {length} does not match "
            f"the planned crop length {crop.length}"
        )

    protein_embedding = protein_embedding.unsqueeze(0).to(device)
    asym_id = asym_id.unsqueeze(0).to(device)
    protein_mask = torch.ones((1, length), dtype=torch.float32, device=device)

    with torch.inference_mode():
        ppm, residue_importance, auxiliary = model(
            protein_embedding, asym_id, protein_mask
        )
        dbd_probability = torch.sigmoid(auxiliary["dbd_logits"])[0, 0].item()

    return PredictionResult(
        ppm=ppm[0].detach().cpu(),
        residue_importance=residue_importance[0, :length].detach().cpu(),
        dbd_probability=float(dbd_probability),
        crop=crop,
    )


def build_model_for_weights(
    weights: Mapping[str, Any],
) -> tuple[torch.nn.Module, str]:
    """Construct the exact full-model variant identified by parameter keys."""

    model_family = infer_model_family(weights)
    if model_family == "main":
        return PWMGenerator(), model_family
    return AblationPWMGenerator(ablation="none"), model_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict a 35-position probability-normalized motif matrix from "
            "unlabeled protein sequences."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--weights",
        type=Path,
        help="Path to public raw model weights (a model.state_dict() file).",
    )
    model_group.add_argument(
        "--checkpoint",
        dest="weights",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV with id/index + protein1_seq (+ optional protein2_seq), or FASTA.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--esm-cache-path", type=Path)
    parser.add_argument(
        "--esm-model-cache",
        type=Path,
        default=Path(".cache/esm"),
        help="Directory containing/downloading the ESM-2 model weights.",
    )
    parser.add_argument(
        "--save-esm-cache",
        type=Path,
        help="Optionally save existing and newly computed MD5 embeddings.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Fail instead of computing a sequence absent from --esm-cache-path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum concatenated model input length (default: 1024).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible random cropping of long inputs (default: 42).",
    )
    parser.add_argument(
        "--no-logo",
        action="store_true",
        help="Do not create PNG and PDF sequence logos for each prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_only and args.esm_cache_path is None:
        raise ValueError("--cache-only requires --esm-cache-path")
    if not 1 <= args.max_length <= ESM_MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"--max-length must be between 1 and {ESM_MAX_SEQUENCE_LENGTH}"
        )
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")

    device = torch.device(args.device)
    samples = read_samples(args.input)
    weights = read_weights(args.weights, map_location=device)
    model, model_family = build_model_for_weights(weights)
    model = model.to(device)
    load_weights_strict(model, weights, map_location=device)
    model.eval()

    embedding_provider = EmbeddingProvider(
        device=device,
        esm_model_cache=args.esm_model_cache,
        cache_path=args.esm_cache_path,
        cache_only=args.cache_only,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_rng = random.Random(args.seed)

    summary_rows = []
    for sample in samples:
        result = predict_sample(
            model,
            sample,
            embedding_provider,
            device,
            args.max_length,
            crop_rng,
        )
        sample_dir = args.output_dir / safe_name(sample.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        ppm_frame = pd.DataFrame(
            result.ppm.numpy(), columns=["A", "C", "G", "T"]
        )
        ppm_frame.assign(
            position=range(1, result.ppm.size(0) + 1)
        )[["position", "A", "C", "G", "T"]].to_csv(
            sample_dir / "predicted_ppm.csv", index=False
        )
        if not args.no_logo:
            save_motif_logo(
                ppm_frame,
                png_path=sample_dir / "predicted_logo.png",
                pdf_path=sample_dir / "predicted_logo.pdf",
                title=f"Predicted motif: {sample.sample_id}",
            )

        residue_rows = []
        score_index = 0
        for window in result.crop.windows:
            for chain_offset in range(window.start, window.end):
                residue_rows.append(
                    {
                        "concatenated_position": (
                            window.concatenated_offset + chain_offset + 1
                        ),
                        "chain": window.chain,
                        "chain_position": chain_offset + 1,
                        "residue_importance": float(
                            result.residue_importance[score_index]
                        ),
                    }
                )
                score_index += 1
        pd.DataFrame(residue_rows).to_csv(
            sample_dir / "residue_importance.csv", index=False
        )

        summary_rows.append(
            {
                "index": sample.sample_id,
                "protein1_length": len(sample.protein1_seq),
                "protein2_length": len(sample.protein2_seq or ""),
                "model_input_length": result.crop.length,
                "random_crop_applied": result.crop.applied,
                "crop_start": result.crop.start + 1,
                "crop_end": result.crop.end,
                "dbd_probability": result.dbd_probability,
                "predicted_ppm": serialize_ppm(result.ppm),
            }
        )
        crop_message = (
            f" using residues {result.crop.start + 1}-{result.crop.end} "
            f"of {result.crop.total_length}"
            if result.crop.applied
            else ""
        )
        print(f"Predicted {sample.sample_id}{crop_message}")

    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "predictions.csv", index=False
    )
    if args.save_esm_cache is not None:
        embedding_provider.save(args.save_esm_cache)

    metadata = {
        "model": "Seq2PWM",
        "weights_file": args.weights.name,
        "weights_model_family": model_family,
        "weights_sha256": sha256(args.weights),
        "input_sha256": sha256(args.input),
        "torch_version": torch.__version__,
        "device": str(device),
        "sample_count": len(samples),
        "new_esm_embeddings": embedding_provider.new_embeddings,
        "max_length": args.max_length,
        "long_sequence_policy": "uniform random contiguous crop",
        "random_seed": args.seed,
        "output_matrix": "35-position probability-normalized motif matrix (PPM)",
        "motif_logo": not args.no_logo,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
