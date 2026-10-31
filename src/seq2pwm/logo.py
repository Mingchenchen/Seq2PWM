"""Render Seq2PWM probability matrices as DNA sequence logos."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASES = ("A", "C", "G", "T")
BASE_COLORS = {
    "A": "green",
    "C": "blue",
    "G": "orange",
    "T": "red",
}


def ppm_to_information_bits(
    ppm: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> pd.DataFrame:
    """Convert a four-column PPM to per-base information content in bits."""

    if isinstance(ppm, pd.DataFrame):
        missing = set(BASES) - set(ppm.columns)
        if missing:
            raise ValueError(f"PPM is missing columns: {sorted(missing)}")
        values = ppm.loc[:, BASES].to_numpy(dtype=np.float64, copy=True)
    else:
        values = np.asarray(ppm, dtype=np.float64)

    if values.ndim != 2 or values.shape[1] != len(BASES):
        raise ValueError("PPM must have shape (positions, 4) in A/C/G/T order")
    if values.shape[0] == 0:
        raise ValueError("PPM must contain at least one position")
    if not np.isfinite(values).all():
        raise ValueError("PPM values must be finite")
    if (values < 0).any():
        raise ValueError("PPM values must be non-negative")

    row_sums = values.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Every PPM row must have a positive sum")
    probabilities = values / row_sums

    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log2(probabilities[positive])
    entropy = -(probabilities * log_probabilities).sum(axis=1)
    information = np.clip(2.0 - entropy, 0.0, 2.0)
    return pd.DataFrame(
        probabilities * information[:, np.newaxis],
        columns=BASES,
    )


def save_motif_logo(
    ppm: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    *,
    png_path: Path,
    pdf_path: Path,
    title: str | None = None,
) -> None:
    """Save publication-quality PNG and vector PDF sequence logos."""

    information_matrix = ppm_to_information_bits(ppm)
    figure_width = max(10.0, len(information_matrix) * 0.4)
    figure, axis = plt.subplots(figsize=(figure_width, 3.8))
    try:
        logomaker.Logo(
            information_matrix,
            ax=axis,
            color_scheme=BASE_COLORS,
            width=0.8,
            vpad=0.05,
            stack_order="big_on_top",
        )
        axis.set_ylim(0.0, 2.0)
        axis.set_ylabel("Information (bits)")
        axis.set_xlabel("Position")
        axis.set_xticks(np.arange(len(information_matrix)))
        axis.set_xticklabels(np.arange(1, len(information_matrix) + 1))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if title:
            axis.set_title(title)
        figure.tight_layout()

        png_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
    finally:
        plt.close(figure)
