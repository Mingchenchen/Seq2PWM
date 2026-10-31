import ast
import hashlib
import logging
import random
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


# 20 standard amino acids + X/UNK.
# X is also used as the padding placeholder. Padding is excluded by
# protein_masks=0, so it does not contribute to downstream computation.
AA_VOCAB = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19,
    "X": 20,
}

VALID_REPRESENTATIONS = {"esm", "amino_ids"}


def seq_to_ids(seq: str, vocab: Dict[str, int] = AA_VOCAB) -> torch.Tensor:
    sequence = str(seq).strip().upper()
    ids = [vocab.get(aa, vocab["X"]) for aa in sequence]
    return torch.tensor(ids, dtype=torch.long)


class Seq2PWMDataset(Dataset):
    """Unified dataset for ESM input and the wo_esm ablation."""

    def __init__(
        self,
        csv_path: str,
        cache_path: Union[str, List[str], None] = None,
        mode: str = "train",
        representation: str = "esm",
    ):
        if representation not in VALID_REPRESENTATIONS:
            raise ValueError(
                f"Unknown representation={representation!r}; "
                f"expected one of {sorted(VALID_REPRESENTATIONS)}."
            )

        self.mode = mode
        self.representation = representation
        self.data = pd.read_csv(csv_path)
        self.validate_data()

        self.esm_cache = None
        if representation == "esm":
            if cache_path is None:
                raise ValueError(
                    "cache_path is required when representation='esm'."
                )

            print(f"[{self.mode}] Loading ESM cache...")
            self.esm_cache = {}
            cache_files = [cache_path] if isinstance(cache_path, str) else cache_path
            for path in cache_files:
                single_cache = torch.load(
                    path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
                self.esm_cache.update(single_cache)
            print(
                f"[{self.mode}] ✅ Cache loaded. "
                f"Total unique keys: {len(self.esm_cache)}"
            )
        else:
            print(f"[{self.mode}] Using amino-acid IDs; ESM cache is skipped.")

    def validate_data(self):
        need_cols = {
            "index", "protein1_seq", "protein2_seq",
            "pfm", "pfm_length", "is_dbd",
        }
        missing = need_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _get_emb(self, seq: str) -> torch.Tensor:
        if self.esm_cache is None:
            raise RuntimeError("ESM cache is unavailable in amino_ids mode.")

        sequence = str(seq)
        seq_md5 = hashlib.md5(sequence.encode("utf-8")).hexdigest()
        try:
            embedding = self.esm_cache[seq_md5]
        except KeyError as exc:
            raise KeyError(
                f"Sequence MD5 {seq_md5} is missing from the ESM cache. "
                "Regenerate the cache with python generate_embeddings.py before continuing."
            ) from exc

        return torch.as_tensor(embedding, dtype=torch.float32)

    def _encode_sequence(self, seq: str) -> torch.Tensor:
        if self.representation == "esm":
            return self._get_emb(seq)
        return seq_to_ids(seq)

    @staticmethod
    def _has_second_chain(seq2) -> bool:
        return not pd.isna(seq2) and str(seq2).strip() != ""

    @staticmethod
    def parse_pwm(pwm_str: str) -> torch.Tensor:
        try:
            rows = pwm_str.strip().split(";")
            pwm = []
            for row in rows:
                values = [float(value) for value in row.split(",")]
                if len(values) != 4:
                    raise ValueError(
                        f"Each PWM row must have 4 values, got {len(values)}"
                    )
                pwm.append(values)
            return torch.tensor(pwm, dtype=torch.float32)
        except Exception as exc:
            logging.error("Error parsing PWM: %s", exc)
            raise

    @staticmethod
    def parse_label(label_str):
        if pd.isna(label_str) or label_str == "":
            return None
        try:
            label_list = ast.literal_eval(label_str)
            return torch.tensor(label_list, dtype=torch.float32)
        except Exception as exc:
            logging.error("Error parsing label: %s", exc)
            return None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        idx_col = row["index"]
        seq1 = row["protein1_seq"]
        seq2 = row["protein2_seq"]
        has_chain2 = self._has_second_chain(seq2)

        input1 = self._encode_sequence(seq1)
        asym1 = torch.ones(input1.size(0), dtype=torch.long)

        label1 = self.parse_label(row.get("protein1_label", None))
        label1_valid_mask = (
            torch.ones(input1.size(0), dtype=torch.float32)
            if label1 is not None
            else torch.zeros(input1.size(0), dtype=torch.float32)
        )
        if label1 is None:
            label1 = torch.zeros(input1.size(0), dtype=torch.float32)

        if not has_chain2:
            concat_input = input1
            concat_asym = asym1
            concat_label = label1
            concat_loss_mask = label1_valid_mask
            label2_valid_mask = None
        else:
            input2 = self._encode_sequence(seq2)
            asym2 = torch.full((input2.size(0),), 2, dtype=torch.long)

            label2 = self.parse_label(row.get("protein2_label", None))
            label2_valid_mask = (
                torch.ones(input2.size(0), dtype=torch.float32)
                if label2 is not None
                else torch.zeros(input2.size(0), dtype=torch.float32)
            )
            if label2 is None:
                label2 = torch.zeros(input2.size(0), dtype=torch.float32)

            concat_input = torch.cat([input1, input2], dim=0)
            concat_asym = torch.cat([asym1, asym2], dim=0)
            concat_label = torch.cat([label1, label2], dim=0)
            concat_loss_mask = torch.cat(
                [label1_valid_mask, label2_valid_mask], dim=0
            )

        if self.mode == "train":
            pwm = self.parse_pwm(row["pfm"])
            pwm_len = row["pfm_length"]
        else:
            pwm = ast.literal_eval(row["pfm"])
            pwm_len = ast.literal_eval(row["pfm_length"])

        has_any_label = bool(
            label1_valid_mask.sum().item() > 0
            or (
                has_chain2
                and label2_valid_mask is not None
                and label2_valid_mask.sum().item() > 0
            )
        )

        aug_group_id = (
            int(row["aug_group_id"])
            if "aug_group_id" in self.data.columns
            else -1
        )
        tf_family_id = (
            int(row["tf_family_id"])
            if "tf_family_id" in self.data.columns
            else -1
        )
        is_dbd = int(row["is_dbd"])

        return (
            idx_col,
            concat_input,
            concat_asym,
            pwm,
            pwm_len,
            concat_label,
            concat_loss_mask,
            has_any_label,
            aug_group_id,
            tf_family_id,
            is_dbd,
        )


def collate_fn(
    batch: List[Tuple],
    max_len: int = 1024,
    mode: str = "train",
    representation: str = "esm",
) -> Dict[str, torch.Tensor]:
    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation={representation!r}; "
            f"expected one of {sorted(VALID_REPRESENTATIONS)}."
        )

    (
        indices,
        residue_inputs,
        asym_ids,
        pwms,
        pwm_lengths,
        labels,
        loss_masks,
        has_labels,
        aug_group_ids,
        tf_family_ids,
        is_dbds,
    ) = zip(*batch)

    batch_size = len(residue_inputs)

    if representation == "esm":
        if residue_inputs[0].ndim != 2:
            raise ValueError(
                "ESM input must have shape (L, hidden_dim), got "
                f"{tuple(residue_inputs[0].shape)}."
            )
        embedding_dim = residue_inputs[0].size(-1)
        padded_inputs = torch.zeros(
            (batch_size, max_len, embedding_dim), dtype=torch.float32
        )
        input_key = "protein_embeddings"
    else:
        if residue_inputs[0].ndim != 1:
            raise ValueError(
                "Amino-acid IDs must have shape (L,), got "
                f"{tuple(residue_inputs[0].shape)}."
            )
        padded_inputs = torch.full(
            (batch_size, max_len), AA_VOCAB["X"], dtype=torch.long
        )
        input_key = "amino_ids"

    padded_asyms = torch.zeros((batch_size, max_len), dtype=torch.long)
    masks = torch.zeros((batch_size, max_len), dtype=torch.float32)
    padded_labels = torch.zeros((batch_size, max_len), dtype=torch.float32)
    padded_loss_masks = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i in range(batch_size):
        residue_input = residue_inputs[i]
        asym = asym_ids[i]
        label = labels[i]
        loss_mask = loss_masks[i]

        if residue_input.size(0) > max_len:
            start = random.randint(0, residue_input.size(0) - max_len)
            end = start + max_len
            residue_input = residue_input[start:end]
            asym = asym[start:end]
            label = label[start:end]
            loss_mask = loss_mask[start:end]

        length = residue_input.size(0)

        if representation == "esm":
            padded_inputs[i, :length, :] = residue_input
        else:
            padded_inputs[i, :length] = residue_input

        padded_asyms[i, :length] = asym
        masks[i, :length] = 1.0
        padded_labels[i, :length] = label
        padded_loss_masks[i, :length] = loss_mask

    result = {
        "indices": indices,
        input_key: padded_inputs,
        "asym_ids": padded_asyms,
        "protein_masks": masks,
        "protein_labels": padded_labels,
        "loss_masks": padded_loss_masks,
        "has_label": torch.tensor(has_labels, dtype=torch.bool),
        "aug_group_id": torch.tensor(aug_group_ids, dtype=torch.long),
        "tf_family_id": torch.tensor(tf_family_ids, dtype=torch.long),
        "is_dbd": torch.tensor(is_dbds, dtype=torch.float32),
    }

    if mode == "train":
        result["pwms"] = torch.stack(pwms)
        result["pwm_lengths"] = pwm_lengths
    else:
        result["pwm_lists"] = list(pwms)
        result["pwm_len_lists"] = list(pwm_lengths)

    return result
