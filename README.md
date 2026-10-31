# Seq2PWM

Seq2PWM is a deep-learning model for predicting DNA-binding motifs directly
from transcription-factor protein sequences. It converts ESM-2 residue
embeddings into a fixed-length 35 × 4 probability matrix over A/C/G/T and also
reports residue-importance scores and a DNA-binding-domain probability.

This repository contains the inference, training, ablation, heterodimer,
monomer-whole-data, and ten-fold cross-validation code accompanying the manuscript
**“Prediction of DNA-binding Specificity from Transcription Factor
Sequences.”**

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Weights](#model-weights)
- [Usage](#usage)
- [Input Format](#input-format)
- [Output Files](#output-files)
- [Data and Benchmarks](#data-and-benchmarks)
- [Training and Configuration](#training-and-configuration)
- [Dependencies](#dependencies)
- [Tests](#tests)
- [Citation](#citation)
- [License](#license)

## Features

- **Sequence-to-motif prediction:** predicts a 35-position DNA probability
  matrix from one TF sequence or a pair of heterodimer sequences.
- **ESM-2 representations:** uses residue embeddings from
  `esm2_t33_650M_UR50D`.
- **Interpretable outputs:** reports residue-importance values and a
  DNA-binding-domain probability in addition to the motif.
- **Reproducible evaluation:** includes monomer and heterodimer test tables,
  experiment configurations, weight loading, and automated tests.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mingchenchen/Seq2PWM.git
cd Seq2PWM
```

### 2. Create the Conda environment

```bash
conda create -n seq2pwm python=3.10.14 pip=24.0 -y
conda activate seq2pwm
```

### 3. Install PyTorch and Seq2PWM dependencies

The reference setup uses PyTorch 2.3.1 with CUDA 12.1 on 64-bit Linux:

```bash
python -m pip install torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For training, evaluation, and plotting, also install:

```bash
python -m pip install -r requirements-full.txt
```

Confirm the environment:

```bash
python -c "import torch, esm; print(torch.__version__, torch.cuda.is_available())"
python predict.py --help
```

## Project Structure

```text
Seq2PWM/
├── configs/
│   ├── ablation/
│   ├── experiments/
│   └── ten_fold.yaml
├── data/
│   └── test/
├── examples/
│   └── proteins.csv
├── src/seq2pwm/
│   ├── cli/
│   ├── dataset.py
│   ├── esm.py
│   ├── model.py
│   ├── model_ablation.py
│   └── weights.py
├── weights/
├── predict.py
├── generate_embeddings.py
├── train.py
├── train_ablation.py
├── infer.py
└── requirements.txt
```

## Model Weights

Download the weights from the GitHub Release and place them in `weights/`.

| Release asset | Intended use |
| --- | --- |
| `seq2pwm_monomer_all.pt` | Monomer model trained on all available monomer training data |
| `seq2pwm_monomer_holdout70.pt` | Monomer model for the 70%-identity cluster hold-out setting |
| `seq2pwm_heterodimer_pair_holdout70.pt` | Heterodimer model for the low-homology pair-level hold-out setting |

## Usage

### Basic monomer inference

```bash
python predict.py \
  --weights weights/seq2pwm_monomer_all.pt \
  --input examples/test_data.csv \
  --output-dir predictions/test_data \
  --save-esm-cache predictions/test_data/esm_cache.pt
```

Seq2PWM automatically saves the predicted probability matrix, DNA sequence logo, residue-importance values, and run metadata. Add `--no-logo` when image output is not needed.

### Heterodimer inference

Prepare a CSV containing both `protein1_seq` and `protein2_seq`, then run:

```bash
python predict.py \
  --weights weights/seq2pwm_heterodimer_pair_holdout70.pt \
  --input data/test/heterodimer_test.csv \
  --output-dir predictions/heterodimer_test
```

### ESM-2 model and offline caches

On first use, Seq2PWM downloads the official ESM-2 weight file to:

```text
.cache/esm/esm2_t33_650M_UR50D.pt
```

To use a precomputed sequence-embedding cache without downloading or
recomputing embeddings:

```bash
python predict.py \
  --weights weights/seq2pwm_monomer_all.pt \
  --input my_proteins.csv \
  --esm-cache-path data/cache/my_proteins.pt \
  --cache-only \
  --output-dir predictions
```

For a sequence longer than 1,024 residues, provide its DNA-binding-domain sequence. The direct-prediction command fails on oversized input rather than applying a random crop.

## Input Format

Input may be an unlabeled CSV with unique identifiers:

```csv
index,protein1_seq
my_tf,MKRKRTAETRSQLVEQLKQQNANAAEQLKQQG
```

For a heterodimer, add the second sequence:

```csv
index,protein1_seq,protein2_seq
my_pair,MKRKRTAETRSQLVEQLKQQNANAAEQLKQQG,MPPQRRTAFTAEQLKQQ
```

Monomer input may also be FASTA:

```fasta
>my_tf
MKRKRTAETRSQLVEQLKQQNANAAEQLKQQG
```

## Output Files

```text
predictions/
├── predictions.csv
├── run_metadata.json
├── esm_cache.pt
└── <sample_id>/
    ├── predicted_ppm.csv
    ├── predicted_logo.png
    ├── predicted_logo.pdf
    └── residue_importance.csv
```

## Data and Benchmarks

The repository contains two labeled tables for installation checks and direct inference:

| File | Records | Sequence columns |
| --- | ---: | --- |
| `data/test/monomer_test.csv` | 90 | `protein1_seq` |
| `data/test/heterodimer_test.csv` | 161 | `protein1_seq`, `protein2_seq` |

The complete processed training data, exact train/validation/test splits, ten-fold weights, and per-example predictions are distributed through the versioned Zenodo reproducibility archive:

<https://doi.org/10.5281/zenodo.23534817>

Full training datasets and large derived artifacts are kept out of Git history. See [`data/README.md`](data/README.md) for the expected archive documentation and data-path mapping.

## Training and Configuration

Configuration files are under `configs/`. Every released experiment uses an explicit seed (`42`) and a configuration-specific data path.

| Configuration | Purpose |
| --- | --- |
| `configs/ten_fold.yaml` | Ten-fold cross-validation |
| `configs/experiments/monomer_whole_data.yaml` | Whole-data monomer training |
| `configs/experiments/heterodimer.yaml` | Heterodimer training |
| `configs/ablation/base.yaml` | Architecture ablations |

Training uses PyTorch DistributedDataParallel with NCCL. Examples:

```bash
# Ten-fold cross-validation; use folds 0 through 9
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/ten_fold.yaml --fold 0

# Whole-data monomer model
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/experiments/monomer_whole_data.yaml

# Heterodimer model
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/experiments/heterodimer.yaml

# Ablation model
torchrun --standalone --nproc_per_node=2 train_ablation.py \
  --config configs/ablation/base.yaml
```

Available ablation modes are `none`, `wo_cnn`, `wo_crossattn`, and `wo_esm`.

## Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| Python | 3.10–3.11 | Runtime |
| PyTorch | 2.3.1 | Neural-network inference and training |
| fair-esm | 2.0.0 | ESM-2 protein embeddings |
| NumPy | 1.26.4 | Numerical operations |
| pandas | 2.2.2 | Input and output tables |
| matplotlib | 3.9.1 | Plotting |
| Logomaker | 0.8 | DNA sequence logos |

Exact direct-inference dependencies are listed in `requirements.txt`.
Additional experiment dependencies are listed in `requirements-full.txt`.

## Citation

Until the article receives a DOI, cite the software release and manuscript:

> Junchen Luo, Yunzhi Lang, Zhi Wang, Xiaoliang Sunney Xie, and Mingchen Chen.
> *Prediction of DNA-binding Specificity from Transcription Factor Sequences.*
> 2026.


## License

Seq2PWM is released under the [MIT License](LICENSE.txt). External datasets and
ESM model parameters remain subject to their original licenses and usage
conditions.

## Acknowledgments

- [ESM](https://github.com/facebookresearch/esm) for protein language-model
  representations.

