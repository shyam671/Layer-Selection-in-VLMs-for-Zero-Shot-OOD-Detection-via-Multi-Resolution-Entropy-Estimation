# Layer Selection in VLMs for Zero-Shot OOD Detection via Multi-Resolution Entropy Estimation

**Shyam Nandan Rai, Francesco Di Salvo, Sebastian Doerrich, Christian Ledig**
xAILab, University of Bamberg

This repository contains the code for **zero-shot out-of-distribution (OOD) detection** on
medical images using vision–language models (VLMs). Instead of relying only on the final
encoder layer, we score OOD at **every transformer layer** and select the most informative
**subset of layers** — using *in-distribution data only* — by minimizing a **multi-resolution
histogram entropy**. The final encoder layer is not consistently the best; intermediate layers
often are, and the optimal depth varies by imaging modality.

## Links

- **Project page:** https://shyam671.github.io/layer-selection-ood/
- **Paper:** see the project page.

## Method overview

The layer-selection method (referred to as **MOD** in the code) works in three steps:

1. **Per-layer scoring.** Forward hooks capture the output of every transformer block. Each
   layer's features are projected into the shared image–text space and matched against the
   in-distribution text prompts to produce per-layer logits. The base OOD score is
   **Maximum Concept Matching (MCM)** — the negative max softmax confidence.
2. **Layer-subset selection (ID only).** Candidate layer subsets are enumerated (each always
   includes the final/reference layer). For every subset, the MCM scores are averaged across
   its layers and the **entropy of that score distribution** is computed as an ensemble over
   histogram bin sizes `{4, 8, 16, 32, 64}`. This multi-resolution estimate is stable where a
   single-resolution entropy is not. The subset with the **lowest entropy** on ID data wins.
3. **Evaluation.** The selected layers are applied to ID and OOD images and evaluated with
   AUROC, AUPR, and FPR95.

The selection logic lives in [`layer_selector.py`](layer_selector.py)
(`select_best_layer_combination`); the per-layer feature extraction lives in
[`utils.py`](utils.py) (`register_hooks`, `process_intermediate_features`).

## Installation

The code targets the existing `qwen-vl-finetune` conda environment.

```bash
conda activate qwen-vl-finetune
pip install -r requirements.txt
```

Two CLIP backends are supported:

- **BiomedCLIP** (`--models_name biomed`) — pulled from Hugging Face via the pip
  `open_clip_torch` package.
- **UniMed-CLIP** (`--models_name ViT-B-16-quickgelu`) — loaded through the **vendored**
  [`src/open_clip`](src/open_clip) package from a local checkpoint (see below).

> Note: [`src/training`](src/training) is the upstream CLIP training harness and is **not used**
> by the OOD inference pipeline.

## Checkpoints & data roots

- Place the UniMed-CLIP checkpoint `unimed_clip_vit_b16.pt` under `--models_ckpt_dir`
  (default `/data/local/MedOOD/models/`).
- Datasets are read from `--dataset_root_dir` (default `/data/local/MedOOD/`) following the
  [OpenMIBOOD](https://github.com/remic-othr/OpenMIBOOD) layout.

## Datasets

| Group | In-distribution | Out-of-distribution (`--dataset_dir`) |
|-------|-----------------|----------------------------------------|
| Chest X-ray | normal / pneumonia | `xray` (COVID) |
| MIDOG histopathology | `1a` | `midog-csid`, `midog-near`, `midog-far-fnac`, `midog-far-ccagt` |
| OASIS3 brain | `train` | `oasis-near-atlas`, `oasis-near-brats`, `oasis-near-ct`, `oasis-far-heart`, `oasis-far-chaos` |

OASIS3 volumes are preprocessed offline into `.pt` tensors — see
[`oasis-dataprep/`](oasis-dataprep) (`preprocess.py`). This step needs `nibabel`, `SimpleITK`,
and [HD-BET](https://github.com/MIC-DKFZ/HD-BET) for skull stripping.

## Usage

Run a single evaluation with `main.py`; results are **appended** to `results.txt`.

```bash
# Layer-selection (MOD) with BiomedCLIP, selecting up to 5 layers
python main.py --text_prompt multi-mod --score MOD --dataset_dir midog-near \
  --number_of_mod_layers 5 --models_name biomed

# MCM baseline (last layer only)
python main.py --text_prompt single --score MCM --dataset_dir midog-near
```

See [`run.sh`](run.sh) for a ready-to-run driver (including the layers `1..11` ablation).

### Key arguments

| Flag | Description |
|------|-------------|
| `--dataset_dir` | OOD benchmark to evaluate (see table above) |
| `--dataset_root_dir` | Root directory of the datasets |
| `--models_name` | `biomed` (BiomedCLIP) or `ViT-B-16-quickgelu` (UniMed-CLIP) |
| `--score` | `MOD` (layer selection), or baselines `MCM` / `energy` / `entropy` / `var` |
| `--text_prompt` | `single` / `multi` / `multi-mod` / `single-neg` prompt set |
| `--number_of_mod_layers` | Max size of the selected layer subset (MOD) |
| `--manual_input_best_layers` / `--input_best_layers` | Pin a single layer index instead of searching |

## Results

MOD ablation (BiomedCLIP, `multi-mod`) as the selected layer subset grows — AUROC improves and
FPR95 drops well below the final-layer-only baseline `[11]`. Columns: **AUROC / AUPR / FPR95**
(from `results.txt`).

| `best_layers` | midog-far-fnac | midog-far-ccagt |
|---------------|----------------|------------------|
| `[11]` (last layer only) | 83.7 / 88.8 / 70.4 | 94.5 / 99.7 / 24.1 |
| `[6, 11]` | 87.7 / 92.0 / 66.1 | 97.3 / 99.8 / 10.5 |
| `[4, 6, 8, 11]` | 93.7 / 95.4 / 36.0 | 98.8 / 99.9 / 3.2 |
| `[3, 4, 6, 8, 11]` | **95.4 / 96.3 / 25.4** | **99.2 / 99.9 / 1.5** |

## Repository structure

```
main.py             CLI entry point: load model, build ID/OOD lists, score, write results.txt
utils.py            Model loading, datasets/transforms, feature hooks, scoring pipelines
layer_selector.py   MOD — multi-resolution-entropy layer-subset selection (paper core)
get_in_class.py     In-distribution text prompts per modality / prompt type
run.sh              Example driver / ablation loop
requirements.txt    Python dependencies
src/open_clip/      Vendored UniMed-CLIP (open_clip) — used by the quickgelu backend
src/training/       Upstream CLIP training harness (unused by inference)
oasis-dataprep/     Offline OASIS3 preprocessing (preprocess.py + util.py)
Plots/              Figure-generation scripts and rendered figures
```

## Citation

```bibtex
@inproceedings{rai2026layerselection,
  title     = {Layer Selection in VLMs for Zero-Shot OOD Detection via Multi-Resolution Entropy Estimation},
  author    = {Rai, Shyam Nandan and Di Salvo, Francesco and Doerrich, Sebastian and Ledig, Christian},
  year      = {2026}
}
```

## Acknowledgements

Built on [OpenCLIP](https://github.com/mlfoundations/open_clip),
[UniMed-CLIP](https://github.com/mbzuai-oryx/UniMed-CLIP),
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224),
and the [OpenMIBOOD](https://github.com/remic-othr/OpenMIBOOD) benchmark.
