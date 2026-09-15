# Layer Selection in VLMs for Zero-Shot OOD Detection via Multi-Resolution Entropy Estimation

**Shyam Nandan Rai, Francesco Di Salvo, Sebastian Doerrich, Christian Ledig**

xAILab, University of Bamberg

## Links

- **Project page:** https://shyam671.github.io/layer-selection-ood/

## Installation

```bash
conda activate MedOOD
pip install -r requirements.txt
```

Two CLIP backends are supported:

- **BiomedCLIP** (`--models_name biomed`) — pulled from Hugging Face via the pip
  `open_clip_torch` package.
- **UniMed-CLIP** (`--models_name ViT-B-16-quickgelu`) — loaded through the 
  [`src/open_clip`](src/open_clip) package from a local checkpoint (see below).

## Checkpoints & data roots

- Place the UniMed-CLIP checkpoint `unimed_clip_vit_b16.pt` under `--models_ckpt_dir`
  (default `/data/local/MedOOD/models/`).
- Datasets are read from `--dataset_root_dir` (default `/data/local/MedOOD/`) following the
  [OpenMIBOOD](https://github.com/remic-othr/OpenMIBOOD) layout.

## Usage

Run a single evaluation with `main.py`; results are **appended** to `results.txt`.

```bash
datasets=(oasis-near-ct oasis-far-heart oasis-far-chaos) 
for d in "${datasets[@]}"; do
   python main.py --text_prompt multi-mod --score MOD --dataset_dir "$d" 
done

```

### Key arguments

| Flag | Description |
|------|-------------|
| `--dataset_dir` | OOD benchmark to evaluate |
| `--models_name` | `biomed` (BiomedCLIP) or `ViT-B-16-quickgelu` (UniMed-CLIP) |
| `--score` | `MOD` (layer selection), or baselines `MCM` |
| `--text_prompt` | `single` / `multi` / `multi-mod` / `single-neg` prompt set |
| `--number_of_mod_layers` | Max size of the selected layer subset  |


## Citation

```bibtex
 @article{rai2026layer,
  title={Layer Selection in VLMs for Zero-Shot OOD Detection via Multi-Resolution Entropy Estimation},
  author={Rai, Shyam Nandan and Di Salvo, Francesco and Doerrich, Sebastian and Ledig, Christian},
  journal={arXiv preprint arXiv:2609.08524},
  year={2026}
}
```

## Acknowledgements

Built on [OpenCLIP](https://github.com/mlfoundations/open_clip),
[UniMed-CLIP](https://github.com/mbzuai-oryx/UniMed-CLIP),
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224),
and the [OpenMIBOOD](https://github.com/remic-othr/OpenMIBOOD) benchmark.
