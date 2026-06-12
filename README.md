# Distributed LoRA Fine-Tuning Pipeline for Photorealistic Temple Image Generation

> A fully automated, end-to-end pipeline that fine-tunes Stable Diffusion 1.5 using Low-Rank Adaptation (LoRA) on a custom dataset of Ahmedabad Hindu temples — distributed across two consumer GPUs over a home LAN using Ray, with no cloud infrastructure required.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Ray](https://img.shields.io/badge/Ray-Distributed-028CF0?logo=ray&logoColor=white)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-Educational-green)

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Phases](#pipeline-phases)
- [Training Configuration](#training-configuration)
- [How LoRA Works Here](#how-lora-works-here)
- [Results](#results)
- [Project Structure](#project-structure)
- [Known Issues](#known-issues)
- [Future Work](#future-work)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

## Overview

Stable Diffusion 1.5 was trained on the internet-scale LAION-5B dataset. It encodes broad visual knowledge but has **no specific understanding** of Ahmedabad's Hindu temple architecture — the proportions of Nagara-style shikharas, the white Rajasthani marble, the gold finials, or the ornamental vocabulary of landmarks like BAPS Swaminarayan Mandir and Kalupur Swaminarayan Mandir.

This project closes that gap. It builds a complete pipeline that collects real temple photographs, fine-tunes SD 1.5 with LoRA to learn this specific architectural style, and produces photorealistic 2048×2048 temple imagery from text prompts — all on consumer hardware.

The entire system is implemented in **pure PyTorch**, deliberately avoiding the PEFT and basicsr libraries to keep every mathematical operation transparent and debuggable.

---

## Motivation

| Challenge | Solution |
| --- | --- |
| No diffusion model knows Ahmedabad temple architecture | Fine-tune SD 1.5 on real local temple photographs |
| Full fine-tuning of 860M params needs 22–28 GB VRAM | LoRA trains only 3.7M params (0.43%), ~30 MB overhead |
| No labelled temple dataset exists | Automated scraping + deduplication + auto-captioning |
| Output resolution limited to 512×512 | ESRGAN 4× super-resolution to 2048×2048 |
| Single GPU insufficient for full workload | Distribute across two GPUs using Ray over LAN |

---

## Key Features

- **Automated dataset generation** — web scraping with `icrawler`, perceptual-hash deduplication, centre-crop preprocessing, and BLIP-large auto-captioning following the DreamBooth convention.
- **Custom LoRA injection** — monkey-patches `nn.Linear.forward` directly, with no PEFT dependency, giving full mathematical traceability.
- **Distributed training** — heterogeneous GPU cluster (RTX 4090 + RTX 3080 Ti) coordinated by Ray over a standard Gigabit LAN, with dynamic node discovery.
- **Two model tracks** — comparative study of SD 1.5 (UNet + DDPM) versus Z-Image Turbo (DiT + flow-matching).
- **Pure-PyTorch ESRGAN** — RRDB-based 4× super-resolution with tiled inference to avoid VRAM exhaustion, with no `basicsr` dependency.

---

## System Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              RAY CLUSTER (LAN)                │
                          └─────────────────────────────────────────────┘

  WORKER NODE (Laptop)                              HEAD NODE (PC)
  RTX 3080 Ti · 12 GB                               RTX 4090 · 24 GB · 192.168.1.31
  ┌────────────────────────┐                        ┌────────────────────────────┐
  │ Phase 1: Dataset Gen   │                        │ Phase 3: LoRA Training      │
  │  • icrawler scrape     │   ── images ──▶        │  • inject LoRA (rank 16/8)  │
  │  • phash dedupe        │                        │  • AdamW + OneCycleLR       │
  │  • 1024² centre-crop   │                        │  • 50 epochs                │
  │  • BLIP-large caption  │                        ├────────────────────────────┤
  └────────────────────────┘                        │ Phase 4: Inference          │
  ┌────────────────────────┐                        │  • SD 1.5 + LoRA → 512²     │
  │ Phase 2: Cluster Test  │   ── ray status ──▶    ├────────────────────────────┤
  │  • node discovery      │                        │ Phase 5: ESRGAN             │
  │  • GPU validation      │                        │  • RRDB 4× → 2048²          │
  └────────────────────────┘                        └────────────────────────────┘
```

---

## Hardware Requirements

| Node | GPU | VRAM | RAM | Role |
| --- | --- | --- | --- | --- |
| **Head (PC)** | NVIDIA RTX 4090 | 24 GB | 64 GB | LoRA training + ESRGAN inference |
| **Worker (Laptop)** | NVIDIA RTX 3080 Ti | 12 GB | 32 GB | Preprocessing + BLIP captioning |

**Network:** Gigabit Ethernet LAN.

> **Note:** The pipeline can run on a single GPU by skipping the worker-node steps. A minimum of 12 GB VRAM is recommended for SD 1.5 LoRA training at rank 16.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/distributed-lora-temple.git
cd distributed-lora-temple
```

### 2. Create the environment

```bash
conda create -n lora-temple python=3.10 -y
conda activate lora-temple
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
pip install ray
pip install icrawler imagehash pillow
pip install numpy tqdm
```

> A consolidated `requirements.txt` is planned — see [Known Issues](#known-issues).

---

## Usage

### Step 1 — Start the Ray cluster

On the **head node** (PC):

```bash
ray start --head --port=6379
```

On the **worker node** (laptop):

```bash
ray start --address=192.168.1.31:6379
```

Verify the cluster:

```bash
python test_cluster.py
```

### Step 2 — Generate the dataset

```bash
python dataset_generator.py --keywords keywords.txt --output ./dataset --max-per-keyword 20
```

### Step 3 — Fine-tune

**Track A — Stable Diffusion 1.5:**

```bash
python finetune_sd15.py --data ./dataset --rank 16 --lr 1e-4 --epochs 50
```

**Track B — Z-Image Turbo:**

```bash
python finetune_zimage.py --data ./dataset --rank 8 --lr 3e-4 --epochs 50
```

### Step 4 — Generate images

```bash
python generate_image.py \
  --lora ./checkpoints/best_sd_lora.pt \
  --prompt "a photo of a white sks temple, intricate stone carvings, golden spire, photorealistic" \
  --steps 50 --guidance 7.5
```

### Step 5 — Super-resolution

```bash
python esrgan.py --input ./output/generated_512.png --scale 4 --tile 128
```

---

## Pipeline Phases

### Phase 1 — Dataset Generation (`dataset_generator.py`)

1. **Scrape** — `icrawler` queries Google and Bing with 84 Ahmedabad-specific keywords (~1,680 raw images).
2. **Deduplicate** — perceptual hash (phash) computes a 64-bit DCT fingerprint per image; pairs with Hamming distance below 10 are treated as duplicates.
3. **Filter** — removes images with a minimum dimension below 256 px.
4. **Preprocess** — centre-crops to square and resizes to 1024×1024 with LANCZOS resampling.
5. **Caption** — BLIP-large generates a description; the DreamBooth identifier is prepended:
   `"a photo of a sks temple, " + BLIP(image)`.
6. **Output** — paired `.jpg` + `.txt` files (~60 unique images).

### Phase 2 — Cluster Validation (`test_cluster.py`)

Confirms Ray connectivity, discovers nodes dynamically, and verifies GPU availability and VRAM per node.

### Phase 3 — LoRA Training (`finetune_sd15.py` / `finetune_zimage.py`)

Injects LoRA adapters and trains for 50 epochs with AdamW and a OneCycleLR schedule.

### Phase 4 — Inference (`generate_image.py` / `generate_zimage.py`)

Loads base model plus LoRA weights and generates 512×512 images (50 DDIM steps, CFG 7.5).

### Phase 5 — Super-Resolution (`esrgan.py`)

Pure-PyTorch RRDB ESRGAN upscales 512×512 → 2048×2048 with tiled inference.

---

## Training Configuration

| Hyperparameter | Track A — SD 1.5 | Track B — Z-Image Turbo |
| --- | --- | --- |
| Base model | `runwayml/stable-diffusion-v1-5` | `Tongyi-MAI/Z-Image-Turbo` |
| Backbone | UNet · DDPM noise prediction | DiT · flow-matching transformer |
| LoRA rank `r` | 16 | 8 |
| LoRA alpha `α` | 16 | 8 |
| Optimiser | AdamW (wd = 0.01) | AdamW (wd = 0.01) |
| Learning rate | 1e-4 | 3e-4 |
| LR schedule | OneCycleLR (pct_start = 0.1) | OneCycleLR (pct_start = 0.1) |
| Loss objective | MSE(ε_pred, ε) | MSE(v_pred, ε − z₀) |
| Epochs / dataset | 50 / 60 images | 50 / 60 images |
| Precision | fp32 (VAE NaN guard) | bf16 autocast |
| Trainable params | ~3.7M (0.43%) | ~1.85M (0.22%) |
| Checkpoint size | ~12 MB | ~6 MB |

---

## How LoRA Works Here

Instead of updating the full weight matrix **W** of each attention projection, LoRA learns two small matrices **A** and **B** whose product approximates the update:

```
W' = W + (α / r) · B · A
```

LoRA is injected into the four attention projections — `to_q`, `to_k`, `to_v`, `to_out` — of every attention block in the UNet (or DiT).

```python
# For each target nn.Linear layer (d_in = d_out = 768, rank r = 16):
A = nn.Parameter(torch.randn(rank, in_features) * 0.01)   # small Gaussian
B = nn.Parameter(torch.zeros(out_features, rank))         # zeros → identity at start

def lora_forward(self, x):
    base = F.linear(x, self.weight, self.bias).detach()   # frozen, no gradient
    delta = (x @ self.lora_a.T) @ self.lora_b.T * (alpha / rank)  # trainable
    return base + delta
```

Initialising **B** to zeros guarantees the adapter output is zero at the start of training, preserving the pretrained model exactly. Only **A** and **B** receive gradients; all 860M original weights stay frozen.

**Per-layer cost** at rank 16: `2 × 16 × 768 = 24,576` parameters versus `768 × 768 = 589,824` for full fine-tuning — a 96% reduction per layer.

---

## Results

After 50 training epochs on Track A (SD 1.5, rank 16), the model's output progressed through three clear phases:

| Stage | Epochs | Visual Characteristics |
| --- | --- | --- |
| **Early** | 1–10 | Scan-line artefacts, colour banding, rough silhouette only |
| **Mid** | 20–35 | Structural coherence, dome forms, emerging marble-and-gold palette |
| **Converged** | 50 | Clear multi-tiered shikhara, gold finials, ornamental colonnade, culturally accurate |

ESRGAN super-resolution then upscaled the 512×512 output to 2048×2048, revealing photorealistic marble veining, individual carved figures, and sharp shikhara tier boundaries.

### Parameter Efficiency

| Method | Trainable Params | Extra VRAM | Checkpoint |
| --- | --- | --- | --- |
| Full fine-tune | 860M | ~14 GB | ~3.4 GB |
| **LoRA r=16 (Track A)** | **3.7M (0.43%)** | **~30 MB** | **~12 MB** |
| LoRA r=8 (Track B) | 1.85M (0.22%) | ~15 MB | ~6 MB |

---

## Project Structure

```
distributed-lora-temple/
├── dataset_generator.py      # Phase 1: scrape, dedupe, resize, BLIP caption
├── test_cluster.py           # Phase 2: Ray cluster validation
├── finetune_sd15.py          # Phase 3: LoRA training — Track A (SD 1.5)
├── finetune_zimage.py        # Phase 3: LoRA training — Track B (Z-Image)
├── inspect_model.py          # Utility: inspect model architecture / layer names
├── test_zimage.py            # Utility: verify Z-Image model signatures
├── generate_image.py         # Phase 4: SD 1.5 + LoRA inference
├── generate_zimage.py        # Phase 4: Z-Image inference
├── esrgan.py                 # Phase 5: pure-PyTorch RRDB 4× super-resolution
├── keywords.txt              # 84 Ahmedabad-specific search keywords
├── dataset/                  # Generated image + caption pairs
├── checkpoints/              # Saved LoRA weights
└── output/                   # Generated and upscaled images
```

---

## Known Issues

| # | Issue | Impact | Fix |
| --- | --- | --- | --- |
| 1 | **LoRA scaling factor** uses `1/rank` instead of `alpha/rank` | Adapter dampened 16×, ~16× slower convergence; incompatible with ComfyUI/A1111 | Set `alpha = rank`, use `scale = alpha / rank` |
| 2 | **Z-Image velocity target** has a spurious `.unsqueeze(2)` | Produces a 5D tensor where 4D is expected; corrupted gradients, soft output | Remove `.unsqueeze(2)` |
| 3 | **Ray serialisation** passes tensors as Python lists | 10–50× slower inter-node transfer | Use `tensor.cpu().numpy()` for Apache Arrow zero-copy |
| 4 | **Small dataset** (~60 images) | Overfitting risk, biased toward BAPS Swaminarayan | Scale to 500+ images from 15+ temples |
| 5 | **No validation split** | Overfitting cannot be detected quantitatively | Add 90/10 train/val split with early stopping |
| 6 | **Hardcoded Windows paths** in 4 scripts | Reduced cross-platform portability | Use `os.path` / `pathlib` |
| 7 | **No `requirements.txt`** | Reproducibility risk | Pin dependency versions |

---

## Future Work

- [ ] Fix the three documented implementation bugs
- [ ] Scale the dataset to 500+ images across 15+ Ahmedabad temples
- [ ] Add a train/validation split with early stopping
- [ ] Compute FID and CLIP Score for quantitative evaluation
- [ ] Migrate the base model to SD-XL or FLUX.1
- [ ] Add numpy serialisation to Ray tasks for faster transfer
- [ ] Containerise with Docker and pin dependencies in `requirements.txt`
- [ ] Add ControlNet integration for architectural sketch-to-photo generation

---

## References

1. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR*, 2022.
2. R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR*, 2022.
3. N. Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation," *CVPR*, 2023.
4. J. Ho, A. Jain, P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020.
5. Y. Lipman et al., "Flow Matching for Generative Modeling," *ICLR*, 2023.
6. X. Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks," *ECCVW*, 2018.
7. P. Moritz et al., "Ray: A Distributed Framework for Emerging AI Applications," *USENIX OSDI*, 2018.
8. J. Li et al., "BLIP: Bootstrapping Language-Image Pre-training," *ICML*, 2022.

---

## Acknowledgements

This project was completed as a Comprehensive Project at the **Department of Information and Communication Technology, Pandit Deendayal Energy University (PDEU)**, Gandhinagar.

- **Author:** Daksh Milesh Dave (22BIT226)
- **Faculty Mentor:** Dr. Mohendra Roy
- **Industry Mentor:** Mr. RajdeepSinh Chavda

> The temple dataset was collected under fair-use educational research provisions. Generated images are for academic and demonstrative purposes only.

---

<div align="center">

**Believe Become**

</div>
