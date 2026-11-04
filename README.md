# HDMix: Hierarchical Discriminative Mixing

**Label-Preserving Data Augmentation for Domain-Specific Image Classification**

## Overview

HDMix is a data augmentation pipeline for domain-specific image classification that creates diverse, realistic training composites **without requiring a generative model**. Unlike Mixup/CutMix, which blend labels into semantically invalid combinations (e.g., 70% hawk + 30% sparrow), HDMix preserves the original label by always keeping the source object intact. Compared to SGD-Mix, it avoids the expensive per-dataset DreamBooth+LoRA diffusion model pre-training step.

## Problem Statement

| Challenge | Description |
|---|---|
| **Limited Data** | Fine-grained datasets have ~30 images per class; standard training overfits quickly. |
| **Label Ambiguity** | Mixup/CutMix assign blended labels that are semantically invalid. |
| **Compute Cost** | SGD-Mix needs a per-dataset diffusion model (DreamBooth+LoRA) before training starts. |

**What we need:** Diversity (more varied images without new labelling), Faithfulness (natural, realistic composites), Label Clarity (labels unchanged — source object always preserved), and Efficiency (no generative model).

## Methodology — HDMix Pipeline

The pipeline has four stages:

1. **Soft Saliency** — Sobel + Laplacian filters detect edges and texture. A Gaussian center prior is applied. Sigmoid thresholding produces a smooth alpha ∈ (0, 1); no hard Otsu cut.
2. **Bg Partner** — `bg = 1 - alpha`. Downsample to 28×28. Pairwise dot-product scores rank all pairs; top-K partners are selected, cycled per epoch for diversity.
3. **Compositing** — Strength `s = f(entropy(alpha))`; high entropy → lower s. Background is blurred and noise-scaled by s. Final blend: `α·src + (1-α)·bg`.
4. **Conf. Filter** — Model evaluates the composite in eval mode. Composites where `pred ≠ label` or confidence is below threshold are discarded. Active after warmup.

## Experimental Setup

**Datasets & Benchmarks**

| Experiment | Datasets |
|---|---|
| Exp 1 — Fine-Grained | CUB, Aircraft, Flowers, Dogs |
| Exp 2 — Long-Tail | CUB-LT, Flower-LT (IF = 100, 50, 10) |
| Exp 3 — Few-Shot | CUB (1, 5, 10, all-shot settings) |
| Exp 4 — Robustness | Waterbird (CUB foreground + Places background) |

**Implementation Details**
- Backbone: ResNet50 @ 448×448
- Optimizer: AdamW — LR 1e-4 (backbone), 1e-3 (head)
- Schedule: Cosine decay + 5-epoch linear warmup
- AMP mixed precision (GradScaler)
- TTA: 5 crops × 3 scales = 15 views
- Label smoothing: 0.05

## Results

### Exp 1 — Fine-Grained (ResNet50 @448)

| Dataset | Baseline | SGD-Mix | HDMix (ours) | Δ vs SGD |
|---|---|---|---|---|
| Aircraft | 85.21% | 90.16% | 85.21% | -4.95% |
| CUB | 82.27% | 87.66% | 82.65% | -5.01% |
| Flowers | 95.46% | 99.59% | 95.43% | -4.16% |
| Dogs | 90.96% | 88.01% | 91.01% | **+3.00%** |

### Exp 2 — CUB-LT (IF=100)

| Split | HDMix | Paper | Δ |
|---|---|---|---|
| All | 49.91% | 49.48% | **+0.43%** |
| Medium | 58.17% | 66.80% | -8.63% |
| Few | 21.31% | 37.00% | -15.69% |

### Exp 3 — Few-Shot CUB

| Shot | HDMix | Paper | Δ |
|---|---|---|---|
| 1-shot | 17.40% | 25.68% | -8.28% |
| 5-shot | 42.11% | 59.61% | -17.50% |
| 10-shot | 58.97% | 71.90% | -12.93% |
| All | 78.22% | 82.88% | -4.66% |

### Exp 4 — Waterbird Background Robustness

WB+Land **+29.6%** · LB+Land **+23.5%** · LB+Water **+25.5%** · Average **+24.8%**

Foreground–background disentanglement prevents background shortcut learning — all four groups beat SGD-Mix.

## Ablation Study

**HDMix vs Baseline (same backbone & training)**

| Setting | Baseline | HDMix | Δ |
|---|---|---|---|
| Dogs | 91.0% | 91.0% | +0.05% |
| CUB | 82.3% | 82.7% | +0.38% |
| LT All IF100 | 49.5% | 49.9% | +0.43% |
| Waterbird Avg | 97.6% | 97.3% | +0.31% |

Few-shot comparison:

| Shot | Baseline | HDMix | Δ |
|---|---|---|---|
| 1-shot | 17.0% | 17.40% | +0.31% |
| 5-shot | 39.02% | 42.11% | **+3.09%** |
| 10-shot | 59.91% | 58.97% | -0.94% |
| All | 76.61% | 78.22% | +1.61% |

## Conclusion

- **Zero generation overhead** — fully on-GPU; no diffusion model pre-training required.
- **Robustness:** +24.8% average; beats SGD-Mix on all four Waterbird groups.
- **5-shot:** +3.09% clear gain over baseline in low-data few-shot regime.
- **LT All-acc:** +0.43%; CUB-LT IF=100 overall accuracy exceeds baseline.
- **FGVC gap vs SGD-Mix:** 4–5% below on Aircraft/CUB — diffusion synthesis has a fundamental generative edge.

## Key Takeaway

HDMix offers a practical, label-preserving, generation-free augmentation alternative that excels in robustness and low-data regimes, while acknowledging that diffusion-based synthesis retains an edge on fine-grained visual classification benchmarks.

---

*Based on the HDMix presentation slides.*
