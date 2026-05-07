# Parking Lot Occupancy Detection — EfficientNet + MobileViT

**ECE 4990 Final Project | Cal Poly Pomona**
**Authors:** Francisco Pulido, Richard Pablo

---

## Overview

This project proposes a **hybrid CNN-Transformer architecture** for parking lot occupancy detection. We extend **EfficientNet-B0** with a **MobileViT block** that fuses local CNN features with lightweight global self-attention — directly targeting the known weakness of CNN-only models under occlusion, shadows, and lighting variation.

### Key Contribution

Standard CNNs (including EfficientNet-P from the surveyed paper) treat spatial regions independently. Our proposed **MobileViT block** inserted after the EfficientNet backbone applies local-global attention to the feature map before classification, capturing spatial context across the full parking slot.

```
Input Image
    │
    ▼
EfficientNet-B0 (backbone, pretrained)
    │  Feature map: (B, 320, 7, 7)
    ▼
MobileViT Block  ← NOVELTY
    │  Local depthwise conv
    │  + Patch-based Transformer (global attention)
    │  + Residual fusion
    ▼
1×1 Conv → AvgPool → Dropout → Linear
    │
    ▼
Binary Classification (Empty / Occupied)
```

---

## Datasets

| Dataset | Source | Images Used | Classes |
|---|---|---|---|
| PKLot | [Kaggle](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset) | up to 4000 | empty / occupied |
| CNRPark-EXT | [Kaggle](https://www.kaggle.com/datasets/ddsshubham/cnrpark-ext) | up to 4000 | empty / occupied |
| Parking Slot Classification | [Kaggle](https://www.kaggle.com/datasets/basabbose/parking-slot-classification-dataset) | up to 4000 | empty / occupied |

---

## Results

| Dataset | Baseline (EfficientNet-B0) F1 | Proposed (+ MobileViT) F1 |
|---|---|---|
| PKLot | — | — |
| CNRPark-EXT | — | — |
| ParkingSlot | — | — |
| Combined | — | — |

*(Fill in after running the notebook)*

---

## Repository Structure

```
├── parking_occupancy_hybrid.ipynb   # Main Colab notebook (train + eval + plots)
├── README.md                        # This file
└── results/                         # Output plots and tables (add after running)
    ├── training_curves.png
    ├── f1_comparison.png
    ├── confusion_matrices.png
    ├── lr_ablation.png
    ├── bs_ablation.png
    └── results_table.csv
```

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open `parking_occupancy_hybrid.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run **Section 3** to upload your `kaggle.json` and download datasets
4. Run all cells in order
5. Results zip downloads automatically at the end

### Option 2: Local

```bash
pip install torch torchvision timm einops scikit-learn matplotlib seaborn kaggle

# Set up Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

jupyter notebook parking_occupancy_hybrid.ipynb
```

---

## Model Architecture Details

### Baseline: EfficientNet-B0
- Pretrained on ImageNet
- Final classifier: Dropout(0.3) → Linear(1280, 2)
- ~5.3M parameters

### Proposed: EfficientNet-B0 + MobileViT Block
- Same EfficientNet-B0 backbone (pretrained, `global_pool=''`)
- MobileViT block on 320-channel feature map:
  - Local: 3×3 depthwise conv + BN + SiLU + 1×1 conv
  - Global: Patch unfolding → 2-layer Transformer (4 heads) → fold back
  - Residual connection
- 1×1 projection conv: 320 → 1280 channels
- AdaptiveAvgPool → Dropout(0.3) → Linear(1280, 2)

### Training Setup
- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Scheduler: Cosine Annealing
- Augmentation: RandomHorizontalFlip, RandomRotation(15°), ColorJitter
- Image size: 224×224
- Batch size: 32
- Epochs: 10
- Metric: Macro F1-Score

---

## References

1. Novitskiy et al. (2023). *Parking Lot Occupancy Detection Using Deep Learning.* arXiv:2306.04288
2. Mehta & Rastegari (2021). *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer.* arXiv:2110.02178
3. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019.
