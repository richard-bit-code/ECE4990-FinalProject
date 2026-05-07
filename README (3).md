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

| Dataset | Download Link | Classes |
|---|---|---|
| PKLot | [Kaggle](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset) | empty / occupied |
| CNRPark-EXT | [Kaggle](https://www.kaggle.com/datasets/ddsshubham/cnrpark-ext) | empty / occupied |
| Parking Slot Classification | [Kaggle](https://www.kaggle.com/datasets/basabbose/parking-slot-classification-dataset) | empty / occupied |

> All three datasets are free to download from Kaggle (free account required). See the **How to Run** section below — no API key needed.

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

1. Click the **"Open in Colab"** button above
2. Set runtime to **GPU**
   - Runtime → Change runtime type → T4 GPU
3. **Download the 3 datasets** from the Kaggle links above (free account, no API key needed)
   - Download each as a `.zip` file to your computer
4. Run **Section 3** in the notebook — it will prompt you to upload the zip files directly into Colab
5. Run all remaining cells in order
6. A results `.zip` will download automatically at the end

---

### Option 2: Local Setup

```bash
pip install torch torchvision timm einops scikit-learn matplotlib seaborn

# Clone the repo
git clone https://github.com/richard-bit-code/ECE4990-FinalProject.git
cd ECE4990-FinalProject

# Download the 3 datasets from Kaggle and place them as:
#   data/pklot/
#   data/cnrpark/
#   data/parking_slot/

# Launch the notebook
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
