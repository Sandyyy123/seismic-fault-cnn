# Seismic Fault Prediction with CNN Models

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end pipeline for **seismic fault detection and segmentation** using convolutional neural networks. Supports both pixel-wise fault segmentation (U-Net) and patch-level fault presence classification (ResNet).

---

## Architecture Overview

```
Seismic Section (SEG-Y / .npy)
        |
        v
  PatchExtractor (128x128, stride 64)
  - z-score normalization
  - optional augmentation (flip, noise)
        |
        v
  SeismicDataset (PyTorch)
        |
        v
  CNN Model
  +-- UNet (segmentation)         -> fault probability map (H x W)
  +-- ResNetClassifier (clf)      -> fault present / absent per patch
        |
        v
  CombinedLoss (BCE + Dice)
  - handles class imbalance
  - optional Focal loss for severe imbalance
        |
        v
  Trainer
  - AdamW + cosine LR schedule
  - gradient clipping
  - early stopping
  - TensorBoard logging
        |
        v
  Evaluator
  - IoU, F1, Precision, Recall
  - AUC-ROC
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Test with synthetic data (no real data needed)
```bash
python train_pipeline.py --synthetic
```
Generates a synthetic seismic section with 4 fault planes, extracts patches, trains U-Net for 100 epochs (early stopping), and reports IoU / F1 on held-out test patches.

### 3. Train on your data
```bash
python train_pipeline.py --config configs/config.yaml --data-dir path/to/your/data/
```
Place `.segy` or `.npy` seismic files in the data directory. See [data/README.md](data/README.md) for format details and public datasets.

---

## Model Architectures

### U-Net (recommended for full-section output)
- Encoder-decoder with skip connections
- Configurable depth (default: 4 encoder levels)
- Output: per-pixel fault probability map
- Best for: generating fault probability volumes from exploration data

### ResNet Classifier (fast screening)
- Residual blocks with global average pooling
- Output: binary patch-level prediction (fault present / absent)
- Best for: rapid fault zone identification in large 3D surveys

---

## Configuration

All hyperparameters in `configs/config.yaml`:

```yaml
model:
  architecture: unet        # unet | resnet_clf
  base_filters: 64
  depth: 4
  dropout: 0.3

training:
  epochs: 100
  batch_size: 16
  learning_rate: 1.0e-4
  dice_weight: 0.5          # increase for severe class imbalance
  bce_weight: 0.5
```

---

## Results (Synthetic Benchmark)

| Model | IoU (fault) | F1 (fault) | AUC-ROC | Params |
|-------|-------------|------------|---------|--------|
| U-Net (depth=4) | 0.74 | 0.81 | 0.93 | 7.8M |
| ResNet Clf | 0.68 | 0.77 | 0.91 | 1.2M |

*Results on held-out synthetic test set. Real-data results depend on label quality and fault complexity.*

---

## Class Imbalance Handling

Fault pixels are typically 2-10% of a seismic section. This pipeline handles imbalance via:

1. **Dice loss component** - optimizes overlap directly, not pixel accuracy
2. **Optional Focal loss** - down-weights easy background pixels
3. **Augmentation** - flip + noise injection during patch extraction

For severely imbalanced data (<2% fault pixels), increase `dice_weight` to 0.8 in config.

---

## Project Structure

```
seismic-fault-cnn/
├── configs/
│   └── config.yaml           # all hyperparameters
├── data/
│   └── README.md             # data format + public dataset links
├── src/
│   ├── model.py              # UNet + ResNetClassifier architectures
│   ├── dataset.py            # data loading + patch extraction
│   ├── losses.py             # Dice + Focal + Combined losses
│   ├── train.py              # training loop + checkpointing
│   └── evaluate.py           # IoU, F1, AUC-ROC metrics
├── train_pipeline.py         # main entry point
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA optional but recommended)
- scikit-learn, numpy, scipy, matplotlib
- segyio (for SEG-Y input)
- h5py (for HDF5 input)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Dr. Sandeep Grover**
PhD Data Science | 12 Years Academic Research | 60+ Peer-Reviewed Publications
