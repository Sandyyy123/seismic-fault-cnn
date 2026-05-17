# Data Directory

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| SEG-Y | `.segy`, `.sgy` | Standard industry format; requires `segyio` |
| NumPy | `.npy`, `.npz` | Fastest; use for pre-processed data |
| HDF5 | `.h5` | For large 3D cubes; specify dataset key in config |

## Public Datasets (No License Issues)

### 1. SEAM Phase I Benchmark (Gulf of Mexico synthetic)
- Download: https://wiki.seg.org/wiki/SEAM
- Format: SEG-Y 3D cube; extract 2D inlines as sections

### 2. F3 Netherlands Offshore (Open Seismic Repository)
- Download: https://github.com/olivesgatech/facies_classification_benchmark
- Well-annotated facies + fault data

### 3. SEG Salt and Overthrust Models
- Download: https://github.com/seg-models
- Clean fault structures; good for segmentation benchmarking

### 4. Synthetic Data (built-in)
Run with `--synthetic` flag to generate synthetic seismic sections with ground-truth fault masks:
```bash
python train_pipeline.py --synthetic
```

## Data Preparation

Place your seismic files here as `.segy` or `.npy`:
```
data/
  train_section_001.npy
  train_section_001_labels.npy   # same name + _labels suffix for masks
  train_section_002.npy
  ...
```

Fault label files should be binary arrays (0=background, 1=fault) of the same shape as the seismic section.
