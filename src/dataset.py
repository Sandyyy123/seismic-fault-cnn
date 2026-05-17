"""
Seismic dataset handling and patch extraction.

Supports:
  - SEG-Y via segyio
  - NumPy .npy / .npz arrays
  - HDF5 .h5 files
  - Synthetic data generation for testing
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Optional
import random


class PatchExtractor:
    """
    Extract overlapping 2D patches from a seismic section and its fault label mask.

    Args:
        patch_size: Square patch dimension in samples.
        stride: Step between patch centers. Use stride < patch_size for overlap.
        normalize: Apply per-patch z-score normalization.
        augment: Random flips and noise injection during extraction.
    """

    def __init__(
        self,
        patch_size: int = 128,
        stride: int = 64,
        normalize: bool = True,
        augment: bool = False,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.normalize = normalize
        self.augment = augment

    def extract(
        self, seismic: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Args:
            seismic: 2D array (n_traces, n_samples) of amplitude values.
            labels:  2D array (n_traces, n_samples) of binary fault labels, or None.

        Returns:
            (patches, label_patches) lists of equal length.
        """
        H, W = seismic.shape
        ps = self.patch_size
        patches, label_patches = [], []

        for i in range(0, H - ps + 1, self.stride):
            for j in range(0, W - ps + 1, self.stride):
                patch = seismic[i : i + ps, j : j + ps].astype(np.float32)
                if self.normalize:
                    mu, sigma = patch.mean(), patch.std()
                    patch = (patch - mu) / (sigma + 1e-8)
                if self.augment:
                    patch = self._augment(patch)
                patches.append(patch)

                lp = labels[i : i + ps, j : j + ps] if labels is not None else None
                label_patches.append(lp)

        return patches, label_patches

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            patch = np.fliplr(patch).copy()
        if random.random() > 0.5:
            patch = np.flipud(patch).copy()
        noise_level = random.uniform(0.0, 0.05)
        patch = patch + np.random.randn(*patch.shape).astype(np.float32) * noise_level
        return patch


class SeismicDataset(Dataset):
    """
    PyTorch Dataset wrapping seismic patches.

    Usage:
        extractor = PatchExtractor(patch_size=128, stride=64, augment=True)
        patches, labels = extractor.extract(seismic_section, fault_mask)
        ds = SeismicDataset(patches, labels, task='segmentation')
    """

    def __init__(
        self,
        patches: List[np.ndarray],
        labels: Optional[List[Optional[np.ndarray]]] = None,
        task: str = "segmentation",  # 'segmentation' | 'classification'
    ):
        self.patches = patches
        self.labels = labels
        self.task = task

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = torch.from_numpy(self.patches[idx]).unsqueeze(0)  # (1, H, W)

        if self.labels is None or self.labels[idx] is None:
            return patch, torch.tensor(-1)

        lbl = self.labels[idx]
        if self.task == "segmentation":
            target = torch.from_numpy(lbl.astype(np.int64))  # (H, W)
        else:
            # Classification: fault present if any pixel is fault
            target = torch.tensor(int(lbl.max() > 0), dtype=torch.long)

        return patch, target


# ---------------------------------------------------------------------------
# Data loaders for different file formats
# ---------------------------------------------------------------------------

def load_segy(filepath: str) -> np.ndarray:
    """Load a 2D SEG-Y file and return amplitude array (n_traces, n_samples)."""
    try:
        import segyio
    except ImportError:
        raise ImportError("segyio not installed. Run: pip install segyio")

    with segyio.open(filepath, ignore_geometry=True) as f:
        data = np.array([tr.astype(np.float32) for tr in f.trace])
    return data


def load_npy(filepath: str) -> np.ndarray:
    """Load seismic data from .npy or .npz file."""
    p = Path(filepath)
    if p.suffix == ".npz":
        npz = np.load(filepath)
        key = list(npz.keys())[0]
        return npz[key].astype(np.float32)
    return np.load(filepath).astype(np.float32)


def load_h5(filepath: str, dataset_key: str = "data") -> np.ndarray:
    """Load seismic data from HDF5 file."""
    import h5py
    with h5py.File(filepath, "r") as f:
        return f[dataset_key][:].astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing without real data)
# ---------------------------------------------------------------------------

def generate_synthetic_section(
    n_traces: int = 512,
    n_samples: int = 512,
    n_faults: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic 2D seismic section with layered reflectors and faults.

    Returns:
        seismic: (n_traces, n_samples) float32 amplitude array.
        fault_mask: (n_traces, n_samples) int64 binary fault label.
    """
    rng = np.random.default_rng(seed)
    seismic = np.zeros((n_traces, n_samples), dtype=np.float32)
    fault_mask = np.zeros((n_traces, n_samples), dtype=np.int64)

    # Generate layered reflectors
    n_layers = 20
    layer_positions = np.sort(rng.integers(20, n_samples - 20, n_layers))
    for pos in layer_positions:
        amplitude = rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
        thickness = rng.integers(2, 8)
        for t in range(n_traces):
            slight_dip = int(rng.integers(-3, 4) * t / n_traces)
            p = max(0, min(n_samples - thickness, pos + slight_dip))
            seismic[t, p : p + thickness] = amplitude * rng.uniform(0.8, 1.2)

    # Add background noise
    seismic += rng.normal(0, 0.05, seismic.shape).astype(np.float32)

    # Inject fault planes (vertical displacement zones)
    fault_traces = sorted(rng.integers(50, n_traces - 50, n_faults).tolist())
    for ft in fault_traces:
        throw = rng.integers(10, 40)
        width = rng.integers(2, 6)
        for t in range(max(0, ft - width), min(n_traces, ft + width)):
            seismic[t, :] = np.roll(seismic[t, :], throw)
            fault_mask[t, :] = 1

    return seismic, fault_mask
