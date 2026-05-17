"""
End-to-end training pipeline for seismic fault detection.

Usage:
    python train_pipeline.py --config configs/config.yaml --data-dir data/
    python train_pipeline.py --synthetic          # quick test with generated data
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.model import build_model
from src.dataset import SeismicDataset, PatchExtractor, generate_synthetic_section, load_npy, load_h5
from src.losses import CombinedLoss
from src.train import Trainer, build_optimizer_and_scheduler
from src.evaluate import Evaluator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data loading
    extractor = PatchExtractor(
        patch_size=cfg["data"]["patch_size"],
        stride=cfg["data"]["patch_stride"],
        normalize=cfg["data"]["normalize"],
        augment=cfg["data"]["augment"],
    )

    if args.synthetic:
        print("Generating synthetic seismic section...")
        seismic, fault_mask = generate_synthetic_section(n_traces=512, n_samples=512, n_faults=4)
        patches, label_patches = extractor.extract(seismic, fault_mask)
    else:
        print(f"Loading data from {args.data_dir}...")
        import glob
        segy_files = glob.glob(f"{args.data_dir}/*.segy") + glob.glob(f"{args.data_dir}/*.sgy")
        npy_files = glob.glob(f"{args.data_dir}/*.npy")
        if not segy_files and not npy_files:
            raise FileNotFoundError(f"No .segy or .npy files found in {args.data_dir}. Use --synthetic for testing.")
        patches, label_patches = [], []
        for f in (segy_files or npy_files):
            s = load_npy(f) if f.endswith(".npy") else None
            if s is not None:
                p, lp = extractor.extract(s)
                patches.extend(p)
                label_patches.extend(lp)

    print(f"Total patches: {len(patches)}")

    # Dataset splits
    full_ds = SeismicDataset(patches, label_patches, task=cfg["model"]["architecture"] == "resnet_clf" and "classification" or "segmentation")
    n = len(full_ds)
    n_train = int(n * cfg["data"]["train_split"])
    n_val = int(n * cfg["data"]["val_split"])
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2)

    # Model, loss, optimizer
    model = build_model(cfg)
    print(f"Model: {cfg['model']['architecture']} | Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CombinedLoss(
        dice_weight=cfg["training"]["dice_weight"],
        bce_weight=cfg["training"]["bce_weight"],
        num_classes=cfg["model"]["num_classes"],
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        log_dir=cfg["paths"]["log_dir"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
    )

    print(f"\nTraining for up to {cfg['training']['epochs']} epochs...")
    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])

    # Load best checkpoint for evaluation
    trainer.load_checkpoint(f"{cfg['paths']['checkpoint_dir']}/best_model.pt")
    evaluator = Evaluator(model, device=device, threshold=cfg["inference"]["threshold"])
    print("\nTest set evaluation:")
    evaluator.print_report(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seismic Fault CNN Training Pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for quick testing")
    args = parser.parse_args()
    main(args)
