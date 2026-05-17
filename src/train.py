"""
Training loop with early stopping, LR scheduling, and TensorBoard logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import numpy as np


class Trainer:
    """
    Generic trainer for seismic fault segmentation/classification models.

    Args:
        model: PyTorch model (UNet or ResNetClassifier).
        criterion: Loss function.
        optimizer: Optimizer instance.
        scheduler: LR scheduler (optional).
        device: 'cuda', 'mps', or 'cpu'.
        checkpoint_dir: Where to save best-model checkpoints.
        log_dir: TensorBoard log directory.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        early_stopping_patience: int = 15,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.no_improve = 0

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for patches, labels in tqdm(loader, desc="Train", leave=False):
            patches = patches.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(patches)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for patches, labels in tqdm(loader, desc="Val", leave=False):
            patches = patches.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(patches)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
    ):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if hasattr(self.scheduler, "is_better"):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("LR", lr, epoch)

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        self.writer.close()

    def _save_checkpoint(self, epoch: int, val_loss: float):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
        print(f"  -> Saved checkpoint (val_loss={val_loss:.4f}) to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")


def build_optimizer_and_scheduler(model: nn.Module, cfg: dict):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sched_type = cfg["training"]["scheduler"]
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
    else:
        scheduler = None
    return optimizer, scheduler
