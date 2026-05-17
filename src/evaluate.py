"""
Evaluation metrics for seismic fault detection.

Metrics:
  - IoU (Intersection over Union) for segmentation
  - F1 / Precision / Recall for fault class
  - AUC-ROC for classification confidence
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from typing import Dict


class Evaluator:
    def __init__(self, model, device: str = "cpu", threshold: float = 0.5):
        self.model = model
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []

        for patches, labels in loader:
            patches = patches.to(self.device)
            logits = self.model(patches)
            probs = F.softmax(logits, dim=1)[:, 1]  # fault class probability

            if logits.dim() == 4:
                preds = (probs > self.threshold).long().cpu().numpy().flatten()
                probs_np = probs.cpu().numpy().flatten()
                labels_np = labels.numpy().flatten()
            else:
                preds = (probs > self.threshold).long().cpu().numpy()
                probs_np = probs.cpu().numpy()
                labels_np = labels.numpy()

            mask = labels_np >= 0  # filter unlabeled
            all_preds.extend(preds[mask].tolist())
            all_probs.extend(probs_np[mask].tolist())
            all_labels.extend(labels_np[mask].tolist())

        return self._compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        eps = 1e-8
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        results = {
            "iou_fault": float(iou),
            "f1_fault": float(f1),
            "precision_fault": float(precision),
            "recall_fault": float(recall),
            "accuracy": float(accuracy),
            "auc_roc": float(auc),
        }
        return results

    def print_report(self, loader: DataLoader):
        metrics = self.evaluate(loader)
        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"  {k:25s}: {v:.4f}")
        print("==========================\n")
        return metrics
