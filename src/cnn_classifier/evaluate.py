"""Evaluation metrics for candlestick CNN."""

from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader

from .model import CandlestickCNN


@dataclass
class EvalMetrics:
    """Evaluation metrics from the paper."""

    accuracy: float
    sensitivity: float  # TP / (TP + FN) - recall for positive class
    specificity: float  # TN / (TN + FP) - recall for negative class
    mcc: float  # Matthews Correlation Coefficient
    tp: int
    tn: int
    fp: int
    fn: int


def evaluate_model(
    model_path: Path,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    image_size: int = 50,
    device: torch.device | None = None,
) -> EvalMetrics:
    """
    Evaluate trained model on test data.

    Args:
        model_path: Path to saved model weights
        test_loader: Test data loader
        image_size: Image size model was trained on
        device: Device to evaluate on

    Returns:
        EvalMetrics with all paper metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandlestickCNN(image_size=image_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).tolist()
            labels_list = labels.tolist()

            # Handle single item batches
            if isinstance(preds, int):
                preds = [preds]
            if isinstance(labels_list, int):
                labels_list = [labels_list]

            all_preds.extend(preds)
            all_labels.extend(labels_list)

    return calculate_metrics(all_labels, all_preds)


def calculate_metrics(labels: list[int], preds: list[int]) -> EvalMetrics:
    """
    Calculate all evaluation metrics.

    Args:
        labels: Ground truth labels (1 = up, 0 = down)
        preds: Predicted labels

    Returns:
        EvalMetrics
    """
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Sensitivity (Recall): TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # MCC using sklearn
    mcc = float(matthews_corrcoef(labels, preds)) if len(set(labels)) > 1 else 0.0

    return EvalMetrics(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        mcc=mcc,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def print_metrics(metrics: EvalMetrics) -> None:
    """Print metrics in a formatted table."""
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Accuracy:    {metrics.accuracy:.4f}")
    print(f"Sensitivity: {metrics.sensitivity:.4f}")
    print(f"Specificity: {metrics.specificity:.4f}")
    print(f"MCC:         {metrics.mcc:.4f}")
    print("-" * 40)
    print("Confusion Matrix:")
    print(f"  TP: {metrics.tp:5d}  FP: {metrics.fp:5d}")
    print(f"  FN: {metrics.fn:5d}  TN: {metrics.tn:5d}")
    print("=" * 40)
