"""PyTorch Dataset for candlestick images with temporal splitting."""

import csv
from datetime import date
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CandlestickDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset for candlestick chart images.

    Supports temporal splitting based on end_date in metadata.
    """

    def __init__(
        self,
        data_dir: Path,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing up/, down/, and metadata.csv
            start_date: Only include samples with end_date >= start_date
            end_date: Only include samples with end_date <= end_date
        """
        self.data_dir = data_dir
        self.samples: list[tuple[Path, int]] = []

        metadata_path = data_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {data_dir}")

        with open(metadata_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_date = date.fromisoformat(row["end_date"])

                # Apply temporal filters
                if start_date is not None and sample_date < start_date:
                    continue
                if end_date is not None and sample_date > end_date:
                    continue

                label = 1 if row["label"] == "up" else 0
                image_path = data_dir / row["label"] / row["filename"]

                if image_path.exists():
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[idx]

        # Load RGB image
        img = Image.open(image_path).convert("RGB")

        # Convert to tensor and normalize to [0, 1]
        # Shape: (H, W, 3) -> (3, H, W)
        import numpy as np
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


def create_temporal_split(
    data_dir: Path,
    train_cutoff: date,
    val_cutoff: date | None = None,
) -> tuple[CandlestickDataset, CandlestickDataset | None, CandlestickDataset]:
    """
    Create train/val/test datasets with temporal split.

    Args:
        data_dir: Directory containing the dataset
        train_cutoff: End date for training data (exclusive)
        val_cutoff: End date for validation data (exclusive), if None no val set

    Returns:
        Tuple of (train_dataset, val_dataset or None, test_dataset)
    """
    train_dataset = CandlestickDataset(data_dir, end_date=train_cutoff)

    if val_cutoff is not None:
        # Train: <= train_cutoff
        # Val: train_cutoff < date <= val_cutoff
        # Test: > val_cutoff
        from datetime import timedelta

        val_start = train_cutoff + timedelta(days=1)
        val_dataset = CandlestickDataset(
            data_dir, start_date=val_start, end_date=val_cutoff
        )
        test_start = val_cutoff + timedelta(days=1)
        test_dataset = CandlestickDataset(data_dir, start_date=test_start)
    else:
        # Train: <= train_cutoff
        # Test: > train_cutoff
        from datetime import timedelta

        val_dataset = None
        test_start = train_cutoff + timedelta(days=1)
        test_dataset = CandlestickDataset(data_dir, start_date=test_start)

    return train_dataset, val_dataset, test_dataset
