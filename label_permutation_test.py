"""
Label Permutation Test

Sanity check: shuffle labels randomly and train.
- If accuracy stays ~50% with both real AND shuffled labels → no signal in images
- If accuracy drops when shuffling → there was signal (shouldn't happen for us)

This confirms the model isn't picking up spurious patterns.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import date
import random

from cnn_classifier.dataset import CandlestickDataset
from cnn_classifier.model import CandlestickCNN


class ShuffledLabelDataset(Dataset):
    """Wrapper that permutes existing labels randomly."""

    def __init__(self, base_dataset: CandlestickDataset, seed: int = 42):
        self.base_dataset = base_dataset
        # Extract real labels and shuffle them
        real_labels = [base_dataset[i][1].item() for i in range(len(base_dataset))]
        random.seed(seed)
        random.shuffle(real_labels)
        self.shuffled_labels = real_labels

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, _ = self.base_dataset[idx]  # Ignore real label
        shuffled_label = torch.tensor(self.shuffled_labels[idx], dtype=torch.long)
        return img, shuffled_label


def train_and_evaluate(train_loader, val_loader, name: str, epochs: int = 30):
    """Train model and return final validation accuracy."""
    model = CandlestickCNN(image_size=50).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            preds = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    acc = 100 * correct / total
    print(f"{name}: {acc:.1f}% accuracy")
    return acc


def main():
    data_dir = Path("data/images")

    # Load datasets - use full date range with temporal split
    train_ds = CandlestickDataset(data_dir, end_date=date(2022, 12, 31))
    val_ds = CandlestickDataset(data_dir, start_date=date(2023, 1, 1))

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    # Print label distribution
    train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    up_count = sum(train_labels)
    down_count = len(train_labels) - up_count
    print(f"Train labels: {up_count} up ({100*up_count/len(train_labels):.1f}%), {down_count} down ({100*down_count/len(train_labels):.1f}%)")

    val_labels = [val_ds[i][1].item() for i in range(len(val_ds))]
    up_count = sum(val_labels)
    down_count = len(val_labels) - up_count
    print(f"Val labels:   {up_count} up ({100*up_count/len(val_labels):.1f}%), {down_count} down ({100*down_count/len(val_labels):.1f}%)")
    print()

    # Test 1: Real labels
    print("=" * 50)
    print("TEST 1: Real Labels")
    print("=" * 50)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    real_acc = train_and_evaluate(train_loader, val_loader, "Real labels")

    # Test 2: Shuffled labels
    print()
    print("=" * 50)
    print("TEST 2: Shuffled Labels (Permutation Test)")
    print("=" * 50)
    shuffled_train = ShuffledLabelDataset(train_ds, seed=42)
    shuffled_val = ShuffledLabelDataset(val_ds, seed=123)  # Different seed

    train_loader = DataLoader(shuffled_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(shuffled_val, batch_size=64, shuffle=False)
    shuffled_acc = train_and_evaluate(train_loader, val_loader, "Shuffled labels")

    # Summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Real labels:     {real_acc:.1f}%")
    print(f"Shuffled labels: {shuffled_acc:.1f}%")
    print()


if __name__ == "__main__":
    main()
