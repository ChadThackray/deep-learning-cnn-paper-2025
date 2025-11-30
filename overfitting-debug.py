from datetime import date
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from cnn_classifier.dataset import CandlestickDataset
from cnn_classifier.model import CandlestickCNN
from cnn_classifier.train import TrainConfig, train_model


def main() -> None:
    data_dir = Path("data/images")
    train_cutoff = date(2022, 12, 31)

    # Full training dataset up to the cutoff
    full_train = CandlestickDataset(data_dir, end_date=train_cutoff)

    # Small subset to test overfitting (adjust size if you like)
    subset_size = 64
    indices = list(range(min(subset_size, len(full_train))))
    subset = Subset(full_train, indices)

    train_loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=True,  # shuffle is fine here; we’re not respecting time
    )

    # Many epochs so the model has time to memorize this tiny set
    config = TrainConfig(
        epochs=200,
        batch_size=32,
        learning_rate=0.001,
        image_size=50,
        early_stopping_patience=None,
    )

    train_model(
        train_loader=train_loader,
        val_loader=None,
        config=config,
        output_path=Path("models/overfit_debug.pt"),
    )

    # Evaluate on the same subset (should get ~100% if model can learn)
    print("\n" + "=" * 50)
    print("EVALUATION ON TRAINING SUBSET")
    print("=" * 50)

    model = CandlestickCNN(image_size=50).cuda()
    model.load_state_dict(torch.load("models/overfit_debug.pt", weights_only=True))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            preds = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    print(f"Accuracy: {100 * correct / total:.1f}%")
    print("(Should be ~100% if model can memorize)")

    # Evaluate on actual test set
    print("\n" + "=" * 50)
    print("EVALUATION ON TEST SET")
    print("=" * 50)

    test_ds = CandlestickDataset(data_dir, start_date=date(2023, 1, 1))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            preds = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    print(f"Accuracy: {100 * correct / total:.1f}%")
    print("(Should be ~50% - no generalization)")


if __name__ == "__main__":
    main()