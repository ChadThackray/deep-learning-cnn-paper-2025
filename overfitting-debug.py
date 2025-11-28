from datetime import date
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from cnn_classifier.dataset import CandlestickDataset
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


if __name__ == "__main__":
    main()