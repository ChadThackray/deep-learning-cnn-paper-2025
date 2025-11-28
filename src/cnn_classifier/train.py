"""Training loop for candlestick CNN."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CandlestickCNN


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    image_size: int = 50
    early_stopping_patience: int | None = None


@dataclass
class TrainResult:
    """Training results."""

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float


def train_model(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    config: TrainConfig,
    output_path: Path,
    device: torch.device | None = None,
) -> TrainResult:
    """
    Train the CNN model.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        config: Training configuration
        output_path: Path to save best model
        device: Device to train on (auto-detected if None)

    Returns:
        TrainResult with training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandlestickCNN(image_size=config.image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"Training on {device}")
    print(f"Train samples: {len(train_loader.dataset)}")  # type: ignore[arg-type]
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")  # type: ignore[arg-type]

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_path)
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Early stopping
            if (
                config.early_stopping_patience is not None
                and patience_counter >= config.early_stopping_patience
            ):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            # No validation, save after each epoch
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            best_epoch = epoch
            print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}")

    print(f"\nTraining complete. Best epoch: {best_epoch + 1}")
    print(f"Model saved to {output_path}")

    return TrainResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss if val_loader else train_losses[-1],
    )
