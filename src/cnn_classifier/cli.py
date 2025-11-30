"""Command-line interface for CNN classifier."""

import argparse
from datetime import date, datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import CandlestickDataset, create_temporal_split
from .evaluate import evaluate_model, print_metrics
from .model import CandlestickCNN
from .train import TrainConfig, train_model


def parse_date(s: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_train(args: argparse.Namespace) -> None:
    """Handle 'train' command."""
    data_dir = Path(args.data)

    train_cutoff = parse_date(args.train_cutoff)
    val_cutoff = parse_date(args.val_cutoff) if args.val_cutoff else None

    train_dataset, val_dataset, _ = create_temporal_split(
        data_dir, train_cutoff, val_cutoff
    )

    # Print sample counts and label distribution
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    train_up = sum(train_labels)
    train_down = len(train_labels) - train_up
    print(f"Train samples: {len(train_dataset)} ({train_up} up, {train_down} down)")

    if val_dataset is not None:
        val_labels = [val_dataset[i][1].item() for i in range(len(val_dataset))]
        val_up = sum(val_labels)
        val_down = len(val_labels) - val_up
        print(f"Val samples: {len(val_dataset)} ({val_up} up, {val_down} down)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffle for time series
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        early_stopping_patience=args.patience,
    )

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_path=Path(args.output),
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Handle 'evaluate' command."""
    data_dir = Path(args.data)
    model_path = Path(args.model)

    test_start = parse_date(args.test_start)
    test_dataset = CandlestickDataset(data_dir, start_date=test_start)

    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    metrics = evaluate_model(
        model_path=model_path,
        test_loader=test_loader,
        image_size=args.image_size,
    )

    print_metrics(metrics)


def cmd_predict(args: argparse.Namespace) -> None:
    """Handle 'predict' command for single image."""
    model_path = Path(args.model)
    image_path = Path(args.image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandlestickCNN(image_size=args.image_size).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Load and preprocess image
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        prediction = "up" if pred_class == 1 else "down"
        confidence = probs[0, pred_class].item()

    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CNN classifier for candlestick chart prediction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data", required=True, help="Path to data directory with metadata.csv"
    )
    train_parser.add_argument(
        "--train-cutoff",
        required=True,
        help="End date for training data (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--val-cutoff",
        help="End date for validation data (YYYY-MM-DD), optional",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs (default: 50)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    train_parser.add_argument(
        "--image-size", type=int, default=50, help="Image size (default: 50)"
    )
    train_parser.add_argument(
        "--patience", type=int, help="Early stopping patience (optional)"
    )
    train_parser.add_argument(
        "--output", default="./models/model.pt", help="Output model path"
    )
    train_parser.set_defaults(func=cmd_train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument(
        "--data", required=True, help="Path to data directory with metadata.csv"
    )
    eval_parser.add_argument(
        "--test-start", required=True, help="Start date for test data (YYYY-MM-DD)"
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    eval_parser.add_argument(
        "--image-size", type=int, default=50, help="Image size (default: 50)"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict single image"
    )
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--image", required=True, help="Path to image")
    predict_parser.add_argument(
        "--image-size", type=int, default=50, help="Image size (default: 50)"
    )
    predict_parser.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
