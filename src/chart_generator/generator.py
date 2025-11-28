"""Dataset generation with sliding windows."""

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .fetcher import fetch_ohlc
from .renderer import RenderConfig, render_candlesticks


@dataclass
class GeneratorConfig:
    """Configuration for dataset generation."""

    window_size: int = 30
    image_size: int = 224
    interval: str = "1d"


@dataclass
class SampleMetadata:
    """Metadata for a single generated sample."""

    ticker: str
    end_date: str
    label: str
    window_size: int
    image_size: int
    filename: str


def determine_label(current_close: float, next_close: float) -> str:
    """Determine label based on whether next day's close is higher than current close."""
    return "up" if next_close > current_close else "down"


def generate_samples_for_ticker(
    ticker: str,
    start: date,
    end: date,
    config: GeneratorConfig,
) -> list[tuple[pd.DataFrame, SampleMetadata]]:
    """
    Generate all samples for a single ticker.

    Returns list of (ohlc_window, metadata) tuples.
    """
    df = fetch_ohlc(ticker, start, end, config.interval)

    if len(df) < config.window_size + 1:
        raise ValueError(
            f"Insufficient data for {ticker}: got {len(df)} candles, "
            f"need at least {config.window_size + 1}"
        )

    samples: list[tuple[pd.DataFrame, SampleMetadata]] = []

    for i in range(len(df) - config.window_size):
        window = df.iloc[i : i + config.window_size]
        current_close = float(window.iloc[-1]["close"])
        next_close = float(df.iloc[i + config.window_size]["close"])

        label = determine_label(current_close, next_close)
        end_date_str = str(window.index[-1].date())

        filename = f"{ticker}_{end_date_str}_w{config.window_size}_s{config.image_size}_{i}.png"

        metadata = SampleMetadata(
            ticker=ticker,
            end_date=end_date_str,
            label=label,
            window_size=config.window_size,
            image_size=config.image_size,
            filename=filename,
        )

        samples.append((window.reset_index(drop=True), metadata))

    return samples


def generate_dataset(
    tickers: list[str],
    start: date,
    end: date,
    output_dir: Path,
    config: GeneratorConfig | None = None,
    render_config: RenderConfig | None = None,
) -> Path:
    """
    Generate dataset with all images (no train/val/test split).

    CV splitting should be handled during training using metadata timestamps.

    Args:
        tickers: List of ticker symbols
        start: Start date for data
        end: End date for data
        output_dir: Base output directory
        config: Generator configuration
        render_config: Rendering configuration

    Returns:
        Path to metadata CSV file
    """
    if config is None:
        config = GeneratorConfig()

    if render_config is None:
        render_config = RenderConfig(size=config.image_size)

    all_metadata: list[SampleMetadata] = []

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            samples = generate_samples_for_ticker(ticker, start, end, config)
            print(f"  Generated {len(samples)} samples")

            for window, meta in samples:
                img = render_candlesticks(window, render_config)

                label_dir = output_dir / meta.label
                label_dir.mkdir(parents=True, exist_ok=True)

                img.save(label_dir / meta.filename)
                all_metadata.append(meta)

        except ValueError as e:
            print(f"  Skipping {ticker}: {e}")
            continue

    if not all_metadata:
        raise ValueError("No samples generated from any ticker")

    metadata_path = output_dir / "metadata.csv"

    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "end_date", "label", "window_size", "image_size", "filename"])
        for meta in all_metadata:
            writer.writerow([
                meta.ticker,
                meta.end_date,
                meta.label,
                meta.window_size,
                meta.image_size,
                meta.filename,
            ])

    print(f"\nDataset generation complete!")
    print(f"  Total samples: {len(all_metadata)}")
    print(f"  Up: {sum(1 for m in all_metadata if m.label == 'up')}")
    print(f"  Down: {sum(1 for m in all_metadata if m.label == 'down')}")
    print(f"  Metadata: {metadata_path}")

    return metadata_path
