"""Command-line interface for chart generator."""

import argparse
from datetime import date, datetime
from pathlib import Path

from .fetcher import fetch_ohlc
from .generator import GeneratorConfig, generate_dataset
from .renderer import RenderConfig, render_candlesticks


def parse_date(s: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_generate(args: argparse.Namespace) -> None:
    """Handle 'generate' command."""
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    config = GeneratorConfig(
        window_size=args.window_size,
        image_size=args.image_size,
        interval=args.interval,
    )

    render_config = RenderConfig(size=args.image_size)

    generate_dataset(
        tickers=tickers,
        start=parse_date(args.start),
        end=parse_date(args.end),
        output_dir=Path(args.output),
        config=config,
        render_config=render_config,
    )


def cmd_render(args: argparse.Namespace) -> None:
    """Handle 'render' command for single image testing."""
    df = fetch_ohlc(
        ticker=args.ticker.upper(),
        start=parse_date(args.start),
        end=parse_date(args.end),
        interval=args.interval,
    )

    if args.candles:
        df = df.tail(args.candles)

    render_config = RenderConfig(size=args.image_size)
    img = render_candlesticks(df, render_config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    print(f"Saved {len(df)} candles to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate candlestick chart images for CNN training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser(
        "generate", help="Generate dataset (use metadata.csv for CV splits)"
    )
    gen_parser.add_argument(
        "--tickers", required=True, help="Comma-separated ticker symbols"
    )
    gen_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    gen_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    gen_parser.add_argument("--interval", default="1d", help="Data interval (default: 1d)")
    gen_parser.add_argument(
        "--window-size", type=int, default=30, help="Candles per image (default: 30)"
    )
    gen_parser.add_argument(
        "--image-size", type=int, default=224, help="Image size in pixels (default: 224)"
    )
    gen_parser.add_argument(
        "--output", default="./data/images", help="Output directory"
    )
    gen_parser.set_defaults(func=cmd_generate)

    render_parser = subparsers.add_parser("render", help="Render single test image")
    render_parser.add_argument("--ticker", required=True, help="Ticker symbol")
    render_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    render_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    render_parser.add_argument("--interval", default="1d", help="Data interval")
    render_parser.add_argument(
        "--candles", type=int, help="Number of candles (uses last N)"
    )
    render_parser.add_argument(
        "--image-size", type=int, default=224, help="Image size in pixels"
    )
    render_parser.add_argument(
        "--output", default="./test.png", help="Output file path"
    )
    render_parser.set_defaults(func=cmd_render)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
