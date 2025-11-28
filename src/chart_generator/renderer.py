"""Candlestick chart rendering using Pillow."""

from dataclasses import dataclass

import pandas as pd
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class RenderConfig:
    """Configuration for candlestick rendering."""

    size: int = 224
    background_color: str = "#FFFFFF"
    bullish_color: str = "#00AA00"  # Green
    bearish_color: str = "#AA0000"  # Red
    body_width_ratio: float = 0.5
    padding_ratio: float = 0.05
    use_rgb: bool = True  # RGB for color, L for grayscale


def render_candlesticks(
    ohlc: pd.DataFrame,
    config: RenderConfig | None = None,
) -> Image.Image:
    """
    Render candlestick chart from OHLC data.

    Args:
        ohlc: DataFrame with columns 'open', 'high', 'low', 'close'
        config: Rendering configuration

    Returns:
        PIL Image with rendered candlesticks
    """
    if config is None:
        config = RenderConfig()

    if ohlc.empty:
        raise ValueError("OHLC data is empty")

    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(ohlc.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    mode = "RGB" if config.use_rgb else "L"
    bg_color = config.background_color if config.use_rgb else 255
    img = Image.new(mode, (config.size, config.size), color=bg_color)
    draw = ImageDraw.Draw(img)

    num_candles = len(ohlc)
    candle_total_width = config.size / num_candles
    body_width = candle_total_width * config.body_width_ratio
    margin = (candle_total_width - body_width) / 2

    price_min = float(ohlc["low"].min())
    price_max = float(ohlc["high"].max())
    price_range = price_max - price_min

    if price_range == 0:
        price_range = 1.0
        price_min -= 0.5
        price_max += 0.5

    padding = price_range * config.padding_ratio
    price_min -= padding
    price_max += padding
    price_range = price_max - price_min

    def price_to_y(price: float) -> int:
        normalized = (price - price_min) / price_range
        return int(config.size * (1 - normalized))

    # Colors for bullish/bearish candles
    if config.use_rgb:
        bullish_color = config.bullish_color
        bearish_color = config.bearish_color
    else:
        bullish_color = 102  # Dark gray
        bearish_color = 0    # Black

    for i, (_, row) in enumerate(ohlc.iterrows()):
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])

        x_center = candle_total_width * i + candle_total_width / 2
        x_left = candle_total_width * i + margin
        x_right = x_left + body_width

        y_high = price_to_y(high_price)
        y_low = price_to_y(low_price)
        y_open = price_to_y(open_price)
        y_close = price_to_y(close_price)

        is_bullish = close_price >= open_price
        color = bullish_color if is_bullish else bearish_color

        draw.line([(x_center, y_high), (x_center, y_low)], fill=color, width=1)

        body_top = min(y_open, y_close)
        body_bottom = max(y_open, y_close)

        if body_bottom - body_top < 1:
            body_bottom = body_top + 1

        draw.rectangle(
            [(x_left, body_top), (x_right, body_bottom)],
            fill=color,
            outline=color,
        )

    return img
