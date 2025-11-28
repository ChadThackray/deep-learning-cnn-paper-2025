# Candlestick CNN

Generate candlestick chart images from Yahoo Finance data and classify them with a CNN.

## Installation

```bash
uv sync
```

## Usage

### Generate Dataset

```bash
uv run chart-generator generate \
  --tickers AAPL,MSFT,GOOGL \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --window-size 30 \
  --image-size 50 \
  --output ./data/images
```

**Options:**
- `--tickers` - Comma-separated ticker symbols
- `--start` - Start date (YYYY-MM-DD)
- `--end` - End date (YYYY-MM-DD)
- `--interval` - Data interval: 1d, 1h, 5m, etc. (default: 1d)
- `--window-size` - Number of candles per image (default: 30)
- `--image-size` - Image dimensions in pixels (default: 224)
- `--output` - Output directory (default: ./data/images)

**Output structure:**
```
data/images/
├── up/
│   └── AAPL_2024-01-15_w30_s50_42.png
├── down/
│   └── ...
└── metadata.csv
```

### Render Single Image (Testing)

```bash
uv run chart-generator render \
  --ticker AAPL \
  --start 2024-01-01 \
  --end 2024-03-01 \
  --candles 30 \
  --image-size 50 \
  --output test.png
```

### Example Experiments

```bash
# Replicate paper settings: 20 candles, 50x50 images
uv run chart-generator generate \
  --tickers AAPL \
  --start 2015-01-01 \
  --end 2024-01-01 \
  --window-size 20 \
  --image-size 50 \
  --output ./data/w20_s50

# Larger context: 50 candles, 50x50 images
uv run chart-generator generate \
  --tickers AAPL,MSFT,GOOGL,AMZN,META \
  --start 2015-01-01 \
  --end 2024-01-01 \
  --window-size 50 \
  --image-size 50 \
  --output ./data/w50_s50
```

## Metadata

Each dataset includes a `metadata.csv` with columns:
- `ticker` - Stock symbol
- `end_date` - Last candle date in the window
- `label` - "up" or "down" (next candle green/red)
- `window_size` - Number of candles in image
- `image_size` - Image dimensions
- `filename` - Image filename

Use this for implementing purged k-fold CV during training.

## Labels

- **up**: Next candle is green (close >= open)
- **down**: Next candle is red (close < open)

---

## CNN Classifier

### Train Model

```bash
uv run cnn-classifier train \
  --data ./data/images \
  --train-cutoff 2016-12-31 \
  --epochs 50 \
  --batch-size 32 \
  --image-size 50 \
  --output ./models/model.pt
```

**Options:**
- `--data` - Path to data directory with metadata.csv
- `--train-cutoff` - End date for training data (YYYY-MM-DD)
- `--val-cutoff` - End date for validation data (optional)
- `--epochs` - Number of epochs (default: 50)
- `--batch-size` - Batch size (default: 32)
- `--lr` - Learning rate (default: 0.001)
- `--image-size` - Image size (default: 50)
- `--patience` - Early stopping patience (optional)
- `--output` - Output model path (default: ./models/model.pt)

### Evaluate Model

```bash
uv run cnn-classifier evaluate \
  --model ./models/model.pt \
  --data ./data/images \
  --test-start 2017-01-01 \
  --image-size 50
```

**Metrics reported:**
- Accuracy
- Sensitivity (Recall)
- Specificity
- MCC (Matthews Correlation Coefficient)

### Predict Single Image

```bash
uv run cnn-classifier predict \
  --model ./models/model.pt \
  --image ./test.png \
  --image-size 50
```

### Example Workflow

```bash
# 1. Generate dataset with temporal range for train/test split
uv run chart-generator generate \
  --tickers AAPL,MSFT,GOOGL \
  --start 2010-01-01 \
  --end 2020-01-01 \
  --window-size 20 \
  --image-size 50 \
  --output ./data/images

# 2. Train on data up to 2016, test on 2017+
uv run cnn-classifier train \
  --data ./data/images \
  --train-cutoff 2016-12-31 \
  --epochs 50 \
  --image-size 50 \
  --output ./models/model.pt

# 3. Evaluate on test set
uv run cnn-classifier evaluate \
  --model ./models/model.pt \
  --data ./data/images \
  --test-start 2017-01-01 \
  --image-size 50
```
