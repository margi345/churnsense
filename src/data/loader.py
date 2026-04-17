import pandas as pd
import yaml
from pathlib import Path

# ── Load config ────────────────────────────────────────────────────────────────
def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Load raw CSV ───────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    config = load_config()
    raw_path = Path(__file__).resolve().parents[2] / config["data"]["raw_path"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}\n"
            f"Please download from Kaggle and place it in data/raw/"
        )

    df = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── Load cleaned interim data ──────────────────────────────────────────────────
def load_interim() -> pd.DataFrame:
    config = load_config()
    interim_path = Path(__file__).resolve().parents[2] / config["data"]["interim_path"]

    if not interim_path.exists():
        raise FileNotFoundError(
            f"Interim data not found at {interim_path}\n"
            f"Please run the cleaner first: from src.data.cleaner import clean"
        )

    df = pd.read_parquet(interim_path)
    print(f"Loaded interim data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── Load processed feature data ────────────────────────────────────────────────
def load_processed() -> pd.DataFrame:
    config = load_config()
    processed_path = Path(__file__).resolve().parents[2] / config["data"]["processed_path"]

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}\n"
            f"Please run feature engineering first: from src.data.features import build_features"
        )

    df = pd.read_parquet(processed_path)
    print(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df