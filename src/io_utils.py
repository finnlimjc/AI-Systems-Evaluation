from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR, MODEL_ORDER

_MODEL_NAME_LOOKUP = {model.lower(): model for model in MODEL_ORDER}


def normalise_model_name(name:str) -> str:
    """Map any casing of a Whisper model name to its canonical name in MODEL_ORDER."""
    canonical = _MODEL_NAME_LOOKUP.get(name.lower(), name)
    return canonical


def save_results(df:pd.DataFrame, filename:str) -> Path:
    """Write a DataFrame to RESULTS_DIR as CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def load_results(filename:str) -> pd.DataFrame:
    """Read a CSV from RESULTS_DIR, mapping model names to the canonical convention."""
    df = pd.read_csv(RESULTS_DIR / filename)
    for column in ("Model", "System"):
        if column in df.columns:
            df[column] = df[column].map(normalise_model_name)
    return df
