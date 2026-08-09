"""Load an optional calibrated PA model and score daily hitter rows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "hr_pa_calibrated.joblib"


def model_is_available(model_path: Path = DEFAULT_MODEL) -> bool:
    return model_path.exists()


def score_probability(features: dict[str, Any], model_path: Path = DEFAULT_MODEL) -> tuple[float | None, dict]:
    if not model_path.exists():
        return None, {"mode": "transparent-prototype", "reason": "No trained artifact found."}
    artifact = joblib.load(model_path)
    columns = artifact["features"]
    frame = pd.DataFrame([{column: features.get(column, np.nan) for column in columns}])
    probability = float(artifact["model"].predict_proba(frame)[0, 1])
    return probability, {"mode": "calibrated-trained-model", "version": artifact.get("version"), "features": columns}
