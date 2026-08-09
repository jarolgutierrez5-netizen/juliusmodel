"""Train and evaluate a calibrated home-run probability model.

Training data format
--------------------
A parquet or CSV file with one row per hitter plate appearance. Required columns:

- home_run: 0/1 target
- game_date: ISO date
- player_id
- barrels_per_pa, hard_hit_rate, max_ev, airborne_pull_rate, hr_per_pa,
  recent_contact_trend, fly_ball_rate, strikeout_rate

Optional matchup columns are accepted and automatically included when present:
handedness_advantage, pitcher_hr_per_pa_allowed, pitcher_barrel_per_pa_allowed,
pitch_type_fit, starter_fly_ball_rate, bullpen_hr_per_pa_allowed,
park_factor, weather_factor.

The trainer uses a chronological split and Platt calibration. It never performs a
random train/test split, which would leak future baseball context into validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

CORE_FEATURES = [
    "barrels_per_pa", "hard_hit_rate", "max_ev", "airborne_pull_rate",
    "hr_per_pa", "recent_contact_trend", "fly_ball_rate", "strikeout_rate",
]
OPTIONAL_FEATURES = [
    "handedness_advantage", "pitcher_hr_per_pa_allowed", "pitcher_barrel_per_pa_allowed",
    "pitch_type_fit", "starter_fly_ball_rate", "bullpen_hr_per_pa_allowed",
    "park_factor", "weather_factor",
]


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def chronological_split(frame: pd.DataFrame, validation_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    data = data.dropna(subset=["game_date"]).sort_values("game_date")
    cutoff_idx = int(len(data) * (1 - validation_fraction))
    return data.iloc[:cutoff_idx], data.iloc[cutoff_idx:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Historical PA-level CSV or parquet")
    parser.add_argument("--model-out", default=str(MODELS / "hr_pa_calibrated.joblib"))
    parser.add_argument("--report-out", default=str(ROOT / "data" / "training_report.json"))
    args = parser.parse_args()

    frame = load_frame(Path(args.input))
    if "home_run" not in frame or "game_date" not in frame:
        raise ValueError("Training data requires home_run and game_date columns.")
    features = [column for column in CORE_FEATURES + OPTIONAL_FEATURES if column in frame.columns]
    missing_core = [column for column in CORE_FEATURES if column not in features]
    if missing_core:
        raise ValueError(f"Training data is missing core fields: {missing_core}")
    train, validation = chronological_split(frame)
    if train.empty or validation.empty:
        raise ValueError("Chronological split produced an empty train or validation set.")

    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=2000, class_weight=None)),
    ])
    base.fit(train[features], train["home_run"].astype(int))
    # Platt scaling is fitted only on chronologically later data.
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(validation[features], validation["home_run"].astype(int))
    pred = calibrated.predict_proba(validation[features])[:, 1]
    y = validation["home_run"].astype(int)

    artifact = {
        "model": calibrated,
        "features": features,
        "training_end": str(train["game_date"].max().date()),
        "validation_start": str(validation["game_date"].min().date()),
        "version": "calibrated-logistic-3.0",
    }
    joblib.dump(artifact, args.model_out)
    report = {
        "model_version": artifact["version"], "n_train": int(len(train)), "n_validation": int(len(validation)),
        "training_end": artifact["training_end"], "validation_start": artifact["validation_start"],
        "validation_brier_score": float(brier_score_loss(y, pred)),
        "validation_log_loss": float(log_loss(y, pred, labels=[0, 1])),
        "validation_auc": float(roc_auc_score(y, pred)) if y.nunique() > 1 else None,
        "features": features,
    }
    Path(args.report_out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
