"""Walk-forward evaluation and calibration report for trained models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_path = Path(args.input)
    frame = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    artifact = joblib.load(args.model)
    features = artifact["features"]
    test = frame[frame["game_date"] > pd.Timestamp(artifact["validation_start"])].copy()
    if test.empty:
        raise ValueError("No holdout rows after validation_start.")
    test["prediction"] = artifact["model"].predict_proba(test[features])[:, 1]
    test["bucket"] = pd.cut(test["prediction"], [0,.01,.02,.03,.04,.05,.07,.10,1], include_lowest=True)
    bucket = test.groupby("bucket", observed=True).agg(calls=("home_run","size"), actual=("home_run","mean"), predicted=("prediction","mean")).reset_index()
    report = {
        "n_test": int(len(test)),
        "brier_score": float(brier_score_loss(test["home_run"], test["prediction"])),
        "log_loss": float(log_loss(test["home_run"], test["prediction"], labels=[0,1])),
        "calibration_buckets": [{"bucket": str(row.bucket), "calls": int(row.calls), "actual": float(row.actual), "predicted": float(row.predicted)} for row in bucket.itertuples()],
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
