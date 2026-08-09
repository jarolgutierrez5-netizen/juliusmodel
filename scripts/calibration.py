"""Out-of-sample calibration reporting for stored daily projections."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def build_report() -> dict:
    history_path = DATA / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    frame = pd.DataFrame(history)
    if frame.empty:
        return {"settled_calls": 0, "message": "No settled calls yet."}
    settled = frame[frame["result"].isin(["hr", "no_hr"])].copy()
    if settled.empty:
        return {"settled_calls": 0, "message": "No settled calls yet."}
    settled["actual_hr"] = (settled["result"] == "hr").astype(int)
    settled["probability"] = settled["projected_hr_probability"].astype(float)
    brier = float(((settled["probability"] - settled["actual_hr"]) ** 2).mean())
    settled["bucket"] = pd.cut(settled["probability"], bins=[0,.10,.15,.20,.25,1], include_lowest=True)
    buckets = settled.groupby("bucket", observed=True).agg(calls=("actual_hr","size"), actual_hr_rate=("actual_hr","mean"), predicted_hr_rate=("probability","mean")).reset_index()
    return {
        "settled_calls": int(len(settled)),
        "home_runs": int(settled["actual_hr"].sum()),
        "hit_rate": float(settled["actual_hr"].mean()),
        "brier_score": brier,
        "expected_home_runs": float(settled["probability"].sum()),
        "calibration_buckets": [{"bucket": str(row.bucket), "calls": int(row.calls), "actual_hr_rate": float(row.actual_hr_rate), "predicted_hr_rate": float(row.predicted_hr_rate)} for row in buckets.itertuples()],
    }

if __name__ == "__main__":
    report = build_report()
    (DATA / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Calibration report written for {report['settled_calls']} settled calls")
