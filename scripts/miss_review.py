"""Create transparent preliminary miss labels after a settled daily slate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def label_miss(call: dict) -> str | None:
    if call.get("result") != "no_hr":
        return None
    if call.get("expected_pa", 4.0) < 3.8:
        return "opportunity-limited"
    if call.get("context_boost", 0) > 0.015:
        return "favorable-context-did-not-convert"
    if call.get("confidence") == "Medium-low":
        return "high-input-uncertainty"
    return "ordinary-baseball-variance"

def main() -> None:
    review = []
    for path in sorted((DATA / "daily").glob("*.json")):
        payload = json.loads(path.read_text())
        for call in payload.get("calls", []):
            label = label_miss(call)
            if label:
                review.append({"date": payload.get("date", path.stem), "player": call["player"], "projection": call["projected_hr_probability"], "review_label": label, "missing_inputs": call.get("missing_inputs", [])})
    (DATA / "miss_reviews.json").write_text(json.dumps(review, indent=2) + "\n")
    print(f"Wrote {len(review)} preliminary miss reviews")

if __name__ == "__main__":
    main()
