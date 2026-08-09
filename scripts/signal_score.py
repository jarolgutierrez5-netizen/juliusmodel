"""Transparent transparent HR signal HR signal score.

This is an interpretable presentation score, not a calibrated probability. It is a
percentile-style 0–100 ranking built from bounded standardized sub-scores. The daily
model continues to use HR probability as the core ranking metric.
"""
from __future__ import annotations

from typing import Any
import numpy as np

LEAGUE = {
    "barrel_per_pa": 0.050,
    "hard_hit_rate": 0.350,
    "max_ev": 110.0,
    "pulled_air_rate": 0.330,
    "hr_per_pa": 0.030,
    "recent_barrel_delta": 0.0,
    "pitcher_hr_pa": 0.030,
    "pitcher_barrel_pa": 0.050,
}


def bounded_z(value: float | None, mean: float, scale: float, lower: float = -2.5, upper: float = 2.5) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(np.clip((value - mean) / scale, lower, upper))


def _score_0_100(z_value: float) -> float:
    return float(np.clip(50 + 20 * z_value, 0, 100))


def signal_score(*, hitter: dict[str, Any], pitcher: dict[str, Any], context_boost: float,
                 projected_hr_probability: float, expected_pa: float) -> dict[str, Any]:
    barrel = hitter.get("barrel_per_pa")
    hard_hit = hitter.get("hard_hit_rate")
    max_ev = hitter.get("max_ev")
    pull_air = hitter.get("pulled_air_rate")
    hr_pa = hitter.get("hr_per_pa")
    recent = hitter.get("recent_barrel_per_pa")
    recent_delta = (recent - barrel) if recent is not None and barrel is not None else None

    inputs = {
        "barrels_per_pa": _score_0_100(bounded_z(barrel, LEAGUE["barrel_per_pa"], 0.020)),
        "hard_hit_rate": _score_0_100(bounded_z(hard_hit, LEAGUE["hard_hit_rate"], 0.070)),
        "max_exit_velocity": _score_0_100(bounded_z(max_ev, LEAGUE["max_ev"], 3.5)),
        "airborne_pull_rate": _score_0_100(bounded_z(pull_air, LEAGUE["pulled_air_rate"], 0.100)),
        "hr_per_pa": _score_0_100(bounded_z(hr_pa, LEAGUE["hr_per_pa"], 0.015)),
        "recent_contact_trend": _score_0_100(bounded_z(recent_delta, LEAGUE["recent_barrel_delta"], 0.015)),
    }
    # The weights reproduce a likely public-facing HR signal architecture while
    # retaining transparent feature names and controlled influence.
    power_score = (
        0.30 * inputs["barrels_per_pa"] + 0.20 * inputs["hard_hit_rate"] +
        0.15 * inputs["max_exit_velocity"] + 0.15 * inputs["airborne_pull_rate"] +
        0.10 * inputs["hr_per_pa"] + 0.10 * inputs["recent_contact_trend"]
    )
    pitcher_hr = pitcher.get("hr_per_pa_statcast")
    pitcher_barrel = pitcher.get("barrel_per_pa")
    pitcher_score = 0.60 * _score_0_100(bounded_z(pitcher_hr, LEAGUE["pitcher_hr_pa"], 0.012)) + 0.40 * _score_0_100(bounded_z(pitcher_barrel, LEAGUE["pitcher_barrel_pa"], 0.020))
    # Context is intentionally capped: it can refine a profile but never replace power.
    context_score = float(np.clip(50 + 1200 * context_boost + 7 * (expected_pa - 4.1), 20, 80))
    raw_score = 0.62 * power_score + 0.23 * pitcher_score + 0.15 * context_score
    # Probability has only a light role here, preserving a separate signal view.
    raw_score = 0.88 * raw_score + 0.12 * np.clip(100 * projected_hr_probability / 0.25, 0, 100)
    final = int(round(np.clip(raw_score, 0, 100)))

    tags = []
    if barrel is not None and barrel >= 0.065:
        tags.append("elite barrel frequency")
    if max_ev is not None and max_ev >= 113:
        tags.append("plus raw power")
    if pull_air is not None and pull_air >= 0.40:
        tags.append("pulled-air power")
    if recent_delta is not None and recent_delta >= 0.010:
        tags.append("recent contact lift")
    if pitcher_hr is not None and pitcher_hr >= 0.038:
        tags.append("homer-prone starter")
    if context_boost >= 0.012:
        tags.append("positive game context")
    if not tags:
        tags.append("balanced power profile")

    if final >= 75 and len(tags) >= 3:
        tier = "Strong"
    elif final >= 58:
        tier = "Watch"
    elif final >= 45 and context_boost >= 0.008:
        tier = "Long shot"
    else:
        tier = "No call"
    return {"signal_score": final, "signal_tier": tier, "signal_components": {key: round(value, 1) for key, value in inputs.items()}, "signal_tags": tags[:4]}
