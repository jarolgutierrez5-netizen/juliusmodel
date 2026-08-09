"""Pitch-type, velocity, and location matchup scoring."""
from __future__ import annotations

import pandas as pd

LEAGUE_BARREL_PER_PA = 0.050

def _pa_count(frame: pd.DataFrame) -> int:
    if frame.empty or "events" not in frame:
        return 0
    return int(frame.loc[frame["events"].notna(), ["game_pk", "at_bat_number"]].drop_duplicates().shape[0])

def pitch_shape_fit(hitter_frame: pd.DataFrame, pitcher_frame: pd.DataFrame) -> tuple[float, list[str]]:
    """Return a capped multiplier using pitcher mix plus hitter barrel skill by pitch type.

    Later enhancements can add zone, velocity band, spin and movement buckets.
    """
    if hitter_frame.empty or pitcher_frame.empty or "pitch_type" not in hitter_frame or "pitch_type" not in pitcher_frame:
        return 1.0, ["pitch-shape interaction unavailable"]
    mix = pitcher_frame["pitch_type"].dropna().value_counts(normalize=True)
    weighted = 0.0
    for pitch_type, usage in mix.items():
        subset = hitter_frame[hitter_frame["pitch_type"] == pitch_type]
        pa = _pa_count(subset)
        bbe = subset[subset["launch_speed"].notna()] if not subset.empty else subset
        barrels = (bbe.get("launch_speed_angle") == 6).sum() if not bbe.empty else 0
        rate = (barrels + 15 * LEAGUE_BARREL_PER_PA) / (pa + 15)
        weighted += float(usage) * rate
    multiplier = 1 + 0.12 * (weighted / LEAGUE_BARREL_PER_PA - 1)
    top_pitches = ", ".join(mix.head(2).index.tolist())
    return max(0.94, min(1.08, multiplier)), [f"starter mix led by {top_pitches}"]
