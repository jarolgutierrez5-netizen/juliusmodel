"""Bullpen availability and HR-environment estimates."""
from __future__ import annotations

from dataclasses import dataclass

LEAGUE_HR_PER_PA = 0.030
LEAGUE_BARREL_PER_PA = 0.050

@dataclass
class BullpenContext:
    multiplier: float
    availability_score: float
    notes: list[str]
    missing_inputs: list[str]


def bullpen_multiplier(*, hr_per_pa_allowed: float | None, barrel_per_pa_allowed: float | None,
                       innings_last_3_days: float | None, batter_side: str | None = None) -> BullpenContext:
    """Build a deliberately capped bullpen adjustment.

    Inputs should be handedness-split bullpen rates where possible. A 3-day workload
    proxy lowers availability score and creates a small upward HR adjustment.
    """
    notes: list[str] = []
    missing: list[str] = []
    if hr_per_pa_allowed is None or barrel_per_pa_allowed is None:
        return BullpenContext(1.0, 0.50, notes, ["handedness-split bullpen HR/barrel rates"])
    multiplier = 1 + 0.22 * (hr_per_pa_allowed / LEAGUE_HR_PER_PA - 1) + 0.12 * (barrel_per_pa_allowed / LEAGUE_BARREL_PER_PA - 1)
    availability = 0.70
    if innings_last_3_days is None:
        missing.append("bullpen workload in prior three days")
    elif innings_last_3_days >= 13:
        multiplier += 0.025
        availability = 0.30
        notes.append("heavily worked bullpen")
    elif innings_last_3_days >= 10:
        multiplier += 0.010
        availability = 0.50
        notes.append("moderately worked bullpen")
    else:
        availability = 0.80
    return BullpenContext(max(0.88, min(1.16, multiplier)), availability, notes, missing)
