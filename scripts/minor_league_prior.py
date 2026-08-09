"""Comparable-player / minor-league power prior for limited MLB samples."""
from __future__ import annotations

from dataclasses import dataclass

LEAGUE_HR_PER_PA = 0.030

@dataclass
class MinorLeaguePrior:
    hr_per_pa_prior: float
    prior_weight_pa: float
    notes: list[str]
    missing_inputs: list[str]


def make_prior(*, mlb_pa: int, milb_pa: int | None = None, milb_hr: int | None = None,
               milb_k_rate: float | None = None, max_ev: float | None = None,
               barrel_per_pa: float | None = None, age: int | None = None) -> MinorLeaguePrior:
    """Return an interpretable player-profile prior, capped to prevent overreaction."""
    notes: list[str] = []
    missing: list[str] = []
    prior = LEAGUE_HR_PER_PA
    weight = 100.0
    if milb_pa and milb_hr is not None and milb_pa >= 100:
        milb_rate = milb_hr / milb_pa
        prior = 0.65 * prior + 0.35 * min(0.075, max(0.012, milb_rate))
        weight += 25
        notes.append("minor-league HR/PA included")
    else:
        missing.append("minor-league HR and PA")
    if max_ev is not None:
        if max_ev >= 114:
            prior += 0.006
            notes.append("elite max-EV power prior")
        elif max_ev >= 110:
            prior += 0.003
    else:
        missing.append("maximum exit velocity")
    if barrel_per_pa is not None and barrel_per_pa >= 0.075:
        prior += 0.004
        notes.append("strong early barrel support")
    if milb_k_rate is not None and milb_k_rate >= 0.30:
        prior -= 0.003
    if age is not None and age <= 24 and mlb_pa < 100:
        weight += 15
        notes.append("young-player partial pooling")
    return MinorLeaguePrior(max(0.012, min(0.065, prior)), weight, notes, missing)
