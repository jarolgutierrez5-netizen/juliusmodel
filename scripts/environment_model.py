"""Park, roof, and weather context with transparent neutral fallbacks."""
from __future__ import annotations

from dataclasses import dataclass

# Validate and refresh these annually from a trusted park-factor provider. Multipliers
# intentionally stay close to neutral to avoid overfitting a single season.
PARK_HR_FACTORS = {
    "Yankee Stadium": {"L": 1.06, "R": 1.01},
    "Great American Ball Park": {"L": 1.05, "R": 1.05},
    "Coors Field": {"L": 1.05, "R": 1.05},
    "Oracle Park": {"L": 0.94, "R": 0.91},
    "T-Mobile Park": {"L": 0.96, "R": 0.96},
    "Petco Park": {"L": 0.96, "R": 0.96},
}

ROOF_VENUES = {
    "Chase Field", "Globe Life Field", "loanDepot park", "American Family Field",
    "T-Mobile Park", "Rogers Centre",
}

@dataclass
class EnvironmentContext:
    multiplier: float
    notes: list[str]
    missing_inputs: list[str]


def environment_context(*, venue: str, bat_side: str, temperature_f: float | None = None,
                        wind_mph: float | None = None, wind_out: bool | None = None,
                        roof_status: str | None = None) -> EnvironmentContext:
    notes: list[str] = []
    missing: list[str] = []
    multiplier = PARK_HR_FACTORS.get(venue, {}).get(bat_side, 1.0)
    if venue not in PARK_HR_FACTORS:
        missing.append("validated handedness-specific park factor")
    else:
        notes.append(f"park factor {multiplier:.2f} for {bat_side}-handed power")

    roof_closed = roof_status and roof_status.lower() in {"closed", "retractable closed"}
    if venue in ROOF_VENUES and roof_status is None:
        missing.append("verified roof status")
    if roof_closed:
        notes.append("roof closed; outdoor weather suppressed")
        return EnvironmentContext(multiplier, notes, missing)

    if temperature_f is None or wind_mph is None or wind_out is None:
        missing.append("game-time temperature and wind")
        return EnvironmentContext(multiplier, notes, missing)
    # Restrained weather adjustment, not a standalone predictor.
    if temperature_f >= 85:
        multiplier *= 1.015
        notes.append("warm-weather lift")
    elif temperature_f <= 52:
        multiplier *= 0.985
        notes.append("cold-weather drag")
    if wind_out and wind_mph >= 10:
        multiplier *= min(1.035, 1 + (wind_mph - 8) * 0.002)
        notes.append("outbound wind lift")
    elif not wind_out and wind_mph >= 10:
        multiplier *= max(0.965, 1 - (wind_mph - 8) * 0.002)
        notes.append("inbound wind drag")
    return EnvironmentContext(multiplier, notes, missing)
