"""Playing-time, starter-workload, and substitution-risk components."""
from __future__ import annotations

import math


def lineup_expected_pa(slot: int) -> float:
    """Baseline PA by confirmed lineup slot, before start and replacement risk."""
    if slot == 1:
        return 4.55
    if slot <= 4:
        return 4.35
    if slot <= 6:
        return 4.05
    return 3.80


def starter_exposure_share(projected_innings: float, workload_known: bool) -> float:
    """Expected portion of hitter PA against the starter, capped conservatively."""
    if not workload_known:
        return 0.65
    return max(0.42, min(0.80, projected_innings / 9.0))


def start_probability(*, confirmed_lineup: bool, platoon_disadvantage: bool = False, catcher: bool = False) -> float:
    if confirmed_lineup:
        return 1.0
    baseline = 0.84
    if platoon_disadvantage:
        baseline -= 0.12
    if catcher:
        baseline -= 0.08
    return max(0.45, baseline)


def substitution_risk(*, batting_order: int, is_platoon: bool = False, catcher: bool = False, rookie_pa: int = 999) -> float:
    """Probability of losing late-game PA to a pinch hitter / replacement."""
    risk = 0.03
    if batting_order >= 7:
        risk += 0.04
    if is_platoon:
        risk += 0.07
    if catcher:
        risk += 0.05
    if rookie_pa < 100:
        risk += 0.04
    return min(risk, 0.28)


def expected_pa(*, batting_order: int, confirmed_lineup: bool, is_platoon: bool, catcher: bool, rookie_pa: int) -> tuple[float, dict]:
    start_prob = start_probability(confirmed_lineup=confirmed_lineup, platoon_disadvantage=is_platoon, catcher=catcher)
    replacement_risk = substitution_risk(batting_order=batting_order, is_platoon=is_platoon, catcher=catcher, rookie_pa=rookie_pa)
    baseline = lineup_expected_pa(batting_order)
    projected = baseline * start_prob * (1 - 0.45 * replacement_risk)
    return projected, {"start_probability": start_prob, "substitution_risk": replacement_risk, "lineup_pa_baseline": baseline}
