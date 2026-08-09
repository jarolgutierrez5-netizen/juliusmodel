"""Daily matchup-adjusted MLB home-run projection model.

Design goals
------------
- Predict home-run probability per plate appearance, then convert to game probability.
- Use partial pooling for limited MLB samples.
- Elevate under-the-radar hitters only when small-sample results are supported by
  MLB contact quality and/or a player-profile prior.
- Separate neutral talent from today-specific starter, park, weather, lineup, and
  substitution context.

Data sources
------------
- MLB Stats API: schedule, confirmed batting orders, probable starters, season PA/HR.
- Baseball Savant Statcast search: hitter and pitcher batted-ball / pitch-mix inputs.

The job remains conservative when a source is unavailable: an unavailable component
gets neutral weight and is logged in the output's `missing_inputs` field.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from bullpen_model import bullpen_multiplier
from environment_model import environment_context
from minor_league_prior import make_prior
from pitch_shape_matchups import pitch_shape_fit
from playing_time_model import expected_pa as playing_time_expected_pa, starter_exposure_share
from model_scoring import score_probability

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TODAY = date.today().isoformat()
MLB_API = "https://statsapi.mlb.com/api/v1"
SAVANT_SEARCH = "https://baseballsavant.mlb.com/statcast_search/csv"

# Calibrated baseline priors. These are intentionally conservative until the model
# has enough stored outcomes for formal calibration.
LEAGUE_HR_PER_PA = 0.030
LEAGUE_BARREL_PER_PA = 0.050
HR_PRIOR_PA = 100.0
STARTER_PA_SHARE = 0.65
MAX_CALLS = int(os.getenv("MAX_CALLS", "15"))

# Handedness-specific park multipliers. Keep this neutral by default; populate only
# after validating a current park-factor source. The model logs missing park input.
PARK_FACTORS: dict[str, dict[str, float]] = {}


def get_json(url: str, params: dict[str, Any] | None = None) -> dict:
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response.json()


def get_statcast(start_date: str, end_date: str, *, batter_id: int | None = None, pitcher_id: int | None = None) -> pd.DataFrame:
    """Download pitch-level Statcast data for one hitter or pitcher."""
    params = {
        "all": "true", "type": "batter", "player_type": "batter",
        "game_date_gt": start_date, "game_date_lt": end_date,
        "hfGT": "R%7C", "player_id": batter_id or pitcher_id,
    }
    if pitcher_id is not None:
        params["type"] = "pitcher"
        params["player_type"] = "pitcher"
    try:
        response = requests.get(SAVANT_SEARCH, params=params, timeout=90)
        response.raise_for_status()
        return pd.read_csv(pd.io.common.StringIO(response.text), low_memory=False)
    except Exception:
        return pd.DataFrame()


def pa_count(statcast: pd.DataFrame) -> int:
    if statcast.empty or "events" not in statcast:
        return 0
    return int(statcast.loc[statcast["events"].notna(), ["game_pk", "at_bat_number"]].drop_duplicates().shape[0])


def batted_ball_features(statcast: pd.DataFrame, recent_start: str) -> dict[str, float | None]:
    """Extract barrel, EV, fly ball and recent-contact features from Statcast."""
    if statcast.empty or "launch_speed" not in statcast:
        return {key: None for key in [
            "sc_pa", "barrel_per_pa", "barrel_rate", "avg_ev", "max_ev", "hard_hit_rate",
            "fly_ball_rate", "recent_barrel_per_pa", "hr_per_pa_statcast", "pulled_air_rate"
        ]}
    bbe = statcast.loc[statcast["launch_speed"].notna()].copy()
    total_pa = pa_count(statcast)
    if bbe.empty or total_pa == 0:
        return {key: None for key in [
            "sc_pa", "barrel_per_pa", "barrel_rate", "avg_ev", "max_ev", "hard_hit_rate",
            "fly_ball_rate", "recent_barrel_per_pa", "hr_per_pa_statcast", "pulled_air_rate"
        ]}
    barrels = (bbe.get("launch_speed_angle") == 6).sum()
    air = bbe["bb_type"].isin(["fly_ball", "line_drive", "popup"])
    # Statcast attack_direction is retained when available. Positive values generally
    # represent pull-side contact for the batter. Unknown direction is simply omitted.
    if "attack_direction" in bbe and air.any():
        pulled_air_rate = (bbe.loc[air, "attack_direction"] > 0).mean()
    else:
        pulled_air_rate = None
    recent = statcast.loc[statcast["game_date"].astype(str) >= recent_start]
    recent_pa = pa_count(recent)
    recent_bbe = recent.loc[recent["launch_speed"].notna()] if not recent.empty else recent
    return {
        "sc_pa": float(total_pa),
        "barrel_per_pa": float(barrels / total_pa),
        "barrel_rate": float(barrels / len(bbe)),
        "avg_ev": float(bbe["launch_speed"].mean()),
        "max_ev": float(bbe["launch_speed"].max()),
        "hard_hit_rate": float((bbe["launch_speed"] >= 95).mean()),
        "fly_ball_rate": float(bbe["bb_type"].isin(["fly_ball", "popup"]).mean()),
        "recent_barrel_per_pa": float((recent_bbe.get("launch_speed_angle") == 6).sum() / recent_pa) if recent_pa else None,
        "hr_per_pa_statcast": float((statcast["events"] == "home_run").sum() / total_pa),
        "pulled_air_rate": float(pulled_air_rate) if pulled_air_rate is not None else None,
    }


def pitch_mix(statcast: pd.DataFrame) -> dict[str, float]:
    if statcast.empty or "pitch_type" not in statcast:
        return {}
    values = statcast["pitch_type"].dropna().value_counts(normalize=True)
    return {str(key): float(value) for key, value in values.items()}


def hitter_pitch_type_fit(hitter_statcast: pd.DataFrame, pitcher_mix: dict[str, float]) -> float | None:
    """Shrink hitter barrel production by pitch type and weight it by pitcher mix."""
    if hitter_statcast.empty or not pitcher_mix or "pitch_type" not in hitter_statcast:
        return None
    base = LEAGUE_BARREL_PER_PA
    total = 0.0
    used = 0.0
    for pitch_type, usage in pitcher_mix.items():
        subset = hitter_statcast[hitter_statcast["pitch_type"] == pitch_type]
        n_pa = pa_count(subset)
        if n_pa < 5:
            value = base
        else:
            bbe = subset[subset["launch_speed"].notna()]
            barrels = (bbe.get("launch_speed_angle") == 6).sum()
            value = (barrels + 15 * base) / (n_pa + 15)
        total += usage * value
        used += usage
    return float(total / used) if used else None



def profile_prior(pa: int, hr: int, features: dict[str, float | None], profile_hr_prior: float = LEAGUE_HR_PER_PA, profile_prior_pa: float = HR_PRIOR_PA) -> float:
    """Partially pooled neutral p(HR)/PA with contact-quality support."""
    shrunk_hr = (hr + profile_prior_pa * profile_hr_prior) / (pa + profile_prior_pa)
    barrel_pa = features.get("barrel_per_pa")
    barrel_component = (barrel_pa if barrel_pa is not None else LEAGUE_BARREL_PER_PA) * 0.55
    # Contact quality can move the profile only modestly; HR totals retain majority weight.
    neutral = 0.70 * shrunk_hr + 0.30 * barrel_component
    recent_barrels = features.get("recent_barrel_per_pa")
    if recent_barrels is not None and barrel_pa is not None:
        recent_factor = float(np.clip(1 + (recent_barrels - barrel_pa) / 0.08, 0.92, 1.08))
        neutral *= recent_factor
    return float(np.clip(neutral, 0.006, 0.120))


def starter_context_multiplier(pitcher_features: dict[str, float | None], pitch_fit: float | None) -> float:
    """Starter-only adjustment. Bullpen remains neutral until a bullpen data source is added."""
    pitcher_hr = pitcher_features.get("hr_per_pa_statcast")
    pitcher_barrels = pitcher_features.get("barrel_per_pa")
    if pitcher_hr is None or pitcher_barrels is None:
        return 1.0
    multiplier = 1 + 0.25 * (pitcher_hr / LEAGUE_HR_PER_PA - 1) + 0.15 * (pitcher_barrels / LEAGUE_BARREL_PER_PA - 1)
    if pitch_fit is not None:
        multiplier += 0.12 * (pitch_fit / LEAGUE_BARREL_PER_PA - 1)
    return float(np.clip(multiplier, 0.82, 1.22))



def confidence_label(pa: int, statcast_pa: float | None, missing_count: int) -> str:
    if pa >= 300 and (statcast_pa or 0) >= 200 and missing_count <= 2:
        return "High"
    if pa >= 100 and (statcast_pa or 0) >= 75:
        return "Medium"
    return "Medium-low"


def classify_under_the_radar(pa: int, batting_order: int, game_prob: float, neutral_prob: float) -> str:
    # Low-PA / low-lineup-visibility candidates need real modeled upside to qualify.
    if game_prob >= 0.135 and (pa < 350 or batting_order >= 6) and neutral_prob >= 0.030:
        return "Under-the-radar"
    return "Established"


def serialize_number(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    end_date = (date.today() - timedelta(days=1)).isoformat()
    season_start = f"{date.today().year}-03-20"
    recent_start = (date.today() - timedelta(days=30)).isoformat()
    schedule = get_json(f"{MLB_API}/schedule", {"sportId": 1, "date": TODAY, "hydrate": "probablePitcher,venue"})
    games = schedule.get("dates", [{}])[0].get("games", [])

    lineup_rows: list[dict[str, Any]] = []
    game_context: dict[int, dict[str, Any]] = {}
    game_labels: dict[int, str] = {}
    for game in games:
        if game["status"].get("detailedState") not in {"Scheduled", "Pre-Game", "Warmup"}:
            continue
        game_pk = game["gamePk"]
        box = get_json(f"{MLB_API}/game/{game_pk}/boxscore")
        game_context[game_pk] = {
            "venue": game.get("venue", {}).get("name", "Unknown venue"),
            "away_pitcher": game["teams"]["away"].get("probablePitcher", {}),
            "home_pitcher": game["teams"]["home"].get("probablePitcher", {}),
        }
        game_labels[game_pk] = f"{game['teams']['away']['team']['name']} at {game['teams']['home']['team']['name']}"
        for side in ("away", "home"):
            team = box["teams"][side]
            opponent_side = "home" if side == "away" else "away"
            opponent = box["teams"][opponent_side]["team"]["name"]
            opponent_pitcher = game_context[game_pk][f"{opponent_side}_pitcher"]
            for player in team.get("players", {}).values():
                order = player.get("battingOrder", "")
                if not order:
                    continue
                lineup_rows.append({
                    "game_pk": game_pk,
                    "venue": game_context[game_pk]["venue"],
                    "player_id": int(player["person"]["id"]),
                    "player": player["person"]["fullName"],
                    "team": team["team"]["name"],
                    "opponent": opponent,
                    "batting_order": int(order) // 100,
                    "opponent_pitcher_id": opponent_pitcher.get("id"),
                    "opponent_pitcher": opponent_pitcher.get("fullName"),
                })
    if not lineup_rows:
        print("No confirmed remaining batting orders. No daily file written.")
        return

    people_ids = ",".join(str(row["player_id"]) for row in lineup_rows)
    people = get_json(f"{MLB_API}/people", {"personIds": people_ids, "hydrate": "stats(group=[hitting],type=[season],sportId=1)"})
    hitter_stats: dict[int, dict[str, Any]] = {}
    for person in people.get("people", []):
        splits = person.get("stats", [{}])[0].get("splits", [])
        stat = splits[0].get("stat", {}) if splits else {}
        hitter_stats[person["id"]] = {
            "pa": int(stat.get("plateAppearances", 0)),
            "hr": int(stat.get("homeRuns", 0)),
            "bat_side": person.get("batSide", {}).get("code", "R"),
        }

    # First pass prevents expensive Statcast calls on players with no baseline chance.
    base_rows = []
    for row in lineup_rows:
        stats = hitter_stats.get(row["player_id"], {"pa": 0, "hr": 0, "bat_side": "R"})
        if stats["pa"] < 15:
            continue
        raw_p = (stats["hr"] + HR_PRIOR_PA * LEAGUE_HR_PER_PA) / (stats["pa"] + HR_PRIOR_PA)
        raw_game = 1 - (1 - raw_p) ** playing_time_expected_pa(batting_order=row["batting_order"], confirmed_lineup=True, is_platoon=False, catcher=False, rookie_pa=stats["pa"])[0]
        base_rows.append({**row, **stats, "raw_game": raw_game})
    # Score every confirmed lineup hitter so each game receives a useful ranked card.
    # MAX_CALLS controls only the slate-wide summary, not the per-game detail.
    base_rows = sorted(base_rows, key=lambda record: record["raw_game"], reverse=True)

    hitter_cache: dict[int, pd.DataFrame] = {}
    pitcher_cache: dict[int, pd.DataFrame] = {}
    calls: list[dict[str, Any]] = []
    for row in base_rows:
        missing: list[str] = []
        hitter_id = row["player_id"]
        if hitter_id not in hitter_cache:
            hitter_cache[hitter_id] = get_statcast(season_start, end_date, batter_id=hitter_id)
        hitter_frame = hitter_cache[hitter_id]
        hitter_features = batted_ball_features(hitter_frame, recent_start)
        pitcher_features = {key: None for key in ["hr_per_pa_statcast", "barrel_per_pa"]}
        pfit = None
        pitcher_id = row.get("opponent_pitcher_id")
        if pitcher_id:
            if pitcher_id not in pitcher_cache:
                pitcher_cache[pitcher_id] = get_statcast(season_start, end_date, pitcher_id=pitcher_id)
            pitcher_frame = pitcher_cache[pitcher_id]
            pitcher_features = batted_ball_features(pitcher_frame, recent_start)
            pfit = hitter_pitch_type_fit(hitter_frame, pitch_mix(pitcher_frame))
            if pitcher_frame.empty:
                missing.append("starter Statcast profile")
        else:
            missing.append("confirmed opposing starter")

        player_prior = make_prior(mlb_pa=row["pa"], max_ev=hitter_features.get("max_ev"), barrel_per_pa=hitter_features.get("barrel_per_pa"))
        missing.extend(player_prior.missing_inputs)
        prototype_per_pa = profile_prior(row["pa"], row["hr"], hitter_features, player_prior.hr_per_pa_prior, player_prior.prior_weight_pa)
        trained_feature_row = {
            "barrels_per_pa": hitter_features.get("barrel_per_pa"),
            "hard_hit_rate": hitter_features.get("hard_hit_rate"),
            "max_ev": hitter_features.get("max_ev"),
            "airborne_pull_rate": hitter_features.get("pulled_air_rate"),
            "hr_per_pa": row["hr"] / max(row["pa"], 1),
            "recent_contact_trend": (hitter_features.get("recent_barrel_per_pa") or prototype_per_pa) - (hitter_features.get("barrel_per_pa") or LEAGUE_BARREL_PER_PA),
            "fly_ball_rate": hitter_features.get("fly_ball_rate"),
            "strikeout_rate": None,
            "handedness_advantage": None,
            "pitcher_hr_per_pa_allowed": pitcher_features.get("hr_per_pa_statcast"),
            "pitcher_barrel_per_pa_allowed": pitcher_features.get("barrel_per_pa"),
            "pitch_type_fit": pfit,
            "starter_fly_ball_rate": pitcher_features.get("fly_ball_rate"),
            "bullpen_hr_per_pa_allowed": None,
            "park_factor": None,
            "weather_factor": None,
        }
        trained_per_pa, model_status = score_probability(trained_feature_row)
        neutral_per_pa = trained_per_pa if trained_per_pa is not None else prototype_per_pa
        if trained_per_pa is None:
            missing.append("trained historical model artifact")
        starter_mult = starter_context_multiplier(pitcher_features, pfit)
        shape_mult, shape_notes = pitch_shape_fit(hitter_frame, pitcher_frame) if pitcher_id and not pitcher_frame.empty else (1.0, ["pitch-shape interaction unavailable"])
        bullpen = bullpen_multiplier(hr_per_pa_allowed=None, barrel_per_pa_allowed=None, innings_last_3_days=None, batter_side=row["bat_side"])
        missing.extend(bullpen.missing_inputs)
        env = environment_context(venue=row["venue"], bat_side=row["bat_side"])
        missing.extend(env.missing_inputs)
        projected_pa, playing_time = playing_time_expected_pa(batting_order=row["batting_order"], confirmed_lineup=True, is_platoon=False, catcher=False, rookie_pa=row["pa"])
        starter_share = starter_exposure_share(projected_innings=5.85, workload_known=False)
        # Starter and bullpen are separately weighted; all unavailable inputs remain neutral.
        context_multiplier = (1 + starter_share * (starter_mult * shape_mult - 1)) * (1 + (1 - starter_share) * (bullpen.multiplier - 1)) * env.multiplier
        matchup_per_pa = float(np.clip(neutral_per_pa * context_multiplier, 0.004, 0.140))
        neutral_game = 1 - (1 - neutral_per_pa) ** projected_pa
        game_prob = 1 - (1 - matchup_per_pa) ** projected_pa
        classification = classify_under_the_radar(row["pa"], row["batting_order"], game_prob, neutral_per_pa)

        signals = [f"{row['hr']} HR in {row['pa']} PA", f"Batting order {row['batting_order']}"]
        if hitter_features.get("barrel_per_pa") is not None:
            signals.append(f"{hitter_features['barrel_per_pa']:.1%} barrels/PA")
        if hitter_features.get("max_ev") is not None:
            signals.append(f"{hitter_features['max_ev']:.1f} mph max EV")
        if pitcher_features.get("barrel_per_pa") is not None:
            signals.append(f"Starter allows {pitcher_features['barrel_per_pa']:.1%} barrels/PA")
        signals = signals[:3]
        risk = "Limited expected PA" if row["batting_order"] >= 7 else "Matchup data remains partly neutral"
        if row["pa"] < 100:
            risk = "Small MLB sample is heavily shrunk toward the player-profile prior"
        if "starter Statcast profile" in missing:
            risk = "Starter-specific contact context unavailable; pitcher adjustment is neutral"

        calls.append({
            "game_pk": row["game_pk"], "game": game_labels.get(row["game_pk"], f"{row['team']} vs {row['opponent']}") , "player": row["player"], "team": row["team"], "opponent": row["opponent"], "venue": row["venue"],
            "opposing_starter": row.get("opponent_pitcher"), "batting_order": row["batting_order"], "expected_pa": round(projected_pa, 2),
            "classification": classification, "projected_hr_probability": round(float(game_prob), 4),
            "neutral_hr_probability": round(float(neutral_game), 4), "context_boost": round(float(game_prob - neutral_game), 4),
            "hr_per_pa": round(float(matchup_per_pa), 4), "model_status": model_status, "prototype_hr_per_pa": round(float(prototype_per_pa), 4), "confidence": confidence_label(row["pa"], hitter_features.get("sc_pa"), len(missing)),
            "signals": (signals + player_prior.notes + shape_notes + env.notes + bullpen.notes)[:5], "primary_risk": risk,
            "playing_time": playing_time, "starter_exposure_share": round(starter_share, 3),
            "missing_inputs": sorted(set(missing)),
            "features": {key: serialize_number(value) for key, value in hitter_features.items()},
            "pitcher_features": {key: serialize_number(value) for key, value in pitcher_features.items() if key in {"hr_per_pa_statcast", "barrel_per_pa", "fly_ball_rate"}},
            "result": "pending",
        })

    calls.sort(key=lambda item: item["projected_hr_probability"], reverse=True)
    # Ensure the final output is short and includes overlooked players when supported.
    under = [call for call in calls if call["classification"] == "Under-the-radar"]
    established = [call for call in calls if call["classification"] == "Established"]
    # Keep a diversified view: the top overall profiles plus a meaningful under-the-radar
    # cohort. The default is 15, overridable with MAX_CALLS in GitHub Actions.
    established_slots = max(1, round(MAX_CALLS * 0.60))
    under_slots = max(1, MAX_CALLS - established_slots)
    final_calls = established[:established_slots] + under[:under_slots]
    if len(final_calls) < MAX_CALLS:
        selected_names = {call["player"] for call in final_calls}
        remaining = [call for call in calls if call["player"] not in selected_names]
        final_calls.extend(remaining[:MAX_CALLS - len(final_calls)])
    final_calls.sort(key=lambda item: item["projected_hr_probability"], reverse=True)

    calls_by_game: dict[str, dict[str, Any]] = {}
    for call in calls:
        game_id = str(call["game_pk"])
        if game_id not in calls_by_game:
            calls_by_game[game_id] = {"game_pk": call["game_pk"], "game": call["game"], "venue": call["venue"], "players": []}
        calls_by_game[game_id]["players"].append(call)
    for game_block in calls_by_game.values():
        ranked_players = sorted(game_block["players"], key=lambda item: item["projected_hr_probability"], reverse=True)
        qualifying = [
            item for item in ranked_players
            if item["classification"] == "Under-the-radar"
            and item["projected_hr_probability"] >= 0.12
            and len(item.get("signals", [])) >= 2
            and "confirmed opposing starter" not in item.get("missing_inputs", [])
        ]
        primary = qualifying[0] if qualifying else None
        secondary = None
        tertiary = None
        if primary:
            for item in qualifying[1:]:
                close_to_primary = item["projected_hr_probability"] >= 0.85 * primary["projected_hr_probability"]
                strong_context = item["projected_hr_probability"] >= 0.11 and item["context_boost"] >= 0.012
                if close_to_primary or strong_context:
                    secondary = item
                    break
            if secondary:
                for item in qualifying[2:]:
                    exceptionally_close = item["projected_hr_probability"] >= 0.92 * secondary["projected_hr_probability"]
                    elite_context = item["projected_hr_probability"] >= 0.12 and item["context_boost"] >= 0.020
                    if exceptionally_close or elite_context:
                        tertiary = item
                        break
        game_block["players"] = [item for item in [primary, secondary, tertiary] if item is not None]
        game_block["game_favorite"] = ranked_players[0]["player"] if ranked_players else None
        game_block["under_the_radar"] = primary["player"] if primary else None
        game_block["secondary_under_the_radar"] = secondary["player"] if secondary else None
        game_block["tertiary_under_the_radar"] = tertiary["player"] if tertiary else None
        game_block["context_riser"] = max(ranked_players, key=lambda item: item["context_boost"])["player"] if ranked_players else None
        game_block["status"] = "qualifying call" if primary else "pass"
        game_block["pass_reason"] = None if primary else "No non-star hitter cleared the 12% probability and support-signal threshold."

    output = {
        "date": TODAY,
        "model_version": "matchup-adjusted-statcast-3.0",
        "method": "Partial-pooling HR/PA model with hitter contact quality, starter Statcast, pitch-mix fit, expected PA, and transparent neutral fallbacks.",
        "calls": final_calls,
        "games": sorted(calls_by_game.values(), key=lambda item: item["game"]),
        "notes": [
            "No sportsbook lines, betting odds, or public sentiment are used.",
            "Players with limited MLB data are partially pooled using a 100-PA HR-rate prior and Statcast contact-quality blend.",
            "Each call includes missing inputs rather than silently substituting stale or guessed data."
        ],
    }
    daily_path = DATA / "daily" / f"{TODAY}.json"
    daily_path.write_text(json.dumps(output, indent=2) + "\n")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    if TODAY not in manifest:
        manifest.append(TODAY)
    manifest_path.write_text(json.dumps(sorted(manifest), indent=2) + "\n")
    print(f"Wrote {daily_path.relative_to(ROOT)} with {len(final_calls)} matchup-adjusted calls")

if __name__ == "__main__":
    main()
