"""Build a leakage-safe historical hitter-game HR training dataset.

The builder downloads historical Statcast in monthly chunks, creates one row per
hitter-game, and constructs rolling pregame features only from prior games.

Default scope is 2023–2025. For a fast smoke test, supply --start and --end over a
shorter period. Full three-season construction can take substantial time and network
bandwidth, so GitHub Actions should run it only manually or on a dedicated runner.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "data" / "training"
RAW_DIR = TRAINING_DIR / "raw_statcast"
SAVANT = "https://baseballsavant.mlb.com/statcast_search/csv"


def daterange_months(start: date, end: date):
    cursor = start.replace(day=1)
    while cursor <= end:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_end = min(end, next_month - timedelta(days=1))
        yield max(start, cursor), chunk_end
        cursor = next_month


def fetch_statcast_chunk(start: date, end: date, cache: bool = True) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = RAW_DIR / f"statcast_{start.isoformat()}_{end.isoformat()}.csv"
    if cache and filename.exists():
        return pd.read_csv(filename, low_memory=False)
    params = {"all": "true", "type": "batter", "player_type": "batter", "game_date_gt": (start - timedelta(days=1)).isoformat(), "game_date_lt": (end + timedelta(days=1)).isoformat(), "hfGT": "R%7C"}
    response = requests.get(SAVANT, params=params, timeout=180)
    response.raise_for_status()
    filename.write_bytes(response.content)
    return pd.read_csv(filename, low_memory=False)


def unique_pa(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["game_date"] = pd.to_datetime(data["game_date"])
    final = data[data["events"].notna()].copy()
    final = final.drop_duplicates(["game_pk", "batter", "at_bat_number"], keep="last")
    return final


def bbe_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["game_date"] = pd.to_datetime(data["game_date"])
    return data[data["launch_speed"].notna()].drop_duplicates(["game_pk", "batter", "at_bat_number"], keep="last").copy()


def make_game_rows(raw: pd.DataFrame) -> pd.DataFrame:
    pa = unique_pa(raw)
    bbe = bbe_rows(raw)
    if pa.empty:
        return pd.DataFrame()
    pa["home_run"] = (pa["events"] == "home_run").astype(int)
    pa["strikeout"] = pa["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    pa["batter_hand"] = pa.get("stand", "R")
    game_base = pa.groupby(["game_date", "game_pk", "batter"], as_index=False).agg(
        home_run=("home_run", "max"), pa=("events", "size"), season_hr=("home_run", "sum"), strikeout_rate_game=("strikeout", "mean"),
        pitcher_id=("pitcher", "first"), batter_hand=("batter_hand", "first"), pitcher_hand=("p_throws", "first"),
        team=("bat_score", "first"),
    )
    if bbe.empty:
        return game_base
    bbe["barrel"] = (bbe.get("launch_speed_angle") == 6).astype(int)
    bbe["hard_hit"] = (bbe["launch_speed"] >= 95).astype(int)
    bbe["airborne"] = bbe["bb_type"].isin(["fly_ball", "line_drive", "popup"])
    bbe["fly_ball"] = bbe["bb_type"].isin(["fly_ball", "popup"])
    if "attack_direction" in bbe:
        bbe["pulled_air"] = ((bbe["attack_direction"] > 0) & bbe["airborne"]).astype(int)
    else:
        bbe["pulled_air"] = np.nan
    batted = bbe.groupby(["game_date", "game_pk", "batter"], as_index=False).agg(
        barrels=("barrel", "sum"), bbe=("barrel", "size"), hard_hit_rate_game=("hard_hit", "mean"),
        max_ev_game=("launch_speed", "max"), fly_ball_rate_game=("fly_ball", "mean"), pulled_air_rate_game=("pulled_air", "mean"),
    )
    return game_base.merge(batted, on=["game_date", "game_pk", "batter"], how="left")


def add_pregame_features(game_rows: pd.DataFrame) -> pd.DataFrame:
    """All rolling features are shifted one game, ensuring no same-game leakage."""
    data = game_rows.sort_values(["batter", "game_date", "game_pk"]).copy()
    for column, fallback in [("barrels", 0), ("bbe", 0), ("hard_hit_rate_game", np.nan), ("max_ev_game", np.nan),
                             ("fly_ball_rate_game", np.nan), ("pulled_air_rate_game", np.nan)]:
        if column not in data:
            data[column] = fallback
    grouped = data.groupby("batter", group_keys=False)
    data["prior_pa"] = grouped["pa"].transform(lambda s: s.shift().expanding().sum())
    data["prior_hr"] = grouped["home_run"].transform(lambda s: s.shift().expanding().sum())
    data["prior_barrels"] = grouped["barrels"].transform(lambda s: s.shift().expanding().sum())
    data["prior_bbe"] = grouped["bbe"].transform(lambda s: s.shift().expanding().sum())
    data["prior_k"] = grouped["strikeout_rate_game"].transform(lambda s: s.shift().expanding().mean())
    data["barrels_per_pa"] = (data["prior_barrels"].fillna(0) + 5.0) / (data["prior_pa"].fillna(0) + 100.0)
    data["hr_per_pa"] = (data["prior_hr"].fillna(0) + 3.0) / (data["prior_pa"].fillna(0) + 100.0)
    data["hard_hit_rate"] = grouped["hard_hit_rate_game"].transform(lambda s: s.shift().rolling(30, min_periods=5).mean())
    data["max_ev"] = grouped["max_ev_game"].transform(lambda s: s.shift().rolling(30, min_periods=3).max())
    data["airborne_pull_rate"] = grouped["pulled_air_rate_game"].transform(lambda s: s.shift().rolling(30, min_periods=5).mean())
    data["fly_ball_rate"] = grouped["fly_ball_rate_game"].transform(lambda s: s.shift().rolling(30, min_periods=5).mean())
    data["strikeout_rate"] = data["prior_k"]
    recent_barrels = grouped["barrels"].transform(lambda s: s.shift().rolling(10, min_periods=3).sum())
    recent_pa = grouped["pa"].transform(lambda s: s.shift().rolling(10, min_periods=3).sum())
    data["recent_contact_trend"] = recent_barrels / recent_pa - data["barrels_per_pa"]
    data["handedness_advantage"] = (data["batter_hand"] != data["pitcher_hand"]).astype(float)
    # Initial dataset does not yet use ex-post pitcher/park/weather features. These are
    # neutral placeholders so the trainer can use a stable schema without leakage.
    for column, value in {
        "pitcher_hr_per_pa_allowed": 0.030, "pitcher_barrel_per_pa_allowed": 0.050,
        "pitch_type_fit": 0.050, "starter_fly_ball_rate": 0.35,
        "bullpen_hr_per_pa_allowed": 0.030, "park_factor": 1.0, "weather_factor": 1.0,
    }.items():
        data[column] = value
    data = data.rename(columns={"batter": "player_id"})
    feature_columns = [
        "game_date", "game_pk", "player_id", "pitcher_id", "home_run", "pa", "batter_hand", "pitcher_hand",
        "barrels_per_pa", "hard_hit_rate", "max_ev", "airborne_pull_rate", "hr_per_pa", "recent_contact_trend",
        "fly_ball_rate", "strikeout_rate", "handedness_advantage", "pitcher_hr_per_pa_allowed",
        "pitcher_barrel_per_pa_allowed", "pitch_type_fit", "starter_fly_ball_rate", "bullpen_hr_per_pa_allowed",
        "park_factor", "weather_factor",
    ]
    return data[feature_columns].replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-03-28")
    parser.add_argument("--end", default="2026-08-08")
    parser.add_argument("--output", default=str(TRAINING_DIR / "hitter_game_features.parquet"))
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    chunks = []
    if end >= date.today():
        raise ValueError("Historical training end date must be before today to avoid future/incomplete data.")
    for chunk_start, chunk_end in tqdm(list(daterange_months(start, end)), desc="Downloading Statcast months"):
        try:
            raw = fetch_statcast_chunk(chunk_start, chunk_end)
            if not raw.empty:
                chunks.append(make_game_rows(raw))
            time.sleep(0.5)
        except Exception as exc:
            print(f"Skipped {chunk_start} to {chunk_end}: {exc}")
    if not chunks:
        raise RuntimeError("No historical Statcast chunks were available.")
    game_rows = pd.concat(chunks, ignore_index=True)
    features = add_pregame_features(game_rows)
    # Drop early rows with too little historical context. Shrunk rate fields remain valid.
    warmup_days = 30 if (end - start).days >= 45 else 0
    features = features[features["game_date"] >= pd.Timestamp(start + timedelta(days=warmup_days))].copy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    schema = {"rows": int(len(features)), "start": str(features["game_date"].min().date()), "end": str(features["game_date"].max().date()), "columns": list(features.columns)}
    output.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    print(json.dumps(schema, indent=2))

if __name__ == "__main__":
    main()
