"""Automatically settle every pending daily HR projection with official MLB final box scores.

The script is safe to run repeatedly. It scans all daily files whose dates are before
or equal to today, settles only calls that are still pending, and preserves unresolved
calls if a game has not reached final status.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MLB_API = "https://statsapi.mlb.com/api/v1"


def get_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response.json()


def game_batting_totals(target_date: str) -> tuple[dict[str, int], bool]:
    """Return player HR totals and whether every listed game is final or canceled."""
    schedule = get_json(f"{MLB_API}/schedule", {"sportId": 1, "date": target_date})
    games = schedule.get("dates", [{}])[0].get("games", [])
    if not games:
        return {}, True
    totals: dict[str, int] = {}
    all_final = True
    terminal_states = {"Final", "Cancelled", "Postponed"}
    for game in games:
        if game["status"].get("detailedState") not in terminal_states:
            all_final = False
            continue
        if game["status"].get("abstractGameState") != "Final":
            continue
        box = get_json(f"{MLB_API}/game/{game['gamePk']}/boxscore")
        for side in ("away", "home"):
            for player in box["teams"][side].get("players", {}).values():
                player_name = player["person"]["fullName"]
                homers = int(player.get("stats", {}).get("batting", {}).get("homeRuns", 0) or 0)
                totals[player_name] = homers
    return totals, all_final


def rebuild_history() -> None:
    history = []
    for daily_path in sorted((DATA / "daily").glob("*.json")):
        payload = json.loads(daily_path.read_text())
        day = payload.get("date", daily_path.stem)
        for call in payload.get("calls", []):
            history.append({"date": day, **call})
    (DATA / "history.json").write_text(json.dumps(history, indent=2) + "\n")


def main() -> None:
    settled_files = 0
    pending_files = 0
    for daily_path in sorted((DATA / "daily").glob("*.json")):
        target_date = daily_path.stem
        if target_date > date.today().isoformat():
            continue
        payload = json.loads(daily_path.read_text())
        calls = payload.get("calls", [])
        if not any(call.get("result", "pending") == "pending" for call in calls):
            continue
        totals, slate_is_final = game_batting_totals(target_date)
        changed = False
        for call in calls:
            if call.get("result", "pending") != "pending":
                continue
            if call["player"] in totals:
                homers = totals[call["player"]]
                call["home_runs"] = homers
                call["result"] = "hr" if homers > 0 else "no_hr"
                call["settled_from"] = "official_mlb_boxscore"
                changed = True
        if changed:
            payload["calls"] = calls
            daily_path.write_text(json.dumps(payload, indent=2) + "\n")
            settled_files += 1
        elif not slate_is_final:
            pending_files += 1
    rebuild_history()
    print(f"Settled daily files: {settled_files}")
    print(f"Slates still pending: {pending_files}")

if __name__ == "__main__":
    main()
