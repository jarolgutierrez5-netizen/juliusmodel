"""Settle pending HR calls from the official MLB final box score."""
from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MLB = "https://statsapi.mlb.com/api/v1"


def get_json(url: str) -> dict:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    target = (date.today() - timedelta(days=1)).isoformat()
    daily_path = DATA / "daily" / f"{target}.json"
    if not daily_path.exists():
        print(f"No daily file for {target}; nothing to settle.")
        return
    payload = json.loads(daily_path.read_text())
    schedule = get_json(f"{MLB}/schedule?sportId=1&date={target}&hydrate=linescore,boxscore")
    totals = {}
    for game in schedule.get("dates", [{}])[0].get("games", []):
        if game["status"].get("abstractGameState") != "Final":
            continue
        box = get_json(f"{MLB}/game/{game['gamePk']}/boxscore")
        for side in ("away", "home"):
            for player in box["teams"][side].get("players", {}).values():
                name = player["person"]["fullName"]
                totals[name] = player.get("stats", {}).get("batting", {}).get("homeRuns", 0)
    for call in payload.get("calls", []):
        if call.get("result") == "pending" and call["player"] in totals:
            call["home_runs"] = totals[call["player"]]
            call["result"] = "hr" if totals[call["player"]] > 0 else "no_hr"
    daily_path.write_text(json.dumps(payload, indent=2) + "\n")
    history_path = DATA / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history = [row for row in history if row.get("date") != target]
    history.extend({"date": target, **call} for call in payload.get("calls", []))
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Settled {target}")

if __name__ == "__main__":
    main()
