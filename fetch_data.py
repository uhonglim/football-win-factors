"""Fetch 2022/23 match-level team statistics from ESPN's public JSON API.

Competitions: Premier League (eng.1), La Liga (esp.1), Champions League (uefa.champions).
Output: data/matches_2223.csv — one row per team per match (each match gives two rows).

Usage:  python fetch_data.py
"""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}"
LEAGUES = {
    "eng.1": ("Premier League", date(2022, 8, 1), date(2023, 5, 31)),
    "esp.1": ("La Liga", date(2022, 8, 10), date(2023, 6, 5)),
    "uefa.champions": ("Champions League", date(2022, 9, 5), date(2023, 6, 11)),
}
# UCL qualifying rounds (Jul/Aug) are excluded: we start at the group stage.

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (research script)"


def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def event_ids(lg, start, end):
    days = [start + timedelta(d) for d in range((end - start).days + 1)]

    def one(d):
        js = get(BASE.format(lg=lg) + "/scoreboard", {"dates": d.strftime("%Y%m%d")})
        out = []
        for e in (js or {}).get("events", []):
            if e["status"]["type"].get("completed"):
                out.append((e["id"], e["date"][:10], e.get("season", {}).get("slug", "")))
        return out

    with ThreadPoolExecutor(8) as ex:
        res = [x for r in ex.map(one, days) for x in r]
    return sorted(set(res))


def to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def match_rows(lg, comp, ev):
    eid, day, stage = ev
    js = get(BASE.format(lg=lg) + "/summary", {"event": eid})
    if not js:
        return []
    comps = js["header"]["competitions"][0]
    side = {c["team"]["id"]: c for c in comps["competitors"]}
    teams = js.get("boxscore", {}).get("teams", [])
    if len(teams) != 2:
        return []
    rows = []
    for t in teams:
        tid = t["team"]["id"]
        c = side[tid]
        row = {
            "competition": comp, "event_id": eid, "date": day, "stage": stage,
            "team": t["team"]["displayName"], "home": int(c["homeAway"] == "home"),
            "goals": to_num(c.get("score")),
            "shootout_winner": int(bool(c.get("winner"))) if c.get("shootoutScore") is not None else None,
        }
        for s in t["statistics"]:
            row[s["name"]] = to_num(s["displayValue"])
        rows.append(row)
    a, b = rows
    for x, y in ((a, b), (b, a)):
        x["opponent"] = y["team"]
        x["goals_against"] = y["goals"]
    return rows


def main():
    all_rows = []
    for lg, (comp, start, end) in LEAGUES.items():
        evs = event_ids(lg, start, end)
        print(f"{comp}: {len(evs)} matches")
        with ThreadPoolExecutor(8) as ex:
            for rows in ex.map(lambda e: match_rows(lg, comp, e), evs):
                all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv("data/matches_2223.csv", index=False)
    print(df.groupby("competition").size() // 2, "matches saved;", df.shape)


if __name__ == "__main__":
    main()
