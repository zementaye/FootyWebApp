"""
Downloads the last 5 completed UEFA Champions League seasons and writes
data/ucl_results.csv in the same schema as data/epl_results.csv (+ two
extra columns: `stage` and `neutral`).

Primary source: github.com/openfootball/champions-league (CC0/public
domain), specifically each season's <season>/cl.txt file in the
"football.txt" structured-text format. This parser was written and
verified against real file content for the 2021-22, 2022-23, 2023-24, and
2024-25 seasons (fetched by hand, not guessed) — including the trickier
notation for extra-time/penalty-shootout legs (e.g.
"1-4 pen. 0-1 a.e.t. (0-1, 0-1)"), the "Gruppe G/H" naming quirk in older
seasons, and year rollover across New Year's without an explicit year on
every date line. It was NOT verified against 2025-26 (that folder may not
exist yet in the source repo if the season only recently finished) — the
script tells you plainly if a season comes back empty rather than
pretending it worked.

Known gap: none of the four hand-verified cl.txt files include qualifying
rounds — only the league/group phase through the final. This script also
tries <season>/quali.txt per season (same repo, same parser — the format
doesn't depend on which stage names appear) as a best-effort second file,
but that filename was NOT verified against real content, so treat its
success/failure log line as the ground truth, not this comment.

Secondary/fallback source: the openfootball/football.json JSON mirror
(<season>/uefa.cl.json), confirmed present for 2024-25 only as of this
writing. Tried only for seasons where the primary text source fails.

NOT RUN FROM THE SANDBOX THAT BUILT THIS — run it yourself:

    pip install requests
    python3 download_ucl_data.py

Then restart the API server (`uvicorn app:app --reload`) — app.py picks up
data/ucl_results.csv automatically on startup and trains a second model for
competition=ucl. Until you've run this once, /api/predict?competition=ucl
and friends return a clear 503 rather than failing to start.

WHY ONLY 5 SEASONS (not full history like EPL): see the long comment on
COMPETITIONS["ucl"]["lookback_seasons"] in src/competitions.py.
"""

import logging
import os
import re
import sys

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_ucl_data")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "data", "ucl_results.csv")

TXT_BASE_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master"
JSON_BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
JSON_CANDIDATE_FILENAMES = ["uefa.cl.json", "cl.json", "uefa.champions-league.json"]

N_SEASONS = 5
LAST_COMPLETED_START_YEAR = 2025  # 2025-26 UCL season; bump once a newer season has finished
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in
           range(LAST_COMPLETED_START_YEAR - N_SEASONS + 1, LAST_COMPLETED_START_YEAR + 1)]

# ---------------------------------------------------------------------------
# stage classification (shared by both parsers) — bucket free-text round
# names into a small fixed vocabulary instead of carrying ~40 raw strings
# across 5 seasons of varying format (pre-2024-25 group stage vs. the
# 2024-25+ single league phase).
# ---------------------------------------------------------------------------
_STAGE_PATTERNS = [
    (r"preliminary", "Qualifying (Preliminary)"),
    (r"first qualifying|1st qualifying|q1\b", "Qualifying (Round 1)"),
    (r"second qualifying|2nd qualifying|q2\b", "Qualifying (Round 2)"),
    (r"third qualifying|3rd qualifying|q3\b", "Qualifying (Round 3)"),
    (r"play-?off", "Play-off round"),
    (r"matchday|group stage|group [a-h]\b|gruppe [a-h]\b|league phase", "League phase / group stage"),
    (r"round of 16|last 16|1/8", "Round of 16"),
    (r"quarter", "Quarter-final"),
    (r"semi", "Semi-final"),
    (r"\bfinal\b", "Final"),
]


def classify_stage(round_text: str) -> str:
    t = (round_text or "").lower()
    for pattern, label in _STAGE_PATTERNS:
        if re.search(pattern, t):
            return label
    return round_text or "Unknown"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


# ---------------------------------------------------------------------------
# Primary parser: football.txt (<season>/cl.txt, <season>/quali.txt)
# ---------------------------------------------------------------------------
_MONTH_ABBR = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
_HEADER_YEAR_RE = re.compile(r"#\s*Date\s+\w{3}\s+(\w{3})\s+(\d{1,2})\s+(\d{4})")
_TITLE_YEAR_RE = re.compile(r"(\d{4})/\d{2}")
_STAGE_HEADER_RE = re.compile(r"^\u25aa\s*(.+?)\s*$")  # "▪ <stage>"
_DATE_LINE_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?$")
_TIME_PREFIX_RE = re.compile(r"^(\d{1,2}:\d{2})\s+(.*)$")
_COUNTRY_CODE_RE = re.compile(r"\s*\([A-Za-z.]{2,5}\)\s*$")
_NUM_PAIR_RE = re.compile(r"(\d+)-(\d+)")


def parse_football_txt(text: str, season: str) -> list:
    """
    Parses openfootball's football.txt match-listing format. Verified against
    real 2021-22 through 2024-25 championsleague/cl.txt content, including:
      - extra-time / penalty-shootout notation, e.g.
        "1-4 pen. 0-1 a.e.t. (0-1, 0-1)" -> this leg's actual goals (including
        extra time, excluding the penalty shootout) are the number pair
        immediately before "a.e.t." — the "pen." pair is a shootout score,
        not goals, and is deliberately NOT used for fthg/ftag.
      - "Gruppe G"/"Gruppe H" (an inconsistency in the source data itself for
        some older seasons, not a bug here) treated the same as "Group X".
      - date lines that omit the year once it's already been stated earlier
        in the file, including the turn of the calendar year (inferred by
        month decreasing without an explicit year).
    """
    season_code = season.replace("-", "")
    lines = text.splitlines()
    rows = []
    current_year = None
    current_month = None
    current_stage_raw = None
    current_date = None

    for line in lines[:10]:
        m = _HEADER_YEAR_RE.search(line)
        if m:
            current_year = int(m.group(3))
            break
    if current_year is None:
        for line in lines[:3]:
            m = _TITLE_YEAR_RE.search(line)
            if m:
                current_year = int(m.group(1))
                break

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        stage_m = _STAGE_HEADER_RE.match(stripped)
        if stage_m:
            current_stage_raw = stage_m.group(1)
            current_date = None
            continue

        date_m = _DATE_LINE_RE.match(stripped)
        if date_m:
            _, mon_abbr, day_str, year_str = date_m.groups()
            month = _MONTH_ABBR.get(mon_abbr)
            day = int(day_str)
            if year_str:
                current_year = int(year_str)
            elif current_month is not None and month is not None and month < current_month:
                current_year = (current_year or 0) + 1
            current_month = month
            current_date = f"{current_year:04d}-{month:02d}-{day:02d}" if (current_year and month) else None
            continue

        if current_date is None or current_stage_raw is None:
            continue  # header/meta text before the first real section — skip safely

        time_m = _TIME_PREFIX_RE.match(stripped)
        rest = time_m.group(2) if time_m else stripped

        if " v " not in rest:
            continue  # not a match line we recognize — skip defensively rather than guess
        team1_part, right = rest.split(" v ", 1)
        parts = re.split(r"\s{2,}", right.strip(), maxsplit=1)
        if len(parts) != 2:
            continue
        team2_part, score_blob = parts
        team1 = _COUNTRY_CODE_RE.sub("", team1_part).strip()
        team2 = _COUNTRY_CODE_RE.sub("", team2_part).strip()
        if not team1 or not team2:
            continue

        all_pairs = _NUM_PAIR_RE.findall(score_blob)
        if not all_pairs:
            continue  # unplayed/postponed fixture line, not a result — skip
        has_pen = "pen." in score_blob
        idx = 1 if has_pen and len(all_pairs) > 1 else 0
        fthg, ftag = (int(x) for x in all_pairs[idx])
        hthg = htag = None
        if len(all_pairs) > idx + 1:
            hthg, htag = (int(x) for x in all_pairs[idx + 1])

        ftr = "H" if fthg > ftag else ("A" if fthg < ftag else "D")
        htr = None
        if hthg is not None and htag is not None:
            htr = "H" if hthg > htag else ("A" if hthg < htag else "D")

        stage = classify_stage(current_stage_raw)
        rows.append({
            "match_id": f"ucl{season_code}-{slugify(team1)}-{slugify(team2)}-{current_date}",
            "season": season, "season_code": season_code, "date": current_date,
            "home_team": team1, "away_team": team2,
            "fthg": fthg, "ftag": ftag, "ftr": ftr,
            "hthg": hthg, "htag": htag, "htr": htr,
            "referee": None, "stage": stage, "neutral": stage == "Final",
        })
    return rows


def fetch_txt(season: str, filename: str):
    url = f"{TXT_BASE_URL}/{season}/{filename}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text
    except requests.RequestException as e:
        logger.debug(f"[{season}] {filename} request failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Fallback parser: football.json mirror (only confirmed present for 2024-25)
# ---------------------------------------------------------------------------
def fetch_and_parse_json(season: str) -> list:
    for filename in JSON_CANDIDATE_FILENAMES:
        url = f"{JSON_BASE_URL}/{season}/{filename}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200 or not resp.text.strip().startswith("{"):
                continue
            data = resp.json()
            matches = data.get("matches") or []
            if not matches:
                continue
            logger.info(f"[{season}] JSON fallback: fetched {len(matches)} entries from {filename}")
        except (requests.RequestException, ValueError):
            continue

        rows = []
        season_code = season.replace("-", "")
        for m in matches:
            score = m.get("score", {}) or {}
            ft = score.get("ft")
            if not ft or len(ft) != 2:
                continue
            hthg, htag = (score.get("ht") or [None, None])
            home, away = m["team1"], m["team2"]
            stage = classify_stage(m.get("round", ""))
            fthg, ftag = ft
            ftr = "H" if fthg > ftag else ("A" if fthg < ftag else "D")
            htr = None
            if hthg is not None and htag is not None:
                htr = "H" if hthg > htag else ("A" if hthg < htag else "D")
            rows.append({
                "match_id": f"ucl{season_code}-{slugify(home)}-{slugify(away)}-{m.get('date')}",
                "season": season, "season_code": season_code, "date": m.get("date"),
                "home_team": home, "away_team": away,
                "fthg": fthg, "ftag": ftag, "ftr": ftr,
                "hthg": hthg, "htag": htag, "htr": htr,
                "referee": None, "stage": stage, "neutral": stage == "Final",
            })
        return rows
    return []


# ---------------------------------------------------------------------------
def fetch_season(season: str) -> list:
    rows = []

    main_text = fetch_txt(season, "cl.txt")
    if main_text:
        main_rows = parse_football_txt(main_text, season)
        logger.info(f"[{season}] cl.txt: parsed {len(main_rows)} matches "
                    f"(stages: {sorted(set(r['stage'] for r in main_rows))})")
        rows.extend(main_rows)
    else:
        logger.warning(f"[{season}] {TXT_BASE_URL}/{season}/cl.txt not reachable")

    quali_text = fetch_txt(season, "quali.txt")
    if quali_text:
        quali_rows = parse_football_txt(quali_text, season)
        logger.info(f"[{season}] quali.txt: parsed {len(quali_rows)} qualifying/play-off matches")
        rows.extend(quali_rows)
    else:
        logger.info(f"[{season}] no quali.txt found at that path (qualifiers may live under a "
                    f"different filename in the source repo, or may not be tracked separately "
                    f"for this season) — {season} will be missing qualifying-round matches")

    if not rows:
        json_rows = fetch_and_parse_json(season)
        if json_rows:
            logger.info(f"[{season}] falling back to JSON mirror: {len(json_rows)} matches")
            rows.extend(json_rows)

    if not rows:
        logger.warning(f"[{season}] no data found from any source — skipping this season entirely")
    return rows


def main():
    all_rows = []
    for season in SEASONS:
        all_rows.extend(fetch_season(season))

    if not all_rows:
        logger.error("No seasons fetched successfully — data/ucl_results.csv NOT written.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    logger.info(f"Wrote {len(df)} matches across {df['season'].nunique()} season(s) to {OUT_PATH}")
    logger.info("Seasons covered: " + ", ".join(sorted(df["season"].unique())))
    logger.info("Stage breakdown:\n" + df["stage"].value_counts().to_string())


if __name__ == "__main__":
    main()
