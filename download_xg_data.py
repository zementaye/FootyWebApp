"""
Downloads historical Understat xG data for the EPL and caches it to
data/epl_xg.csv, matched to this project's match_id.

NOT RUN FROM THE SANDBOX THAT BUILT THIS (no network access there) — run it
yourself:

    pip install understatapi
    python3 download_xg_data.py

Then re-run `python3 src/backtest.py` — it will pick up data/epl_xg.csv
automatically if present and include the xG features; if the file's missing
it falls back to the goals-only feature set with a printed note, so nothing
breaks if you skip this step.

See src/xg_source.py for the important caveats (team-name mapping, 2014-15
Understat coverage start, untested-from-sandbox status).
"""

import os

import pandas as pd

from src.xg_source import fetch_understat_xg, match_to_results

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    results = pd.read_csv(os.path.join(HERE, "data", "epl_results.csv"), parse_dates=["date"])

    # Understat's EPL coverage starts 2014-15. Extend the upper end each year.
    seasons = list(range(2014, 2026))

    print(f"Fetching Understat xG for seasons {seasons[0]}-{seasons[-1]}...")
    xg_raw = fetch_understat_xg(seasons, league="EPL")
    print(f"Fetched {len(xg_raw)} Understat match rows.")

    xg_matched = match_to_results(xg_raw, results)
    print(f"Matched {len(xg_matched)}/{len(xg_raw)} rows to a match_id "
          f"({len(xg_raw) - len(xg_matched)} unmatched — see warning above if any).")

    out_path = os.path.join(HERE, "data", "epl_xg.csv")
    xg_matched.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
