# Deploying the Football Prediction Site

This is a self-contained FastAPI app: it trains the model in-process on startup
(~30-45s), scrapes upcoming fixtures + odds every 6 hours, and serves a
dashboard + JSON API. Tested locally end-to-end (model training, prediction
endpoint, backtest endpoint, static frontend all confirmed working). The one
thing I could NOT test from this sandbox is the live fixture scraper —
football-data.co.uk blocks the sandbox's network, but the scraper is written
against their documented CSV schema and will work from a normal cloud host.
Test it yourself first with `python3 src/scraper.py` before deploying.

## Option A: Render.com (easiest, has a free tier)

1. Push this `webapp/` folder to a GitHub repo.
2. On Render.com: New → Web Service → connect your repo.
3. Render will detect `render.yaml` and `Dockerfile` automatically. Click Deploy.
4. First build takes a few minutes (installing xgboost etc.). First request
   after any idle period on the free tier will be slow (~30-60s) since Render's
   free tier spins containers down — model retrains in-process on cold start.
5. Your site will be live at `https://football-predictions-XXXX.onrender.com`.

## Option B: Railway.app

1. Push to GitHub, then on Railway: New Project → Deploy from GitHub repo.
2. Railway auto-detects the `Dockerfile`. No extra config needed.
3. Add a custom domain or use the generated `*.up.railway.app` URL.

## Option C: Fly.io

```bash
brew install flyctl   # or see fly.io/docs/getting-started
cd webapp
fly launch            # follow prompts, it'll detect the Dockerfile
fly deploy
```

## Option D: Any VPS (DigitalOcean, Linode, a home server, etc.)

```bash
docker build -t football-predictions .
docker run -d -p 8000:8000 --restart unless-stopped football-predictions
```
Then put a reverse proxy (nginx/Caddy) in front for HTTPS + a real domain.

## Before you deploy: things worth doing first

1. **Run the scraper standalone** (`python3 src/scraper.py`) from your own
   machine to confirm football-data.co.uk's fixtures.csv format hasn't changed
   and odds columns are populating as expected for upcoming matches.
2. **Decide how you'll refresh training data.** Right now the bundled
   `epl_results.csv`/`epl_odds.csv` are a snapshot — the model won't learn from
   new results unless you periodically re-download and swap in updated data
   (see `data/download_data.py` in the parent project) and restart the service,
   or extend `model_service.retrain()` to be called on a schedule after
   appending newly completed matches.
3. **The `value_bet_flag` logic excludes draws by default**, based on the
   backtest finding that draw bets consistently lost money. You can change
   that in `src/model_service.py` (`predict_match`) if you want to test
   otherwise.
4. **This is not a profitable system** — see `data/backtest_summary.json` /
   the dashboard's own honesty banner. Don't wire this to real staking without
   understanding that a walk-forward backtest on real odds came out negative.
   Treat it as a probability/analytics tool, not an auto-profit machine.

## What changed in this update

- **New feature: rest days / fixture congestion** (`src/features.py::add_rest_days`,
  wired into both `src/backtest.py` and `src/model_service.py`). Walk-forward
  safe, same leakage-free pattern as the existing form features. **Caveat**:
  this dataset is EPL-only, so it only sees gaps between a team's *EPL*
  fixtures — it can't see midweek Champions/Europa League or domestic cup
  matches, so it understates congestion for teams playing in Europe. A real
  improvement here would mean adding a multi-competition fixture calendar as
  its own data source.
- **`src/backtest.py` is now runnable standalone**: `python3 src/backtest.py`
  from this folder loads `data/epl_results.csv` + `data/epl_odds.csv`, runs
  the walk-forward backtest, writes `data/backtest_summary.json`, and appends
  a row to `data/backtest_runs_log.jsonl`. Previously this module had no
  entry point in this bundle — it was only invoked from the separate parent
  project. Requires `xgboost` installed (see `requirements.txt`); this
  sandbox couldn't install it to actually re-run the backtest, so
  `data/backtest_summary.json` in this bundle is still the previous run's
  numbers — **re-run it yourself before trusting the new fields.**
- **CLV (closing-line value) tracking added to the backtest.** Each bet now
  records `closing_odds` and `clv_pct` (positive = you beat the closing
  price), using Pinnacle's close as the reference — the sharpest commonly-
  available benchmark. The summary reports `avg_clv_pct` and
  `pct_bets_beating_close`. This matters because ROI on a finite sample is
  noisy; consistently beating the close is a lower-variance signal that
  something real, not luck, is driving results.
- **Bootstrap confidence intervals on ROI** (`bootstrap_roi_ci` in
  `src/backtest.py`), block-bootstrapped by season (not by individual bet) to
  respect the fact that bets within a season share one fitted model and
  aren't independent. `flat_stake_roi_ci95` is now in the summary, overall
  and per selection (H/D/A). A -8% point estimate with a CI that comfortably
  contains 0 is a different claim than one that doesn't — check the interval,
  not just the point estimate.
- **`log_backtest_run` appends every run's config + headline results to
  `data/backtest_runs_log.jsonl`.** This exists to guard against
  reporting only the best-looking configuration after trying several
  (edge thresholds, feature sets, staking rules) — diff new results against
  this log before trusting them.
- **xG (expected goals) features, sourced from Understat** (`src/xg_source.py`,
  `download_xg_data.py`, wired into `src/features.py::add_rolling_xg_form`).
  Adds rolling xG-for form and a "finishing overperformance" signal
  (goals scored minus xG, over the last 5 matches — persistent overperformance
  is a known finishing-quality/luck proxy that tends to partially mean-revert).
  **This is opt-in and untested against the live Understat site** — this
  sandbox had no network access, same limitation as the existing
  football-data.co.uk scraper. Before it does anything:
  1. `pip install understatapi`
  2. `python3 download_xg_data.py` — this also logs how many Understat rows
     it couldn't confidently match to a `match_id`; check that count isn't
     high before trusting the feature (it usually means a team-name mapping
     in `src/xg_source.py::TEAM_NAME_MAP` needs fixing — it was built from
     documented naming conventions, not verified against a live pull).
  3. Re-run `python3 src/backtest.py` — it auto-detects `data/epl_xg.csv`
     and includes the xG features if present, and falls back cleanly to the
     goals-only feature set (today's behavior) if you skip this. Same
     auto-detection applies to the live app (`model_service.py`) — no config
     flag to flip, it just picks up the file if it's there.
  Understat's EPL coverage starts with the 2014-15 season, so roughly the
  first 20 seasons of this project's 1993-2026 history will have no xG match
  — by design, those rows get a neutral fill plus a `has_xg_data=0` flag
  rather than being dropped from training, so adding this doesn't shrink your
  usable history.

- **UEFA Champions League added as a second competition** (`src/competitions.py`,
  `download_ucl_data.py`, generalized `src/model_service.py`/`app.py`).
  Everything that was hardcoded "EPL" now reads from a competition registry;
  adding UCL required no changes to the feature engineering, Elo, Dixon-Coles,
  or ML layers — they were already competition-agnostic. What's new:
  1. `download_ucl_data.py` — same "run it yourself, network needed" pattern
     as `download_xg_data.py`. Pulls the last 5 completed UCL seasons from
     the openfootball project's public-domain `champions-league` repo
     (`<season>/cl.txt`, the "football.txt" structured-text format) and
     writes `data/ucl_results.csv` in the same schema as `epl_results.csv`
     plus `stage` (league phase / round of 16 / etc.) and `neutral` (true
     only for the final, which is played at a fixed neutral venue). Unlike
     the xG/fixtures scripts, this parser **was verified against real file
     content** for 2021-22 through 2024-25 (hand-fetched, not guessed) —
     including the extra-time/penalty-shootout notation, a "Gruppe G/H"
     naming quirk in older seasons' data, and correctly excluding penalty-
     shootout goals from `fthg`/`ftag` (those aren't real match goals). A
     dry run against the full 2024-25 file matched its own header count
     exactly: 189/189 matches, stage totals lining up with the header's own
     breakdown.
     **Known gap, found the same way (not guessed): none of the four
     verified season files include qualifying rounds** — only league/group
     phase through the final. The script also tries `<season>/quali.txt`
     per season (same parser) as a best-effort second file, but that
     filename was *not* verified, so trust its own success/failure log line
     over this sentence. Practically: expect ~125-189 matches/season
     (league phase + knockouts) reliably, and qualifiers/play-off round only
     if `quali.txt` happens to exist at that path.
  2. **UCL trains on only its last 5 seasons, not full history** — this is a
     deliberate default, not a data-availability shortcut: the competition
     format changed for 2024-25 (36-team league phase replacing the 32-team
     group stage), and qualifying-round fields reshuffle most years. See the
     comment on `lookback_seasons` in `src/competitions.py` for the full
     reasoning. EPL is unaffected and still trains on its full 1993-2026
     history.
  3. `elo.py` gained optional neutral-venue support (zeroes the home-advantage
     bump for rows flagged `neutral=True`) so the UCL final's Elo update isn't
     skewed by a home-advantage constant that doesn't apply there. The
     Dixon-Coles layer does **not** yet do the same (it fits one global home-
     advantage constant) — a known, documented gap, not a silent one.
  4. Query param `competition=epl|ucl` on `/api/predict`, `/api/fixtures`,
     `/api/refresh-fixtures`; `/api/health` now reports readiness per
     competition. **Run `download_ucl_data.py` and restart before UCL
     endpoints work** — until then they return a clear 503, the app doesn't
     crash, and EPL keeps working exactly as before.
  5. Two things this did *not* add, called out rather than faked: a
     historical UCL odds source (so `/api/backtest` stays EPL-only and UCL
     predictions never get a `value_bet_flag`), and a live UCL fixtures feed
     (football-data.co.uk, `src/scraper.py`'s source, only covers domestic
     leagues — `/api/fixtures?competition=ucl` says so plainly rather than
     silently returning nothing). Single-match predictions
     (`/api/predict?competition=ucl`) aren't affected by either gap.
  The 2025-26 season specifically was **not** verified (that folder may not
  exist yet in the source repo depending on when you run this) — the script
  logs plainly if a season comes back empty rather than pretending it
  worked; check the logged per-season match counts before trusting
  `data/ucl_results.csv`.

## Environment variables / config

None required to run — everything is bundled (data files, model training
happens at startup). If you want to point at a different odds book, edit
`ODDS_HOME_CANDIDATES` etc. in `src/scraper.py`.

## Scaling notes

- Single `uvicorn` worker is intentional: the model is trained once and held
  in memory. If you scale to multiple workers/instances, each will retrain
  its own copy on startup — wasteful (extra ~30s cold start each) but not
  incorrect, since predictions are read-only after training.
- For a `retrain()` triggered across multiple instances, you'd want to persist
  the trained model artifact (pickle) to shared storage (S3, etc.) rather than
  each instance retraining independently.
