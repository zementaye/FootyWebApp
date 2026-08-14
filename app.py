"""
FastAPI backend for the football prediction site.

Endpoints:
  GET  /api/health               liveness check
  GET  /api/predict?home=&away=  single match prediction (+ value bet flag if odds given)
  GET  /api/fixtures             upcoming fixtures (scraped) with predictions + value bets
  GET  /api/backtest             precomputed walk-forward backtest summary
  POST /api/refresh-fixtures     force a re-scrape (also runs on a schedule)
  GET  /                         static dashboard (static/index.html)

Run locally:
  uvicorn app:app --reload --port 8000

See README_DEPLOY.md for deploying this to Render/Railway/Fly.io.
"""

import logging
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from src.model_service import ModelService
from src import scraper
from src.competitions import COMPETITIONS, DEFAULT_COMPETITION, get_competition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Per-competition state now, keyed by competition key (see src/competitions.py).
# "models" only ends up containing keys whose results CSV actually exists on
# disk — a competition added to the registry without data yet (e.g. UCL before
# you've run download_ucl_data.py) is skipped with a clear startup log line
# instead of crashing the whole app.
STATE = {"models": {}, "fixtures_cache": {}, "fixtures_error": {}}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BACKTEST_SUMMARY_PATH = os.path.join(DATA_DIR, "backtest_summary.json")


def _get_model(competition: str) -> ModelService:
    key, _ = get_competition(competition)
    model = STATE["models"].get(key)
    if model is None:
        available = ", ".join(STATE["models"]) or "(none trained)"
        raise HTTPException(
            503,
            f"No trained model for competition '{key}'. Available: {available}. "
            + ("Run download_ucl_data.py and restart the server to enable it."
               if key == "ucl" else "")
        )
    return model


def refresh_fixtures_job(competition: str = "epl"):
    key, cfg = get_competition(competition)
    league_codes = cfg["fixtures_league_codes"]
    if not league_codes:
        STATE["fixtures_cache"][key] = []
        STATE["fixtures_error"][key] = None
        logger.info(f"[{key}] No live-fixtures source configured for this competition — skipping scrape "
                    f"(see fixtures_league_codes in src/competitions.py).")
        return
    try:
        raw_fixtures = scraper.fetch_upcoming_fixtures(leagues=league_codes)
        model = STATE["models"].get(key)
        enriched = []
        for fx in raw_fixtures:
            try:
                pred = model.predict_match(
                    fx["home_team"], fx["away_team"],
                    odds_home=fx.get("odds_home"), odds_draw=fx.get("odds_draw"), odds_away=fx.get("odds_away"),
                    match_date=fx.get("date"),
                )
                pred["league"] = fx["league"]
                pred["date"] = fx["date"]
                pred["time"] = fx.get("time")
                enriched.append(pred)
            except Exception as e:
                logger.warning(f"[{key}] Skipping fixture {fx.get('home_team')} vs {fx.get('away_team')}: {e}")
        STATE["fixtures_cache"][key] = enriched
        STATE["fixtures_error"][key] = None
        logger.info(f"[{key}] Refreshed {len(enriched)} fixtures")
    except Exception as e:
        logger.error(f"[{key}] Fixture scrape failed: {e}")
        STATE["fixtures_error"][key] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for key, cfg in COMPETITIONS.items():
        if not os.path.exists(cfg["results_path"]):
            logger.warning(
                f"[{key}] Skipping — no results file at {cfg['results_path']}. "
                + ("Run `python3 download_ucl_data.py` first." if key == "ucl" else "")
            )
            continue
        logger.info(f"[{key}] Starting up: training model...")
        STATE["models"][key] = ModelService(
            cfg["results_path"], cfg["odds_path"], cfg["xg_path"],
            home_advantage=cfg["home_advantage"], lookback_seasons=cfg["lookback_seasons"],
            label=cfg["label"],
        )
        logger.info(f"[{key}] Model ready. Attempting initial fixture scrape...")
        refresh_fixtures_job(key)

    scheduler = BackgroundScheduler()
    for key in STATE["models"]:
        scheduler.add_job(refresh_fixtures_job, "interval", hours=6, args=[key], id=f"refresh_{key}")
    scheduler.start()
    STATE["scheduler"] = scheduler

    yield

    scheduler.shutdown()


app = FastAPI(title="Football Prediction API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "competitions": {
            key: {"label": cfg["label"], "model_ready": key in STATE["models"] and STATE["models"][key].ready}
            for key, cfg in COMPETITIONS.items()
        },
    }


@app.get("/api/predict")
def predict(home: str = Query(...), away: str = Query(...),
            odds_home: float = Query(None), odds_draw: float = Query(None), odds_away: float = Query(None),
            competition: str = Query(DEFAULT_COMPETITION, description="epl or ucl — see /api/health")):
    model = _get_model(competition)
    try:
        return model.predict_match(home, away, odds_home, odds_draw, odds_away)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/fixtures")
def fixtures(competition: str = Query(DEFAULT_COMPETITION, description="epl or ucl — see /api/health")):
    key, cfg = get_competition(competition)
    cache = STATE["fixtures_cache"].get(key, [])
    error = STATE["fixtures_error"].get(key)

    empty_reason = None
    if not cfg["fixtures_league_codes"]:
        empty_reason = (
            f"No live-fixtures source is wired up for {cfg['label']} — football-data.co.uk (the "
            f"source used for EPL) only covers domestic top-flight leagues, not UEFA competitions. "
            f"/api/predict?competition={key} still works for any two team names you give it."
        )
    elif not cache and not error:
        empty_reason = (
            "Scrape succeeded but returned no matches for the selected league(s). "
            "This is expected during the off-season or the days before a season kicks "
            "off (football-data.co.uk only lists a short rolling window of upcoming, "
            "priced fixtures). Check back closer to matchday."
        )
    return {
        "competition": key,
        "fixtures": cache,
        "scrape_error": error,
        "empty_reason": empty_reason,
        "note": "Odds/predictions refresh every 6 hours. Draws are excluded from value_bet_flag "
                "based on backtest findings that draw bets lost money consistently.",
    }


@app.post("/api/refresh-fixtures")
def refresh_fixtures(competition: str = Query(DEFAULT_COMPETITION)):
    key, _ = get_competition(competition)
    if key not in STATE["models"]:
        raise HTTPException(503, f"No trained model for competition '{key}'.")
    refresh_fixtures_job(key)
    return {"competition": key, "refreshed": len(STATE["fixtures_cache"].get(key, [])),
            "error": STATE["fixtures_error"].get(key)}


@app.get("/api/backtest")
def backtest_summary():
    # Precomputed backtest is EPL-only for now — the same walk-forward harness
    # in src/backtest.py can be pointed at data/ucl_results.csv, but its odds
    # merge step assumes epl_odds.csv's bookmaker columns, which don't exist
    # for UCL (see competitions.py: odds_path=None for "ucl"). Wiring up a
    # historical UCL odds source is the remaining piece if you want that.
    if os.path.exists(BACKTEST_SUMMARY_PATH):
        with open(BACKTEST_SUMMARY_PATH) as f:
            summary = json.load(f)
        if "headline_takeaway" not in summary:
            roi = summary.get("flat_stake_roi")
            n = summary.get("n_bets")
            if roi is not None and n is not None:
                summary["headline_takeaway"] = (
                    f"This model lost money in backtesting: {roi:+.1%} ROI across {n} "
                    f"simulated flat-stake bets ({summary.get('seasons_tested', 'multiple seasons')}). "
                    f"Not a source of positive-EV picks — treat predictions as a research tool, not betting advice."
                )
            else:
                summary["headline_takeaway"] = "Backtest summary is missing expected fields."
        return summary
    return {
        "note": "No precomputed backtest summary bundled. Run `python3 main.py backtest` "
                "in the original project and copy output/backtest_summary.json into webapp/data/."
    }


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
