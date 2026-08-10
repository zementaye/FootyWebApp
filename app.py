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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

STATE = {"model": None, "fixtures_cache": [], "fixtures_error": None}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_PATH = os.path.join(DATA_DIR, "epl_results.csv")
ODDS_PATH = os.path.join(DATA_DIR, "epl_odds.csv")
BACKTEST_SUMMARY_PATH = os.path.join(DATA_DIR, "backtest_summary.json")


def refresh_fixtures_job():
    try:
        raw_fixtures = scraper.fetch_upcoming_fixtures(leagues=["E0"])  # EPL only by default
        model: ModelService = STATE["model"]
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
                logger.warning(f"Skipping fixture {fx.get('home_team')} vs {fx.get('away_team')}: {e}")
        STATE["fixtures_cache"] = enriched
        STATE["fixtures_error"] = None
        logger.info(f"Refreshed {len(enriched)} fixtures")
    except Exception as e:
        logger.error(f"Fixture scrape failed: {e}")
        STATE["fixtures_error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: training model...")
    STATE["model"] = ModelService(RESULTS_PATH, ODDS_PATH)
    logger.info("Model ready. Attempting initial fixture scrape...")
    refresh_fixtures_job()

    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_fixtures_job, "interval", hours=6)
    scheduler.start()
    STATE["scheduler"] = scheduler

    yield

    scheduler.shutdown()


app = FastAPI(title="Football Prediction API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": STATE["model"].ready if STATE["model"] else False}


@app.get("/api/predict")
def predict(home: str = Query(...), away: str = Query(...),
            odds_home: float = Query(None), odds_draw: float = Query(None), odds_away: float = Query(None)):
    model: ModelService = STATE["model"]
    if not model or not model.ready:
        raise HTTPException(503, "Model still training, try again shortly")
    try:
        return model.predict_match(home, away, odds_home, odds_draw, odds_away)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/fixtures")
def fixtures():
    return {
        "fixtures": STATE["fixtures_cache"],
        "scrape_error": STATE["fixtures_error"],
        "note": "Odds/predictions refresh every 6 hours. Draws are excluded from value_bet_flag "
                "based on backtest findings that draw bets lost money consistently.",
    }


@app.post("/api/refresh-fixtures")
def refresh_fixtures():
    refresh_fixtures_job()
    return {"refreshed": len(STATE["fixtures_cache"]), "error": STATE["fixtures_error"]}


@app.get("/api/backtest")
def backtest_summary():
    if os.path.exists(BACKTEST_SUMMARY_PATH):
        with open(BACKTEST_SUMMARY_PATH) as f:
            return json.load(f)
    return {
        "note": "No precomputed backtest summary bundled. Run `python3 main.py backtest` "
                "in the original project and copy output/backtest_summary.json into webapp/data/."
    }


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
