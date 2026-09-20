"""Aplicación web de Tiplify (FastAPI + plantillas Jinja)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from itertools import groupby

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config
from .data import loader
from .model import features as ft
from .model import predictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tiplify")

app = FastAPI(title="Tiplify", description="Pronósticos de LaLiga, Premier League y Serie A")
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=str(config.BASE_DIR / "app" / "templates"))

MONTHS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

DEFAULT_LIMIT = 20


# --- Filtros de plantilla ----------------------------------------------------


def pct(value: float | None, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def num(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return f"{float(value):.{decimals}f}"


def signed(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:+.{decimals}f}"


def long_date(value) -> str:
    if value is None:
        return "—"
    moment = pd.Timestamp(value)
    return f"{WEEKDAYS[moment.weekday()]} {moment.day} de {MONTHS[moment.month - 1]}"


def short_date(value) -> str:
    if value is None:
        return "—"
    moment = pd.Timestamp(value)
    return f"{moment.day:02d}/{moment.month:02d}"


def date_time(value) -> str:
    if value is None:
        return "—"
    moment = pd.Timestamp(value)
    return f"{moment.day:02d}/{moment.month:02d}/{moment.year} {moment:%H:%M}"


def matchday(value: str) -> str:
    """'Matchday 7' -> 'Jornada 7'."""
    if not value:
        return ""
    return value.replace("Matchday", "Jornada").strip()


def kickoff(value: str) -> str:
    """Las jornadas lejanas aún no tienen horario asignado."""
    if not value or value.strip() in {"--:--", "nan", ""}:
        return "Hora por confirmar"
    return f"{value.strip()} h"


templates.env.filters.update(
    {
        "pct": pct,
        "num": num,
        "signed": signed,
        "long_date": long_date,
        "short_date": short_date,
        "date_time": date_time,
        "matchday": matchday,
        "kickoff": kickoff,
    }
)


# --- Utilidades --------------------------------------------------------------


def jsonable(value):
    """Convierte numpy/pandas a tipos serializables en JSON."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def freshness_context(league: config.League | None = None) -> dict:
    info = loader.data_freshness(league)
    return {
        "results_updated": info.get("results"),
        "fixtures_updated": info.get("fixtures"),
        "calendar_updated": info.get("calendar"),
    }


def parse_league(slug: str) -> config.League:
    try:
        return config.get_league(slug)
    except KeyError:
        raise HTTPException(status_code=404, detail="Liga no encontrada") from None


def base_context(request: Request, engine: predictor.Engine | None = None, league: config.League | None = None) -> dict:
    league = league or (engine.league if engine is not None else config.default_league())
    context = {
        "request": request,
        "league": league.name,
        "league_slug": league.slug,
        "country": league.country,
        "kickoff_note": league.kickoff_note,
        "season": config.season_label(),
        "margin_1x2": config.MARGIN_1X2,
        "margin_binary": config.MARGIN_BINARY,
        "leagues": list(config.LEAGUES.values()),
        "paths": {
            "home": f"/{league.slug}",
            "standings": f"/{league.slug}/clasificacion",
            "refresh": f"/{league.slug}/actualizar",
            "api": f"/api/{league.slug}/partidos",
        },
        "section": "",
        **freshness_context(league),
    }
    if engine is not None:
        goals = engine.goals_model
        context.update(
            {
                "model_matches": goals.matches_used,
                "model_teams": len(goals.teams),
                "model_fitted_at": engine.fitted_at,
                "home_advantage_factor": float(np.exp(goals.home_advantage)),
                "league_home_goals": goals.league_home_goals,
                "league_away_goals": goals.league_away_goals,
            }
        )
    return context


def error_page(request: Request, message: str, status_code: int = 500, league: config.League | None = None):
    return templates.TemplateResponse(
        "error.html",
        {**base_context(request, league=league), "message": message},
        status_code=status_code,
    )


# --- Vistas ------------------------------------------------------------------


def _matches_page(request: Request, league: config.League, n: int):
    try:
        engine = predictor.get_engine(league)
    except Exception as exc:
        log.exception("Fallo al preparar el modelo")
        return error_page(request, str(exc), league=league)

    fixtures = predictor.upcoming_fixtures(engine)
    total_upcoming = len(fixtures)
    predictions = predictor.upcoming_predictions(engine, limit=n, full=False)

    days = [
        {"date": day, "matches": list(group)}
        for day, group in groupby(predictions, key=lambda item: pd.Timestamp(item["date"]).date())
    ]

    context = base_context(request, engine, league)
    context["section"] = "matches"
    return templates.TemplateResponse(
        "index.html",
        {
            **context,
            "days": days,
            "shown": len(predictions),
            "total_upcoming": total_upcoming,
            "limit": n,
            "next_limit": min(total_upcoming, n + 20),
        },
    )


def _match_page(request: Request, league: config.League, match_id: str):
    try:
        engine = predictor.get_engine(league)
    except Exception as exc:
        log.exception("Fallo al preparar el modelo")
        return error_page(request, str(exc), league=league)

    prediction = predictor.find_prediction(engine, match_id)
    if prediction is None:
        return error_page(
            request,
            "No he encontrado ese partido entre los próximos del calendario.",
            status_code=404,
            league=league,
        )

    context = base_context(request, engine, league)
    context["section"] = "matches"
    return templates.TemplateResponse(
        "match.html",
        {
            **context,
            "p": prediction,
            "main_lines": config.MAIN_LINES,
            "form_matches": config.FORM_MATCHES,
        },
    )


def _standings_page(request: Request, league: config.League):
    try:
        engine = predictor.get_engine(league)
    except Exception as exc:
        log.exception("Fallo al preparar el modelo")
        return error_page(request, str(exc), league=league)

    table = engine.table
    rows = []
    for record in table.to_dict("records"):
        team = record["team"]
        strength = engine.goals_model.team_strength(team)
        corners = engine.secondary.get("corners")
        cards = engine.secondary.get("cards")
        rows.append(
            {
                **record,
                "display": ft.display_name(team, league.slug),
                "attack": strength["attack"],
                "defence": strength["defence"],
                "corners_rate": corners.team_profile(team)["generate"] if corners else None,
                "cards_rate": cards.team_profile(team)["generate"] if cards else None,
            }
        )

    context = base_context(request, engine, league)
    context["section"] = "standings"
    return templates.TemplateResponse("standings.html", {**context, "rows": rows})


@app.get("/")
def index(request: Request, n: int = Query(DEFAULT_LIMIT, ge=1, le=380)):
    return _matches_page(request, config.default_league(), n)


@app.get("/partido/{match_id}")
def match_detail(request: Request, match_id: str):
    return _match_page(request, config.default_league(), match_id)


@app.get("/clasificacion")
def standings_page(request: Request):
    return _standings_page(request, config.default_league())


@app.post("/actualizar")
@app.get("/actualizar")
def refresh():
    loader.clear_cache()
    predictor.get_engine(config.default_league(), force=True)
    return RedirectResponse(url="/laliga", status_code=303)


@app.get("/{league}")
def index_league(
    request: Request,
    league: str,
    n: int = Query(DEFAULT_LIMIT, ge=1, le=380),
):
    return _matches_page(request, parse_league(league), n)


@app.get("/{league}/partido/{match_id}")
def match_detail_league(request: Request, league: str, match_id: str):
    return _match_page(request, parse_league(league), match_id)


@app.get("/{league}/clasificacion")
def standings_league(request: Request, league: str):
    return _standings_page(request, parse_league(league))


@app.post("/{league}/actualizar")
@app.get("/{league}/actualizar")
def refresh_league(league: str):
    chosen = parse_league(league)
    loader.clear_cache()
    predictor.get_engine(chosen, force=True)
    return RedirectResponse(url=f"/{chosen.slug}", status_code=303)


# --- API JSON ----------------------------------------------------------------


def _api_status_payload(engine: predictor.Engine) -> dict:
    league = engine.league
    return {
        "liga": league.name,
        "slug": league.slug,
        "temporada": config.season_label(),
        "partidos_entrenamiento": engine.goals_model.matches_used,
        "equipos": len(engine.goals_model.teams),
        "ventaja_local": engine.goals_model.home_advantage,
        "rho": engine.goals_model.rho,
        "modelo_entrenado": engine.fitted_at,
        "datos": loader.data_freshness(league),
        "metricas": {
            name: None
            if model is None
            else {
                "media_local": model.league_home,
                "media_visitante": model.league_away,
                "dispersion": model.dispersion,
                "partidos": model.matches_used,
            }
            for name, model in engine.secondary.items()
        },
    }


@app.get("/api/partidos")
def api_matches(n: int = Query(DEFAULT_LIMIT, ge=1, le=380), completo: bool = False):
    engine = predictor.get_engine()
    predictions = predictor.upcoming_predictions(engine, limit=n, full=completo)
    return JSONResponse(jsonable({"total": len(predictions), "partidos": predictions}))


@app.get("/api/partidos/{match_id}")
def api_match(match_id: str):
    engine = predictor.get_engine()
    prediction = predictor.find_prediction(engine, match_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    return JSONResponse(jsonable(prediction))


@app.get("/api/estado")
def api_status():
    engine = predictor.get_engine()
    return JSONResponse(jsonable(_api_status_payload(engine)))


@app.get("/api/{league}/partidos")
def api_matches_league(
    league: str, n: int = Query(DEFAULT_LIMIT, ge=1, le=380), completo: bool = False
):
    engine = predictor.get_engine(parse_league(league))
    predictions = predictor.upcoming_predictions(engine, limit=n, full=completo)
    return JSONResponse(jsonable({"total": len(predictions), "partidos": predictions}))


@app.get("/api/{league}/partidos/{match_id}")
def api_match_league(league: str, match_id: str):
    engine = predictor.get_engine(parse_league(league))
    prediction = predictor.find_prediction(engine, match_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    return JSONResponse(jsonable(prediction))


@app.get("/api/{league}/estado")
def api_status_league(league: str):
    engine = predictor.get_engine(parse_league(league))
    return JSONResponse(jsonable(_api_status_payload(engine)))
