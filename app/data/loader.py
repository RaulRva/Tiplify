"""Descarga y normalización de los datos de football-data.co.uk.

Los CSV se guardan en `.cache/` para no volver a descargarlos en cada arranque
y para que la app siga funcionando sin conexión con los últimos datos válidos.
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime

import pandas as pd
import requests

from .. import config
from .teams import resolve_key, slugify

log = logging.getLogger(__name__)

# Columnas del CSV -> nombre interno.
_STAT_COLUMNS = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "HxG": "home_xg",
    "AxG": "away_xg",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
}

_ODDS_COLUMNS = {
    "AvgH": "odds_home",
    "AvgD": "odds_draw",
    "AvgA": "odds_away",
    "B365H": "odds_home_b365",
    "B365D": "odds_draw_b365",
    "B365A": "odds_away_b365",
    "Avg>2.5": "odds_over25",
    "Avg<2.5": "odds_under25",
}


def _cache_path(name: str):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / name


def _fetch(url: str, cache_name: str, ttl: int) -> str | None:
    """Devuelve el CSV como texto, usando caché si es reciente.

    Si la descarga falla se reutiliza la copia en disco aunque haya caducado.
    """
    path = _cache_path(cache_name)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < ttl
    if fresh:
        return path.read_text(encoding="utf-8", errors="replace")

    try:
        response = requests.get(
            url,
            timeout=config.HTTP_TIMEOUT,
            headers={"User-Agent": "Tiplify/1.0"},
        )
        response.raise_for_status()
        text = response.content.decode("utf-8", errors="replace")
        path.write_text(text, encoding="utf-8")
        return text
    except Exception as exc:  # red caída, 404 de temporada futura, etc.
        log.warning("No se pudo descargar %s (%s)", url, exc)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return None


def _parse_dates(series: pd.Series) -> pd.Series:
    """Las temporadas antiguas usan dd/mm/yy y las nuevas dd/mm/yyyy."""
    parsed = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            series[missing], format="%d/%m/%y", errors="coerce"
        )
    return parsed


def _read_csv(text: str) -> pd.DataFrame:
    # Algunas filas finales vienen vacías o con columnas de más.
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip", encoding_errors="replace")


def _normalize(raw: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame["date"] = _parse_dates(raw["Date"])
    frame["time"] = raw["Time"].astype(str) if "Time" in raw else "--:--"
    frame["home"] = raw["HomeTeam"].astype(str).str.strip()
    frame["away"] = raw["AwayTeam"].astype(str).str.strip()

    for source, target in {**_STAT_COLUMNS, **_ODDS_COLUMNS}.items():
        frame[target] = (
            pd.to_numeric(raw[source], errors="coerce")
            if source in raw
            else float("nan")
        )

    if "Referee" in raw:
        frame["referee"] = raw["Referee"].astype(str).str.strip()
    else:
        frame["referee"] = ""

    frame["season"] = season or ""
    frame = frame.dropna(subset=["date", "home", "away"])
    frame = frame[(frame["home"] != "") & (frame["away"] != "")]
    return frame


def load_results(league: config.League | str | None = None) -> pd.DataFrame:
    """Histórico de partidos jugados de las temporadas configuradas."""
    league = config.get_league(league)
    frames: list[pd.DataFrame] = []
    for season in config.SEASONS:
        url = config.RESULTS_URL.format(season=season, league=league.code)
        text = _fetch(url, f"{league.code}_{season}.csv", config.RESULTS_TTL_SECONDS)
        if not text:
            continue
        try:
            raw = _read_csv(text)
        except Exception as exc:
            log.warning("CSV ilegible para la temporada %s (%s)", season, exc)
            continue
        if "HomeTeam" not in raw:
            continue
        frames.append(_normalize(raw, season))

    if not frames:
        return pd.DataFrame()

    results = pd.concat(frames, ignore_index=True)
    # Solo partidos ya jugados (con resultado final).
    results = results.dropna(subset=["home_goals", "away_goals"])
    results["home_cards"] = results["home_yellow"].fillna(0) + results["home_red"].fillna(0)
    results["away_cards"] = results["away_yellow"].fillna(0) + results["away_red"].fillna(0)
    results = results.sort_values("date").reset_index(drop=True)
    return results


def make_match_id(date: pd.Timestamp, home: str, away: str) -> str:
    return f"{date:%Y%m%d}-{slugify(home)}-{slugify(away)}"


def _load_odds(league: config.League) -> pd.DataFrame:
    """Cuotas de los próximos partidos (football-data cubre ~1 semana)."""
    text = _fetch(config.FIXTURES_URL, "fixtures.csv", config.FIXTURES_TTL_SECONDS)
    if not text:
        return pd.DataFrame()

    try:
        raw = _read_csv(text)
    except Exception as exc:
        log.warning("Fichero de cuotas ilegible (%s)", exc)
        return pd.DataFrame()

    if "Div" not in raw or "HomeTeam" not in raw:
        return pd.DataFrame()

    raw = raw[raw["Div"].astype(str).str.strip() == league.code]
    if raw.empty:
        return pd.DataFrame()

    odds = _normalize(raw)
    odds["match_id"] = [
        make_match_id(row.date, row.home, row.away) for row in odds.itertuples()
    ]
    columns = ["match_id", "referee", *_ODDS_COLUMNS.values()]
    return odds[columns].drop_duplicates(subset="match_id")


def load_calendar(league: config.League | str | None = None) -> pd.DataFrame:
    """Calendario completo de la temporada (las 38 jornadas)."""
    league = config.get_league(league)
    label = config.season_label()
    text = _fetch(
        config.CALENDAR_URL.format(label=label, file=league.calendar_file),
        f"calendar_{league.slug}_{label}.json",
        config.CALENDAR_TTL_SECONDS,
    )
    if not text:
        return pd.DataFrame()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Calendario JSON ilegible (%s)", exc)
        return pd.DataFrame()

    rows = []
    for match in payload.get("matches", []):
        home = resolve_key(match.get("team1", ""), league.slug)
        away = resolve_key(match.get("team2", ""), league.slug)
        if not home or not away:
            log.debug("Equipo sin equivalencia: %s / %s", match.get("team1"), match.get("team2"))
            continue
        rows.append(
            {
                "date": pd.to_datetime(match.get("date"), errors="coerce"),
                "time": match.get("time") or "--:--",
                "home": home,
                "away": away,
                "round": match.get("round", ""),
                "played": bool(match.get("score")),
            }
        )

    calendar = pd.DataFrame(rows).dropna(subset=["date"])
    if calendar.empty:
        return calendar
    return calendar.sort_values(["date", "time"]).reset_index(drop=True)


def load_fixtures(
    league: config.League | str | None = None,
    known_teams: set[str] | None = None,
    include_odds: bool = True,
) -> pd.DataFrame:
    """Próximos partidos con sus cuotas cuando estén disponibles."""
    league = config.get_league(league)
    fixtures = load_calendar(league)
    source = "calendar"

    if fixtures.empty:
        # Sin calendario completo, al menos mostramos la semana con cuotas.
        fixtures = _fixtures_from_odds_file(league)
        source = "odds"
        if fixtures.empty:
            return pd.DataFrame()

    # Un partido de hoy sigue siendo "próximo" hasta que se juega: el
    # calendario trae el marcador en cuanto termina, así que sirve para
    # descartarlo aunque la jornada siga en curso.
    today = pd.Timestamp(datetime.now().date())
    fixtures = fixtures[(fixtures["date"] >= today) & ~fixtures["played"]].copy()

    if known_teams:
        fixtures = fixtures[
            fixtures["home"].isin(known_teams) & fixtures["away"].isin(known_teams)
        ].copy()

    if fixtures.empty:
        return fixtures

    fixtures["match_id"] = [
        make_match_id(row.date, row.home, row.away) for row in fixtures.itertuples()
    ]
    fixtures["source"] = source

    if include_odds:
        odds = _load_odds(league)
        if not odds.empty:
            fixtures = fixtures.merge(odds, on="match_id", how="left", suffixes=("", "_odds"))

    for column in _ODDS_COLUMNS.values():
        if column not in fixtures:
            fixtures[column] = float("nan")
    if "round" not in fixtures:
        fixtures["round"] = ""

    return fixtures.sort_values(["date", "time"]).reset_index(drop=True)


def _fixtures_from_odds_file(league: config.League) -> pd.DataFrame:
    """Calendario de emergencia construido desde el fichero de cuotas."""
    text = _fetch(config.FIXTURES_URL, "fixtures.csv", config.FIXTURES_TTL_SECONDS)
    if not text:
        return pd.DataFrame()
    try:
        raw = _read_csv(text)
    except Exception:
        return pd.DataFrame()
    if "Div" not in raw or "HomeTeam" not in raw:
        return pd.DataFrame()
    raw = raw[raw["Div"].astype(str).str.strip() == league.code]
    if raw.empty:
        return pd.DataFrame()
    fixtures = _normalize(raw)
    fixtures["round"] = ""
    fixtures["played"] = False
    return fixtures[["date", "time", "home", "away", "round", "played"]]


def data_freshness(league: config.League | str | None = None) -> dict[str, object]:
    """Metadatos para mostrar en la UI cuándo se actualizaron los datos."""
    league = config.get_league(league)
    info: dict[str, object] = {"results": None, "fixtures": None, "calendar": None}
    for label, name in (
        ("results", f"{league.code}_{config.SEASONS[0]}.csv"),
        ("fixtures", "fixtures.csv"),
        ("calendar", f"calendar_{league.slug}_{config.season_label()}.json"),
    ):
        path = _cache_path(name)
        if path.exists():
            info[label] = datetime.fromtimestamp(path.stat().st_mtime)
    return info


def clear_cache() -> None:
    """Borra los CSV en caché para forzar una descarga limpia."""
    if not config.CACHE_DIR.exists():
        return
    for pattern in ("*.csv", "*.json"):
        for path in config.CACHE_DIR.glob(pattern):
            try:
                path.unlink()
            except OSError as exc:
                log.warning("No se pudo borrar %s (%s)", path, exc)
