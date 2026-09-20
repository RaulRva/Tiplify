"""Forma reciente, clasificación y enfrentamientos directos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..data.teams import display_name

_PAIRS = [
    ("goals", "home_goals", "away_goals"),
    ("xg", "home_xg", "away_xg"),
    ("shots", "home_shots", "away_shots"),
    ("shots_target", "home_shots_target", "away_shots_target"),
    ("corners", "home_corners", "away_corners"),
    ("cards", "home_cards", "away_cards"),
    ("fouls", "home_fouls", "away_fouls"),
]


def long_view(results: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo y partido, con las estadísticas a favor y en contra."""
    if results.empty:
        return pd.DataFrame()

    frames = []
    for venue, team_col, opponent_col in (("home", "home", "away"), ("away", "away", "home")):
        side = pd.DataFrame(
            {
                "date": results["date"],
                "season": results["season"],
                "team": results[team_col],
                "opponent": results[opponent_col],
                "venue": venue,
            }
        )
        for name, home_col, away_col in _PAIRS:
            for_col, against_col = (
                (home_col, away_col) if venue == "home" else (away_col, home_col)
            )
            side[f"{name}_for"] = pd.to_numeric(results[for_col], errors="coerce")
            side[f"{name}_against"] = pd.to_numeric(results[against_col], errors="coerce")
        frames.append(side)

    view = pd.concat(frames, ignore_index=True)
    diff = view["goals_for"] - view["goals_against"]
    view["result"] = np.where(diff > 0, "W", np.where(diff == 0, "D", "L"))
    view["points"] = np.where(diff > 0, 3, np.where(diff == 0, 1, 0))
    view["total_goals"] = view["goals_for"] + view["goals_against"]
    view["total_corners"] = view["corners_for"] + view["corners_against"]
    view["total_cards"] = view["cards_for"] + view["cards_against"]
    view["btts"] = (view["goals_for"] > 0) & (view["goals_against"] > 0)
    return view.sort_values("date").reset_index(drop=True)


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _safe_sum(series: pd.Series) -> int | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return int(values.sum()) if len(values) else None


def recent_form(
    view: pd.DataFrame,
    team: str,
    matches: int = config.FORM_MATCHES,
    venue: str | None = None,
    league: str | None = None,
) -> dict:
    """Resumen de los últimos `matches` partidos de un equipo."""
    subset = view[view["team"] == team]
    if venue:
        subset = subset[subset["venue"] == venue]
    subset = subset.tail(matches)

    if subset.empty:
        return {
            "matches": 0,
            "streak": [],
            "recent": [],
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points_per_game": None,
            "goals_for": None,
            "goals_against": None,
            "goals_for_total": None,
            "goals_against_total": None,
            "xg_for": None,
            "xg_against": None,
            "shots_for": None,
            "shots_target_for": None,
            "corners_for": None,
            "corners_against": None,
            "cards_for": None,
            "over25_rate": None,
            "btts_rate": None,
            "clean_sheets": 0,
        }

    recent = [
        {
            "date": row.date,
            "opponent": display_name(row.opponent, league),
            "venue": row.venue,
            "result": row.result,
            "score": f"{int(row.goals_for)}-{int(row.goals_against)}",
            "corners": None if pd.isna(row.corners_for) else int(row.corners_for),
            "cards": None if pd.isna(row.cards_for) else int(row.cards_for),
            "shots": None if pd.isna(row.shots_for) else int(row.shots_for),
            "xg": None if pd.isna(row.xg_for) else round(float(row.xg_for), 2),
        }
        for row in subset.itertuples()
    ]
    recent.reverse()  # el más reciente primero

    return {
        "matches": int(len(subset)),
        "streak": list(subset["result"])[::-1],
        "recent": recent,
        "wins": int((subset["result"] == "W").sum()),
        "draws": int((subset["result"] == "D").sum()),
        "losses": int((subset["result"] == "L").sum()),
        "points_per_game": _safe_mean(subset["points"]),
        "goals_for": _safe_mean(subset["goals_for"]),
        "goals_against": _safe_mean(subset["goals_against"]),
        "goals_for_total": _safe_sum(subset["goals_for"]),
        "goals_against_total": _safe_sum(subset["goals_against"]),
        "xg_for": _safe_mean(subset["xg_for"]),
        "xg_against": _safe_mean(subset["xg_against"]),
        "shots_for": _safe_mean(subset["shots_for"]),
        "shots_target_for": _safe_mean(subset["shots_target_for"]),
        "corners_for": _safe_mean(subset["corners_for"]),
        "corners_against": _safe_mean(subset["corners_against"]),
        "cards_for": _safe_mean(subset["cards_for"]),
        "over25_rate": float((subset["total_goals"] > 2.5).mean()),
        "btts_rate": float(subset["btts"].mean()),
        "clean_sheets": int((subset["goals_against"] == 0).sum()),
    }


def head_to_head(
    results: pd.DataFrame,
    home: str,
    away: str,
    limit: int = config.H2H_MATCHES,
    league: str | None = None,
) -> dict:
    """Historial de enfrentamientos directos, el más reciente primero."""
    if results.empty:
        return {"matches": [], "count": 0}

    mask = ((results["home"] == home) & (results["away"] == away)) | (
        (results["home"] == away) & (results["away"] == home)
    )
    subset = results[mask].sort_values("date").tail(limit)
    if subset.empty:
        return {
            "matches": [],
            "count": 0,
            "home_wins": 0,
            "draws": 0,
            "away_wins": 0,
            "avg_goals": None,
            "avg_corners": None,
            "avg_cards": None,
            "btts_rate": None,
            "over25_rate": None,
        }

    matches = []
    home_wins = draws = away_wins = 0
    for row in subset.itertuples():
        hg, ag = int(row.home_goals), int(row.away_goals)
        if row.home == home:
            winner = "home" if hg > ag else ("away" if ag > hg else "draw")
        else:
            winner = "away" if hg > ag else ("home" if ag > hg else "draw")
        if winner == "home":
            home_wins += 1
        elif winner == "away":
            away_wins += 1
        else:
            draws += 1

        corners = row.home_corners + row.away_corners
        cards = row.home_cards + row.away_cards
        matches.append(
            {
                "date": row.date,
                "season": row.season,
                "home": display_name(row.home, league),
                "away": display_name(row.away, league),
                "score": f"{hg}-{ag}",
                "winner": winner,
                "corners": None if pd.isna(corners) else int(corners),
                "cards": None if pd.isna(cards) else int(cards),
            }
        )
    matches.reverse()

    totals = subset["home_goals"] + subset["away_goals"]
    return {
        "matches": matches,
        "count": int(len(subset)),
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "avg_goals": _safe_mean(totals),
        "avg_corners": _safe_mean(subset["home_corners"] + subset["away_corners"]),
        "avg_cards": _safe_mean(subset["home_cards"] + subset["away_cards"]),
        "btts_rate": float(((subset["home_goals"] > 0) & (subset["away_goals"] > 0)).mean()),
        "over25_rate": float((totals > 2.5).mean()),
    }


def standings(results: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Clasificación de la temporada indicada (por defecto, la más reciente)."""
    if results.empty:
        return pd.DataFrame()

    season = season or config.SEASONS[0]
    subset = results[results["season"] == season]
    if subset.empty:
        return pd.DataFrame()

    view = long_view(subset)
    table = (
        view.groupby("team")
        .agg(
            partidos=("points", "size"),
            puntos=("points", "sum"),
            ganados=("result", lambda s: int((s == "W").sum())),
            empatados=("result", lambda s: int((s == "D").sum())),
            perdidos=("result", lambda s: int((s == "L").sum())),
            gf=("goals_for", "sum"),
            gc=("goals_against", "sum"),
        )
        .reset_index()
    )
    table["dif"] = table["gf"] - table["gc"]
    table = table.sort_values(["puntos", "dif", "gf"], ascending=False).reset_index(drop=True)
    table.insert(0, "pos", table.index + 1)
    return table
