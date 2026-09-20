"""Modelo multiplicativo para córners, tarjetas, tiros, faltas y xG.

Para cada estadística se estima, por equipo, una tasa de *generar* (ataque) y
otra de *conceder* (defensa), más la media de la liga separada por localía:

    esperado(local i vs visitante j) = media_local x genera_i x concede_j
    esperado(visitante j en casa de i) = media_visitante x genera_j x concede_i

Los parámetros se ajustan con iteraciones proporcionales ponderadas por tiempo
(equivalente a un Poisson log-lineal) y con encogimiento hacia la media de la
liga, para que un equipo con pocos partidos no dé estimaciones extremas.

Córners y tarjetas tienen más varianza que un Poisson puro, así que las
probabilidades de línea se calculan con una binomial negativa cuya
sobredispersión se estima de los residuos del propio histórico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from .. import config
from .dixon_coles import time_weights


@dataclass
class RateModel:
    metric: str
    league_home: float
    league_away: float
    generate: dict[str, float]
    concede: dict[str, float]
    dispersion: float  # varianza/media del total (1.0 = Poisson)
    matches_used: int

    def predict(self, home: str, away: str) -> tuple[float, float]:
        gen_h = self.generate.get(home, 1.0)
        con_h = self.concede.get(home, 1.0)
        gen_a = self.generate.get(away, 1.0)
        con_a = self.concede.get(away, 1.0)
        expected_home = self.league_home * gen_h * con_a
        expected_away = self.league_away * gen_a * con_h
        return float(expected_home), float(expected_away)

    def team_profile(self, team: str) -> dict[str, float]:
        return {
            "generate": float(self.generate.get(team, 1.0)),
            "concede": float(self.concede.get(team, 1.0)),
        }


def fit_rate_model(
    results: pd.DataFrame,
    home_column: str,
    away_column: str,
    metric: str,
    reference_date: pd.Timestamp | None = None,
    xi: float = config.TIME_DECAY_XI,
    prior_matches: float = config.RATE_MODEL_PRIOR_MATCHES,
    iterations: int = config.RATE_MODEL_ITERATIONS,
) -> RateModel | None:
    """Devuelve el modelo, o None si no hay datos suficientes de esa métrica."""
    if results.empty or home_column not in results or away_column not in results:
        return None

    frame = results.dropna(subset=[home_column, away_column])
    if len(frame) < 40:
        return None

    reference = reference_date or frame["date"].max()
    weights = time_weights(frame["date"], reference, xi)
    if weights.sum() <= 0:
        return None

    teams = sorted(set(frame["home"]) | set(frame["away"]))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    home_idx = frame["home"].map(index).to_numpy()
    away_idx = frame["away"].map(index).to_numpy()
    value_home = frame[home_column].to_numpy(dtype=float)
    value_away = frame[away_column].to_numpy(dtype=float)

    league_home = float(np.average(value_home, weights=weights))
    league_away = float(np.average(value_away, weights=weights))
    if league_home <= 0 or league_away <= 0:
        return None

    generate = np.ones(n)
    concede = np.ones(n)

    # Peso del "prior": equivale a `prior_matches` partidos en la media de liga.
    prior_units = prior_matches * (league_home + league_away) / 2.0

    # Totales observados por equipo (generados y concedidos), ponderados.
    generated = np.zeros(n)
    conceded = np.zeros(n)
    np.add.at(generated, home_idx, weights * value_home)
    np.add.at(generated, away_idx, weights * value_away)
    np.add.at(conceded, home_idx, weights * value_away)
    np.add.at(conceded, away_idx, weights * value_home)

    for _ in range(iterations):
        # Base esperada asumiendo generate = 1 para el equipo en cuestión.
        base_generate = np.zeros(n)
        np.add.at(base_generate, home_idx, weights * league_home * concede[away_idx])
        np.add.at(base_generate, away_idx, weights * league_away * concede[home_idx])
        generate = (generated + prior_units) / np.maximum(base_generate + prior_units, 1e-9)
        generate /= max(generate.mean(), 1e-9)

        base_concede = np.zeros(n)
        np.add.at(base_concede, home_idx, weights * league_away * generate[away_idx])
        np.add.at(base_concede, away_idx, weights * league_home * generate[home_idx])
        concede = (conceded + prior_units) / np.maximum(base_concede + prior_units, 1e-9)
        concede /= max(concede.mean(), 1e-9)

    predicted_home = league_home * generate[home_idx] * concede[away_idx]
    predicted_away = league_away * generate[away_idx] * concede[home_idx]
    predicted_total = predicted_home + predicted_away
    observed_total = value_home + value_away

    residual = (observed_total - predicted_total) ** 2
    mean_predicted = float(np.average(predicted_total, weights=weights))
    dispersion = 1.0
    if mean_predicted > 0:
        dispersion = float(np.average(residual, weights=weights) / mean_predicted)
    dispersion = float(np.clip(dispersion, 1.0, 3.0))

    return RateModel(
        metric=metric,
        league_home=league_home,
        league_away=league_away,
        generate={team: float(generate[i]) for team, i in index.items()},
        concede={team: float(concede[i]) for team, i in index.items()},
        dispersion=dispersion,
        matches_used=int(len(frame)),
    )


def prob_over(mean: float, line: float, dispersion: float = 1.0) -> float:
    """P(total > línea). Poisson si no hay sobredispersión, si no binomial negativa."""
    if mean <= 0:
        return 0.0
    threshold = int(np.floor(line))
    if dispersion <= 1.02:
        return float(poisson.sf(threshold, mean))
    p = 1.0 / dispersion
    r = mean / (dispersion - 1.0)
    return float(nbinom.sf(threshold, r, p))


def line_table(mean: float, lines: list[float], dispersion: float = 1.0) -> list[dict]:
    rows = []
    for line in lines:
        over = prob_over(mean, line, dispersion)
        rows.append({"line": line, "over": over, "under": 1.0 - over})
    return rows
