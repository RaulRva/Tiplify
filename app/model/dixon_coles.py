"""Modelo de goles Dixon-Coles con decaimiento temporal.

Cada equipo tiene un parámetro de ataque y otro de defensa; hay un factor de
localía global y un parámetro `rho` que corrige la dependencia entre los goles
de ambos equipos en resultados bajos (0-0, 1-0, 0-1, 1-1), donde el Poisson
puro se queda corto. Los partidos antiguos pesan menos (vida media en días).

Referencia: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .. import config


def time_weights(dates: pd.Series, reference: pd.Timestamp, xi: float) -> np.ndarray:
    days = (reference - dates).dt.total_seconds().to_numpy() / 86400.0
    days = np.clip(days, 0.0, None)
    return np.exp(-xi * days)


def _tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Corrección de Dixon-Coles para los marcadores bajos."""
    tau = np.ones_like(lambda_home, dtype=float)
    mask = (home_goals == 0) & (away_goals == 0)
    tau[mask] = 1.0 - lambda_home[mask] * lambda_away[mask] * rho
    mask = (home_goals == 0) & (away_goals == 1)
    tau[mask] = 1.0 + lambda_home[mask] * rho
    mask = (home_goals == 1) & (away_goals == 0)
    tau[mask] = 1.0 + lambda_away[mask] * rho
    mask = (home_goals == 1) & (away_goals == 1)
    tau[mask] = 1.0 - rho
    return np.clip(tau, 1e-10, None)


@dataclass
class GoalsModel:
    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    league_home_goals: float
    league_away_goals: float
    matches_used: int
    weighted_matches: dict[str, float] = field(default_factory=dict)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Goles esperados (lambda) de local y visitante."""
        atk_h = self.attack.get(home, 0.0)
        def_h = self.defence.get(home, 0.0)
        atk_a = self.attack.get(away, 0.0)
        def_a = self.defence.get(away, 0.0)
        lambda_home = np.exp(self.home_advantage + atk_h + def_a)
        lambda_away = np.exp(atk_a + def_h)
        return float(lambda_home), float(lambda_away)

    def score_matrix(
        self,
        home: str,
        away: str,
        lambda_home: float | None = None,
        lambda_away: float | None = None,
        max_goals: int = config.MAX_GOALS,
    ) -> np.ndarray:
        """Matriz de probabilidad conjunta P(goles_local, goles_visitante)."""
        if lambda_home is None or lambda_away is None:
            lambda_home, lambda_away = self.expected_goals(home, away)

        goals = np.arange(max_goals + 1)
        p_home = poisson.pmf(goals, lambda_home)
        p_away = poisson.pmf(goals, lambda_away)
        matrix = np.outer(p_home, p_away)

        # Corrección rho en la esquina de marcadores bajos.
        matrix[0, 0] *= 1.0 - lambda_home * lambda_away * self.rho
        matrix[0, 1] *= 1.0 + lambda_home * self.rho
        matrix[1, 0] *= 1.0 + lambda_away * self.rho
        matrix[1, 1] *= 1.0 - self.rho

        matrix = np.clip(matrix, 0.0, None)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    def team_strength(self, team: str) -> dict[str, float]:
        """Ataque y defensa en escala multiplicativa (1.0 = media de la liga)."""
        return {
            "attack": float(np.exp(self.attack.get(team, 0.0))),
            "defence": float(np.exp(self.defence.get(team, 0.0))),
            "sample": float(self.weighted_matches.get(team, 0.0)),
        }


def fit_goals_model(
    results: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    xi: float = config.TIME_DECAY_XI,
    ridge: float = config.GOALS_RIDGE,
) -> GoalsModel:
    """Ajusta el modelo por máxima verosimilitud ponderada."""
    if results.empty:
        raise ValueError("No hay partidos para entrenar el modelo de goles.")

    reference = reference_date or results["date"].max()
    teams = sorted(set(results["home"]) | set(results["away"]))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    home_idx = results["home"].map(index).to_numpy()
    away_idx = results["away"].map(index).to_numpy()
    home_goals = results["home_goals"].to_numpy(dtype=float)
    away_goals = results["away_goals"].to_numpy(dtype=float)
    weights = time_weights(results["date"], reference, xi)
    total_weight = float(weights.sum())

    # Parámetros: n ataques, n defensas, localía, rho.
    def unpack(params: np.ndarray):
        attack = params[:n]
        defence = params[n : 2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1]
        # Identificabilidad: el ataque medio es 0.
        attack = attack - attack.mean()
        return attack, defence, home_adv, rho

    def negative_log_likelihood(params: np.ndarray) -> float:
        attack, defence, home_adv, rho = unpack(params)
        log_lh = home_adv + attack[home_idx] + defence[away_idx]
        log_la = attack[away_idx] + defence[home_idx]
        log_lh = np.clip(log_lh, -6.0, 3.0)
        log_la = np.clip(log_la, -6.0, 3.0)
        lambda_h = np.exp(log_lh)
        lambda_a = np.exp(log_la)

        tau = _tau(home_goals, away_goals, lambda_h, lambda_a, rho)
        log_p = (
            np.log(tau)
            + home_goals * log_lh
            - lambda_h
            + away_goals * log_la
            - lambda_a
        )
        # Encoge los parámetros hacia 0 (= media de la liga). El peso efectivo
        # es relativo al total de partidos ponderados, así que los equipos con
        # poca muestra son los que más se acercan a la media.
        penalty = ridge * total_weight * (np.sum(attack**2) + np.sum(defence**2)) / n
        return -float(np.sum(weights * log_p)) + penalty

    start = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.03]])
    bounds = [(-2.5, 2.5)] * n + [(-2.5, 2.5)] * n + [(-0.5, 1.5), (-0.25, 0.25)]

    solution = minimize(
        negative_log_likelihood,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 600, "ftol": 1e-9},
    )

    attack, defence, home_adv, rho = unpack(solution.x)

    weighted_matches: dict[str, float] = {}
    for team, i in index.items():
        mask = (home_idx == i) | (away_idx == i)
        weighted_matches[team] = float(weights[mask].sum())

    return GoalsModel(
        teams=teams,
        attack={team: float(attack[i]) for team, i in index.items()},
        defence={team: float(defence[i]) for team, i in index.items()},
        home_advantage=float(home_adv),
        rho=float(rho),
        league_home_goals=float(np.average(home_goals, weights=weights)),
        league_away_goals=float(np.average(away_goals, weights=weights)),
        matches_used=int(len(results)),
        weighted_matches=weighted_matches,
    )


# --- Derivados de la matriz de marcadores ------------------------------------


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return {"home": home, "draw": draw, "away": away}


def total_goals_distribution(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0] + matrix.shape[1] - 1
    distribution = np.zeros(size)
    for total in range(size):
        distribution[total] = sum(
            matrix[h, total - h]
            for h in range(max(0, total - matrix.shape[1] + 1), min(total, matrix.shape[0] - 1) + 1)
        )
    return distribution


def prob_over(distribution: np.ndarray, line: float) -> float:
    threshold = int(np.floor(line)) + 1
    return float(distribution[threshold:].sum())


def prob_btts(matrix: np.ndarray) -> float:
    return float(matrix[1:, 1:].sum())


def top_scorelines(matrix: np.ndarray, limit: int = 6) -> list[dict[str, float | int]]:
    flat = [
        {"home": h, "away": a, "prob": float(matrix[h, a])}
        for h in range(matrix.shape[0])
        for a in range(matrix.shape[1])
    ]
    flat.sort(key=lambda item: item["prob"], reverse=True)
    return flat[:limit]


def handicap_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Probabilidad de que el local gane por 1, 2 o más goles, etc."""
    size = matrix.shape[0]
    margins: dict[int, float] = {}
    for h in range(size):
        for a in range(matrix.shape[1]):
            margins[h - a] = margins.get(h - a, 0.0) + float(matrix[h, a])
    return {
        "home_by_2plus": sum(p for m, p in margins.items() if m >= 2),
        "home_by_1": margins.get(1, 0.0),
        "draw": margins.get(0, 0.0),
        "away_by_1": margins.get(-1, 0.0),
        "away_by_2plus": sum(p for m, p in margins.items() if m <= -2),
    }
