"""Modelo conjunto del partido, para poder valorar apuestas combinadas.

La cuota de una combinada **no** es el producto de las cuotas de sus patas.
"Gana el Barça" y "más de 2.5 goles" van de la mano: si el Barça gana suele
ser 3-0 o 3-1, así que la combinada ocurre más veces de lo que diría el
producto y su cuota justa es más baja. Con córners y goles pasa lo mismo, y
con "menos de 2.5 goles" ocurre al revés.

Para capturarlo se simula el partido entero de una vez:

1. El marcador sale de la matriz Dixon-Coles, así que cualquier mercado de
   resultado o de goles queda exacto por construcción.
2. Córners, tarjetas y tiros se enganchan al marcador con una **cópula
   gaussiana**: cada métrica tiene una variable latente normal correlacionada
   con el total de goles y con las demás métricas. La correlación se mide de
   los residuos del histórico, no de los valores brutos: lo que interesa es
   cuánto se mueven juntas *más allá* de lo que ya predicen los modelos.

Las probabilidades individuales no cambian: la cópula solo decide cómo se
reparten conjuntamente. Cada evento de línea se evalúa comparando la latente
con el umbral que reproduce exactamente su probabilidad marginal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from .. import config


def _gaussian_from_spearman(rho_spearman: float) -> float:
    """Correlación de la normal latente que reproduce esa correlación de rangos."""
    return float(2.0 * np.sin(np.pi * rho_spearman / 6.0))


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Corrige una matriz de correlación que no sea definida positiva.

    Al estimar cada par por separado (para aprovechar todas las filas de cada
    métrica) el resultado puede no ser una matriz de correlación válida.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() > 1e-8:
        return matrix
    clipped = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T
    scale = np.sqrt(np.diag(clipped))
    return clipped / np.outer(scale, scale)


@dataclass
class JointModel:
    """Cópula que une el marcador con las estadísticas secundarias."""

    metrics: tuple[str, ...]
    correlation: np.ndarray
    cholesky: np.ndarray

    def simulate(self, matrix: np.ndarray, seed: int, draws: int = config.SIM_DRAWS) -> Simulation:
        rng = np.random.default_rng(seed)
        latent = rng.standard_normal((draws, self.correlation.shape[0])) @ self.cholesky.T

        # El marcador se sortea con la latente 0. Ordenar los marcadores por
        # goles totales es lo que hace que la correlación con las métricas
        # tenga el signo correcto: latentes altas -> partidos con más goles.
        size_home, size_away = matrix.shape
        home_grid, away_grid = np.meshgrid(
            np.arange(size_home), np.arange(size_away), indexing="ij"
        )
        flat_home = home_grid.ravel()
        flat_away = away_grid.ravel()
        order = np.argsort(flat_home + flat_away, kind="stable")

        probabilities = matrix.ravel()[order]
        cumulative = np.cumsum(probabilities)
        cumulative /= cumulative[-1]

        picked = np.searchsorted(cumulative, norm.cdf(latent[:, 0]), side="left")
        picked = np.clip(picked, 0, len(order) - 1)

        return Simulation(
            home_goals=flat_home[order][picked],
            away_goals=flat_away[order][picked],
            latent={name: latent[:, i + 1] for i, name in enumerate(self.metrics)},
            draws=draws,
        )


@dataclass
class Simulation:
    """Un partido simulado muchas veces. Cada evento es una máscara booleana."""

    home_goals: np.ndarray
    away_goals: np.ndarray
    latent: dict[str, np.ndarray]
    draws: int

    @property
    def total_goals(self) -> np.ndarray:
        return self.home_goals + self.away_goals

    def metric_over(self, metric: str, under_probability: float) -> np.ndarray | None:
        """Máscara de "la métrica supera la línea", dada P(no la supera).

        El umbral se calcula para que la frecuencia de la máscara coincida con
        la probabilidad marginal que ya calculó el modelo de la métrica.
        """
        values = self.latent.get(metric)
        if values is None:
            return None
        threshold = norm.ppf(np.clip(under_probability, 1e-6, 1 - 1e-6))
        return values > threshold


def fit_joint_model(
    results: pd.DataFrame,
    goals_model,
    secondary: dict,
    metrics: tuple[str, ...],
) -> JointModel:
    """Mide la correlación entre los residuos de goles y de cada estadística."""
    residuals: dict[str, np.ndarray] = {}

    attack_home = results["home"].map(goals_model.attack).to_numpy(dtype=float)
    defence_home = results["home"].map(goals_model.defence).to_numpy(dtype=float)
    attack_away = results["away"].map(goals_model.attack).to_numpy(dtype=float)
    defence_away = results["away"].map(goals_model.defence).to_numpy(dtype=float)
    expected_goals = np.exp(goals_model.home_advantage + attack_home + defence_away) + np.exp(
        attack_away + defence_home
    )
    observed_goals = (results["home_goals"] + results["away_goals"]).to_numpy(dtype=float)
    residuals["goals"] = observed_goals - expected_goals

    from .predictor import SECONDARY_METRICS

    for metric in metrics:
        model = secondary.get(metric)
        if model is None:
            residuals[metric] = np.full(len(results), np.nan)
            continue
        home_column, away_column, _lines = SECONDARY_METRICS[metric]
        observed = (results[home_column] + results[away_column]).to_numpy(dtype=float)
        generate_home = results["home"].map(model.generate).to_numpy(dtype=float)
        concede_home = results["home"].map(model.concede).to_numpy(dtype=float)
        generate_away = results["away"].map(model.generate).to_numpy(dtype=float)
        concede_away = results["away"].map(model.concede).to_numpy(dtype=float)
        expected = (
            model.league_home * generate_home * concede_away
            + model.league_away * generate_away * concede_home
        )
        residuals[metric] = observed - expected

    names = ("goals",) + metrics
    size = len(names)
    correlation = np.eye(size)

    for i in range(size):
        for j in range(i + 1, size):
            left = residuals[names[i]]
            right = residuals[names[j]]
            mask = np.isfinite(left) & np.isfinite(right)
            if mask.sum() < config.CORRELATION_MIN_MATCHES:
                continue
            rho_spearman = spearmanr(left[mask], right[mask]).statistic
            if not np.isfinite(rho_spearman):
                continue
            value = _gaussian_from_spearman(float(rho_spearman))
            correlation[i, j] = correlation[j, i] = value

    correlation = _nearest_psd(correlation)
    return JointModel(
        metrics=metrics,
        correlation=correlation,
        cholesky=np.linalg.cholesky(correlation),
    )
