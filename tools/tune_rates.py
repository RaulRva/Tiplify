"""Barrido de parámetros del modelo de córners, tarjetas y tiros.

El modelo de goles y el de estadísticas secundarias no tienen por qué usar la
misma memoria: la fuerza ofensiva de un equipo persiste más que su tendencia a
forzar córners. Aquí se mide cada métrica por separado, siempre comparando
contra la línea base de "predecir la media de la liga".
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data import loader  # noqa: E402
from app.model import rates  # noqa: E402

EPSILON = 1e-12

METRICS = {
    "corners": ("home_corners", "away_corners", 9.5),
    "cards": ("home_cards", "away_cards", 4.5),
    "shots": ("home_shots", "away_shots", 22.5),
    "shots_target": ("home_shots_target", "away_shots_target", 8.5),
}


def evaluate(
    results: pd.DataFrame,
    metric: str,
    eval_matches: int,
    step: int,
    half_life: float,
    prior: float,
    min_train: int = 600,
) -> dict:
    home_column, away_column, line = METRICS[metric]
    xi = math.log(2) / half_life
    start = max(min_train, len(results) - eval_matches)

    errors: list[float] = []
    loglosses: list[float] = []
    base_errors: list[float] = []
    base_loglosses: list[float] = []

    for block_start in range(start, len(results), step):
        block = results.iloc[block_start : block_start + step]
        cutoff = block["date"].min()
        train = results.iloc[:block_start]
        train = train[train["date"] < cutoff]
        if len(train) < min_train:
            continue

        model = rates.fit_rate_model(
            train,
            home_column,
            away_column,
            metric,
            reference_date=cutoff,
            xi=xi,
            prior_matches=prior,
        )
        if model is None:
            continue

        for row in block.itertuples():
            observed = getattr(row, home_column) + getattr(row, away_column)
            if pd.isna(observed):
                continue
            if row.home not in model.generate or row.away not in model.generate:
                continue

            expected = sum(model.predict(row.home, row.away))
            errors.append(abs(expected - observed))
            probability = rates.prob_over(expected, line, model.dispersion)
            hit = observed > line
            loglosses.append(-math.log(max(probability if hit else 1 - probability, EPSILON)))

            league_mean = model.league_home + model.league_away
            base_errors.append(abs(league_mean - observed))
            base_probability = rates.prob_over(league_mean, line, model.dispersion)
            base_loglosses.append(
                -math.log(max(base_probability if hit else 1 - base_probability, EPSILON))
            )

    return {
        "half_life": half_life,
        "prior": prior,
        "n": len(errors),
        "mae": float(np.mean(errors)) if errors else float("nan"),
        "logloss": float(np.mean(loglosses)) if loglosses else float("nan"),
        "mae_base": float(np.mean(base_errors)) if base_errors else float("nan"),
        "logloss_base": float(np.mean(base_loglosses)) if base_loglosses else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=int, default=600)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--half-lives", type=float, nargs="*", default=[120, 200, 300, 450, 700])
    parser.add_argument("--priors", type=float, nargs="*", default=[2, 6, 12])
    parser.add_argument("--metrics", nargs="*", default=list(METRICS))
    args = parser.parse_args()

    results = loader.load_results()
    print(f"Histórico: {len(results)} partidos. Evaluando los últimos {args.eval}.\n")

    for metric in args.metrics:
        print(f"--- {metric} (línea {METRICS[metric][2]}) ---")
        reports = []
        for half_life in args.half_lives:
            for prior in args.priors:
                report = evaluate(results, metric, args.eval, args.step, half_life, prior)
                reports.append(report)
                print(
                    f"  vida media {half_life:>5.0f}d  prior {prior:>4.0f}  "
                    f"MAE {report['mae']:.4f} (base {report['mae_base']:.4f})  "
                    f"logloss {report['logloss']:.4f} (base {report['logloss_base']:.4f})"
                )
        best = min(reports, key=lambda item: item["logloss"])
        gain = best["logloss_base"] - best["logloss"]
        print(
            f"  => mejor: vida media {best['half_life']:.0f}d, prior {best['prior']:.0f} "
            f"| gana {gain:.4f} de logloss a la media de la liga "
            f"({best['n']} partidos)\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
