"""Backtest walk-forward del modelo, comparado con el mercado de apuestas.

Para cada bloque de partidos se entrena el modelo SOLO con los partidos
anteriores a ese bloque y se predicen sus resultados. Así medimos el error
igual que lo veremos en producción, sin filtrar información del futuro.

Métricas:
  logloss  -> verosimilitud del 1X2 (menor es mejor). El mercado suele estar
              alrededor de 0.98-1.02 en LaLiga; un modelo decente ronda 1.00-1.05.
  brier    -> error cuadrático multiclase (menor es mejor).
  acierto  -> % de veces que el resultado más probable fue el correcto.
  MAE      -> error absoluto medio en córners y tarjetas totales.

Uso:
    python tools/backtest.py                 # rejilla por defecto
    python tools/backtest.py --eval 380 --step 20
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.data import loader  # noqa: E402
from app.model import dixon_coles as dc  # noqa: E402
from app.model import rates  # noqa: E402

EPSILON = 1e-12


def _actual_outcome(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def _market_probabilities(row: pd.Series) -> dict[str, float] | None:
    odds = [row.get("odds_home"), row.get("odds_draw"), row.get("odds_away")]
    if any(o is None or pd.isna(o) or float(o) <= 1.0 for o in odds):
        return None
    inverse = [1.0 / float(o) for o in odds]
    total = sum(inverse)
    keys = ("home", "draw", "away")
    return {key: value / total for key, value in zip(keys, inverse)}


def run_backtest(
    results: pd.DataFrame,
    eval_matches: int,
    step: int,
    half_life: float,
    ridge: float,
    min_train: int = 600,
) -> dict:
    xi = math.log(2) / half_life
    start = max(min_train, len(results) - eval_matches)

    model_logloss: list[float] = []
    model_brier: list[float] = []
    model_hits: list[int] = []
    market_logloss: list[float] = []
    market_brier: list[float] = []
    market_hits: list[int] = []
    corner_errors: list[float] = []
    card_errors: list[float] = []
    corner_over_logloss: list[float] = []
    card_over_logloss: list[float] = []
    # Líneas base: predecir siempre la media de la liga. Si el modelo no las
    # bate, sus ajustes por equipo no están aportando nada.
    corner_baseline_errors: list[float] = []
    card_baseline_errors: list[float] = []
    corner_baseline_logloss: list[float] = []
    card_baseline_logloss: list[float] = []
    fits = 0

    for block_start in range(start, len(results), step):
        block = results.iloc[block_start : block_start + step]
        train = results.iloc[:block_start]
        cutoff = block["date"].min()
        train = train[train["date"] < cutoff]
        if len(train) < min_train:
            continue

        try:
            goals_model = dc.fit_goals_model(train, reference_date=cutoff, xi=xi, ridge=ridge)
        except (ValueError, RuntimeError):
            continue
        corners_model = rates.fit_rate_model(
            train, "home_corners", "away_corners", "corners", reference_date=cutoff, xi=xi
        )
        cards_model = rates.fit_rate_model(
            train, "home_cards", "away_cards", "cards", reference_date=cutoff, xi=xi
        )
        fits += 1

        for row in block.itertuples():
            if row.home not in goals_model.attack or row.away not in goals_model.attack:
                continue

            matrix = goals_model.score_matrix(row.home, row.away)
            probabilities = dc.outcome_probabilities(matrix)
            actual = _actual_outcome(row.home_goals, row.away_goals)

            model_logloss.append(-math.log(max(probabilities[actual], EPSILON)))
            model_brier.append(
                sum(
                    (probabilities[key] - (1.0 if key == actual else 0.0)) ** 2
                    for key in ("home", "draw", "away")
                )
            )
            model_hits.append(int(max(probabilities, key=probabilities.get) == actual))

            market = _market_probabilities(pd.Series(row._asdict()))
            if market:
                market_logloss.append(-math.log(max(market[actual], EPSILON)))
                market_brier.append(
                    sum(
                        (market[key] - (1.0 if key == actual else 0.0)) ** 2
                        for key in ("home", "draw", "away")
                    )
                )
                market_hits.append(int(max(market, key=market.get) == actual))

            for model, observed, line, errors, loglosses, base_errors, base_loglosses in (
                (
                    corners_model,
                    row.home_corners + row.away_corners,
                    9.5,
                    corner_errors,
                    corner_over_logloss,
                    corner_baseline_errors,
                    corner_baseline_logloss,
                ),
                (
                    cards_model,
                    row.home_cards + row.away_cards,
                    4.5,
                    card_errors,
                    card_over_logloss,
                    card_baseline_errors,
                    card_baseline_logloss,
                ),
            ):
                if model is None or pd.isna(observed):
                    continue
                expected = sum(model.predict(row.home, row.away))
                errors.append(abs(expected - observed))
                probability = rates.prob_over(expected, line, model.dispersion)
                hit = observed > line
                loglosses.append(
                    -math.log(max(probability if hit else 1 - probability, EPSILON))
                )

                league_mean = model.league_home + model.league_away
                base_errors.append(abs(league_mean - observed))
                base_probability = rates.prob_over(league_mean, line, model.dispersion)
                base_loglosses.append(
                    -math.log(max(base_probability if hit else 1 - base_probability, EPSILON))
                )

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "half_life": half_life,
        "ridge": ridge,
        "matches": len(model_logloss),
        "fits": fits,
        "logloss": mean(model_logloss),
        "brier": mean(model_brier),
        "accuracy": mean(model_hits),
        "market_logloss": mean(market_logloss),
        "market_brier": mean(market_brier),
        "market_accuracy": mean(market_hits),
        "market_matches": len(market_logloss),
        "corners_mae": mean(corner_errors),
        "corners_logloss": mean(corner_over_logloss),
        "corners_mae_base": mean(corner_baseline_errors),
        "corners_logloss_base": mean(corner_baseline_logloss),
        "cards_mae": mean(card_errors),
        "cards_logloss": mean(card_over_logloss),
        "cards_mae_base": mean(card_baseline_errors),
        "cards_logloss_base": mean(card_baseline_logloss),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=int, default=380, help="partidos a evaluar")
    parser.add_argument("--step", type=int, default=20, help="cada cuántos partidos reentrenar")
    parser.add_argument("--half-lives", type=float, nargs="*", default=[400, 600, 900])
    parser.add_argument("--ridges", type=float, nargs="*", default=[0.08, 0.12, 0.2])
    args = parser.parse_args()

    results = loader.load_results()
    print(f"Histórico: {len(results)} partidos ({results['date'].min():%Y-%m-%d} "
          f"a {results['date'].max():%Y-%m-%d})")
    print(f"Evaluando los últimos {args.eval} partidos, reentrenando cada {args.step}.\n")

    rows = []
    for half_life in args.half_lives:
        for ridge in args.ridges:
            started = time.time()
            report = run_backtest(results, args.eval, args.step, half_life, ridge)
            report["seconds"] = round(time.time() - started, 1)
            rows.append(report)
            print(
                f"vida media {half_life:>5.0f}d | ridge {ridge:<5.2f} | "
                f"logloss {report['logloss']:.4f} | brier {report['brier']:.4f} | "
                f"acierto {report['accuracy']:.1%} | "
                f"córners MAE {report['corners_mae']:.2f} | "
                f"tarjetas MAE {report['cards_mae']:.2f} | "
                f"{report['seconds']}s"
            )

    best = min(rows, key=lambda row: row["logloss"])
    print("\n" + "=" * 78)
    print(f"MEJOR: vida media {best['half_life']:.0f} días, ridge {best['ridge']:.2f}")
    print(f"  modelo   logloss {best['logloss']:.4f}  brier {best['brier']:.4f}  "
          f"acierto {best['accuracy']:.1%}  ({best['matches']} partidos)")
    if best["market_logloss"]:
        print(f"  mercado  logloss {best['market_logloss']:.4f}  "
              f"brier {best['market_brier']:.4f}  acierto {best['market_accuracy']:.1%}  "
              f"({best['market_matches']} partidos con cuotas)")
        delta = best["logloss"] - best["market_logloss"]
        veredicto = "mejor que" if delta < 0 else "peor que"
        print(f"  El modelo es {veredicto} el mercado por {abs(delta):.4f} de logloss.")
    print(
        f"  córners  MAE {best['corners_mae']:.3f} (media liga {best['corners_mae_base']:.3f})"
        f"  logloss 9.5 {best['corners_logloss']:.4f} "
        f"(media liga {best['corners_logloss_base']:.4f})"
    )
    print(
        f"  tarjetas MAE {best['cards_mae']:.3f} (media liga {best['cards_mae_base']:.3f})"
        f"  logloss 4.5 {best['cards_logloss']:.4f} "
        f"(media liga {best['cards_logloss_base']:.4f})"
    )
    print("\nAplica el resultado en app/config.py: HALF_LIFE_DAYS y GOALS_RIDGE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
