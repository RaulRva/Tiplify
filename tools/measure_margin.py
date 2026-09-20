"""Mide el margen de las casas de apuestas y cómo lo reparten.

Una casa no publica la cuota justa: le quita un margen. Si la quitara
proporcionalmente, bastaría con dividir la cuota justa entre (1 + margen).
Pero es sabido que cargan más a los no favoritos, así que aquí se comprueba
con datos antes de elegir la fórmula.

El contraste: se normalizan las cuotas de cada partido para que sumen 1
(reparto proporcional) y se compara la probabilidad resultante con la
frecuencia real de ese resultado. Si los no favoritos ganan menos veces de lo
que dice su probabilidad normalizada, es que llevaban margen de más.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from app.data import loader  # noqa: E402


def overround(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return sum(1.0 / frame[column] for column in columns)


def power_exponent(implied: np.ndarray) -> float:
    """Exponente k tal que sum(p_i ** k) = 1, con p_i = 1/cuota."""

    def error(k: float) -> float:
        return float(np.sum(implied**k) - 1.0)

    try:
        return brentq(error, 0.5, 2.0)
    except ValueError:
        return float("nan")


def main() -> int:
    results = loader.load_results()

    columns = ["odds_home", "odds_draw", "odds_away"]
    frame = results.dropna(subset=columns).copy()
    print(f"Partidos con cuotas 1X2: {len(frame)}")

    margin = overround(frame, columns)
    print(f"Margen 1X2: media {100 * (margin.mean() - 1):.2f}%, mediana {100 * (margin.median() - 1):.2f}%")

    goals_columns = ["odds_over25", "odds_under25"]
    goals = results.dropna(subset=goals_columns)
    goals_margin = overround(goals, goals_columns)
    print(
        f"Margen más/menos 2.5: media {100 * (goals_margin.mean() - 1):.2f}%, "
        f"mediana {100 * (goals_margin.median() - 1):.2f}%"
    )

    # --- ¿Proporcional o cargado a los no favoritos? ------------------------
    implied = np.column_stack([1.0 / frame[column].to_numpy() for column in columns])
    proportional = implied / implied.sum(axis=1, keepdims=True)

    outcome = np.column_stack(
        [
            (frame["home_goals"] > frame["away_goals"]).to_numpy(),
            (frame["home_goals"] == frame["away_goals"]).to_numpy(),
            (frame["home_goals"] < frame["away_goals"]).to_numpy(),
        ]
    ).astype(float)

    flat_probability = proportional.ravel()
    flat_outcome = outcome.ravel()
    edges = [0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 1.0]

    print("\nReparto proporcional: probabilidad estimada vs frecuencia real")
    print("  rango        n     estimada    real     dif")
    for low, high in zip(edges, edges[1:]):
        mask = (flat_probability >= low) & (flat_probability < high)
        if mask.sum() < 50:
            continue
        estimated = flat_probability[mask].mean()
        actual = flat_outcome[mask].mean()
        print(
            f"  {low:.2f}-{high:.2f}  {mask.sum():>5d}   {estimated:>7.1%}  {actual:>7.1%}  "
            f"{actual - estimated:>+7.1%}"
        )

    # Exponente del método "power": si sale claramente por debajo de 1, la
    # casa carga más margen sobre los no favoritos.
    exponents = np.array([power_exponent(row) for row in implied])
    exponents = exponents[np.isfinite(exponents)]
    print(f"\nExponente del método power: mediana {np.median(exponents):.4f}")
    print("  (1.00 = margen repartido proporcionalmente; <1 = más margen a los no favoritos)")

    # --- ¿Cuánto margen lleva cada opción según su probabilidad? ------------
    # La app estima el precio de casa con el método de potencia, que carga más
    # a las opciones poco probables. Aquí se comprueba contra cuotas reales
    # cuánto margen lleva de verdad cada opción según lo probable que sea.
    print("\nMargen real por opción, según su probabilidad justa")
    print("  (justa = la implícita tras quitar el margen con el método de potencia)")
    for label, columns in (
        ("1X2", ["odds_home", "odds_draw", "odds_away"]),
        ("Más/menos 2.5", goals_columns),
    ):
        subset = results.dropna(subset=columns)
        raw = np.column_stack([1.0 / subset[column].to_numpy() for column in columns])
        fair_rows, margin_rows = [], []
        for row in raw:
            k = power_exponent(row)
            if not np.isfinite(k):
                continue
            fair = row**k
            fair_rows.append(fair)
            margin_rows.append(row / fair - 1.0)
        fair_flat = np.concatenate(fair_rows)
        margin_flat = np.concatenate(margin_rows)

        print(f"\n  {label} ({len(fair_rows)} partidos)")
        print("    prob justa      n    margen de esa opción")
        for low, high in zip(edges, edges[1:]):
            mask = (fair_flat >= low) & (fair_flat < high)
            if mask.sum() < 50:
                continue
            print(
                f"    {low:.2f}-{high:.2f}   {mask.sum():>5d}   {margin_flat[mask].mean():>7.1%}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
