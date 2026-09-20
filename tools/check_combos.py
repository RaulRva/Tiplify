"""Comprueba la simulación conjunta que da precio a las combinadas.

Verifica dos cosas:

1. Que las probabilidades simuladas de las apuestas simples coinciden con las
   analíticas. Si no, la cópula estaría deformando los mercados individuales.
2. Que la cuota de las combinadas se separa del producto de las cuotas, que es
   justo lo que aporta el modelo conjunto frente a multiplicar a ciegas.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import config  # noqa: E402
from app.model import predictor  # noqa: E402


def main() -> int:
    engine = predictor.get_engine()

    names = ("goles",) + predictor.JOINT_METRICS
    print("Correlación entre residuos (cópula gaussiana):")
    print("            " + "".join(f"{n[:9]:>11s}" for n in names))
    for i, row_name in enumerate(names):
        cells = "".join(f"{engine.joint_model.correlation[i, j]:>11.3f}" for j in range(len(names)))
        print(f"{row_name:<12s}{cells}")

    fixtures = predictor.upcoming_fixtures(engine).head(40)
    worst_marginal = 0.0
    gaps: list[tuple[float, str, str]] = []

    for _, fixture in fixtures.iterrows():
        prediction = engine.predict(fixture, full=False)
        matrix = engine.goals_model.score_matrix(
            str(fixture["home"]),
            str(fixture["away"]),
            prediction["goals"]["home"],
            prediction["goals"]["away"],
        )
        seed = zlib.crc32(prediction["match_id"].encode("utf-8"))
        simulation = engine.joint_model.simulate(matrix, seed=seed)

        # 1. Marginales: simulado contra analítico.
        for candidate in engine._bet_candidates(prediction, simulation):
            simulated = float(np.count_nonzero(candidate["mask"])) / simulation.draws
            worst_marginal = max(worst_marginal, abs(simulated - candidate["prob"]))

        # 2. Combinadas: cuota real contra producto de cuotas.
        singles = engine._bet_candidates(prediction, simulation)
        for combo in engine._combined_candidates(singles, simulation):
            gap = combo["independent_odds"] / combo["odds"] - 1.0
            gaps.append((gap, prediction["match_id"], combo["selection"]))

    print(f"\nMarginales simuladas vs analíticas: peor desvío {worst_marginal:.4f}")

    if not gaps:
        print("No salió ninguna combinada en estos partidos.")
        return 0

    gaps.sort()
    print(f"\nCombinadas encontradas: {len(gaps)}")
    print(f"Corrección por correlación: mediana {np.median([g[0] for g in gaps]):+.1%}")
    print("\nMayor corrección a la baja (patas que van de la mano):")
    for gap, match_id, selection in gaps[-3:][::-1]:
        print(f"  {gap:+7.1%}  {selection}  [{match_id}]")
    print("\nMayor corrección al alza (patas que se estorban):")
    for gap, match_id, selection in gaps[:3]:
        print(f"  {gap:+7.1%}  {selection}  [{match_id}]")

    if worst_marginal > 0.02:
        print("\nPROBLEMA: las marginales simuladas no cuadran con las analíticas.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
