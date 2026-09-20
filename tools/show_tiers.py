"""Muestra las tres apuestas por nivel de riesgo de los próximos partidos."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import config  # noqa: E402
from app.model import predictor  # noqa: E402


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    quiet = "--quiet" in sys.argv
    engine = predictor.get_engine()
    predictions = predictor.upcoming_predictions(engine, limit=limit)

    for p in [] if quiet else predictions:
        print(
            f"\n{p['date']:%d/%m} {p['time']}  {p['home']['name']} vs {p['away']['name']}"
        )
        for tier in p["tiers"]:
            label = config.BET_TIER_LABELS[tier["target"]]
            print(
                f"   ~{tier['target']:<4.1f} {label:<12s} {tier['selection']:<40s} "
                f"casa {tier['house_odds']:>5.2f}  justa {tier['odds']:>5.2f}  "
                f"({tier['prob']:.0%})  [{tier['market']}]"
            )

    # Comprobaciones: familias distintas y cuotas en orden creciente.
    problems = []
    for p in predictions:
        tiers = p["tiers"]
        if len(tiers) != len(config.BET_TIERS):
            problems.append(f"{p['match_id']}: solo {len(tiers)} apuestas")
            continue
        # Ninguna de las tres puede compartir mercado con otra, ni siquiera a
        # través de una pata de una combinada.
        used: set[str] = set()
        for tier in tiers:
            if used & set(tier["families"]):
                problems.append(f"{p['match_id']}: mercado repetido en {tier['selection']}")
            used |= set(tier["families"])
        odds = [t["house_odds"] for t in tiers]
        if odds != sorted(odds):
            problems.append(f"{p['match_id']}: cuotas desordenadas {odds}")
        for tier in tiers:
            if tier["house_odds"] >= tier["odds"]:
                problems.append(
                    f"{p['match_id']}: la casa paga más que la cuota justa en "
                    f"{tier['selection']}"
                )

    # Reparto de mercados y desvío respecto a la cuota objetivo.
    from collections import Counter

    families = Counter(t["market"] for p in predictions for t in p["tiers"])
    print("\nMercados usados:")
    for market, count in families.most_common():
        print(f"  {market:<18s} {count:>4d}")

    combos = [t for p in predictions for t in p["tiers"] if t["combo"]]
    print(f"\nCombinadas: {len(combos)} de {sum(len(p['tiers']) for p in predictions)} apuestas")
    if combos:
        gaps = [t["independent_odds"] / t["odds"] - 1 for t in combos]
        print(
            f"  corrección por correlación: mediana {np.median(gaps):+.1%}, "
            f"rango {min(gaps):+.1%} a {max(gaps):+.1%}"
        )

    margins = [t["odds"] / t["house_odds"] - 1 for p in predictions for t in p["tiers"]]
    print(f"\nMargen de casa aplicado: mediana {np.median(margins):.1%}, máx {max(margins):.1%}")

    print("\nDesvío del precio de casa respecto al objetivo:")
    for i, target in enumerate(config.BET_TIERS):
        odds = sorted(p["tiers"][i]["house_odds"] for p in predictions if len(p["tiers"]) == 3)
        worst = max(abs(o - target) for o in odds)
        median = odds[len(odds) // 2]
        print(f"  ~{target}: mediana {median:.2f}, peor desvío {worst:.2f}")

    print()
    if problems:
        print(f"{len(problems)} PROBLEMAS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{len(predictions)} partidos, todos con 3 apuestas de mercados distintos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
