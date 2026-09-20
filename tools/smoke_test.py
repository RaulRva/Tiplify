"""Prueba rápida del motor: entrena, lista próximos partidos e imprime una ficha."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import predictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    engine = predictor.get_engine()
    gm = engine.goals_model
    print(f"\nPartidos de entrenamiento: {gm.matches_used}")
    print(f"Equipos: {len(gm.teams)}")
    print(f"Ventaja de local: {gm.home_advantage:.3f} | rho: {gm.rho:.4f}")
    print(f"Media liga: {gm.league_home_goals:.2f} - {gm.league_away_goals:.2f}")
    print(f"Modelo xG: {'sí' if engine.xg_model else 'no'}", end="")
    if engine.xg_model:
        print(f" ({engine.xg_model.matches_used} partidos)")
    else:
        print()

    for name, model in engine.secondary.items():
        if model is None:
            print(f"  {name:14s} sin datos")
        else:
            print(
                f"  {name:14s} media {model.league_home:.2f}/{model.league_away:.2f}"
                f" dispersión {model.dispersion:.2f} ({model.matches_used} partidos)"
            )

    print("\nTop 5 ataques:")
    ranked = sorted(gm.attack.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for team, value in ranked:
        print(f"  {team:14s} {value:+.3f}")

    predictions = predictor.upcoming_predictions(engine)
    print(f"\nPróximos partidos encontrados: {len(predictions)}")
    for p in predictions[:14]:
        o = p["outcome"]
        print(
            f"  {p['date']:%d/%m} {p['time']} {p['home']['code']}-{p['away']['code']} "
            f"{o['home']:>4.0%}/{o['draw']:>4.0%}/{o['away']:>4.0%} "
            f"goles {p['goals']['total']:.2f} "
            f"córners {p['corners']['total']:.1f} "
            f"tarjetas {p['cards']['total']:.1f} "
            f"| conf {p['confidence']['label']:<6s} "
            f"| {p['picks'][0]['selection'][:32]:<32s} {p['picks'][0]['prob']:.0%}"
        )

    if not predictions:
        print("\nNo hay partidos próximos en el calendario ahora mismo.")
        return 0

    p = predictions[0]
    print("\n" + "=" * 70)
    print(f"FICHA: {p['home']['name']} vs {p['away']['name']}  ({p['match_id']})")
    print("=" * 70)
    print(p["summary"])
    print(f"\nConfianza: {p['confidence']['label']} ({p['confidence']['score']:.0%})")
    print(f"BTTS: {p['btts']:.1%}")
    for row in p["goal_lines"]:
        print(f"  goles {row['line']:g}: over {row['over']:.1%} / under {row['under']:.1%}")
    for name in ("corners", "cards", "shots", "shots_target"):
        block = p.get(name)
        if not block:
            continue
        print(f"\n{name}: {block['home']:.2f} - {block['away']:.2f} (total {block['total']:.2f})")
        for row in block["lines"]:
            print(f"  {row['line']:g}: over {row['over']:.1%}")
    print("\nMarcadores más probables:")
    for row in p["scorelines"]:
        print(f"  {row['home']}-{row['away']}: {row['prob']:.1%}")
    print("\nPicks:")
    for pick in p["picks"]:
        print(f"  {pick['market']:22s} {pick['selection']:34s} {pick['prob']:.1%}")
    print("\nMercado:", p["market"])
    print("\nH2H:", p["h2h"]["count"], "partidos")
    for m in p["h2h"]["matches"][:5]:
        print(f"  {m['date']:%d/%m/%Y} {m['home']} {m['score']} {m['away']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
