"""Comprueba que las predicciones son internamente coherentes.

Recorre todos los partidos pendientes y verifica que las probabilidades suman
1, que las líneas son monótonas, que no hay valores imposibles y que ningún
campo que la interfaz espera llega vacío. Sirve de red de seguridad tras tocar
el modelo o los datos.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.model import predictor  # noqa: E402

TOLERANCE = 1e-6


def _json_default(value):
    """Solo las fechas pueden no ser serializables; lo demás es un error."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"tipo no serializable: {type(value).__name__}")


def check(prediction: dict) -> list[str]:
    problems: list[str] = []
    tag = prediction["match_id"]

    def fail(message: str) -> None:
        problems.append(f"{tag}: {message}")

    outcome = prediction["outcome"]
    total = sum(outcome.values())
    if abs(total - 1.0) > 1e-4:
        fail(f"las probabilidades 1X2 suman {total:.6f}")
    for key, value in outcome.items():
        if not 0.0 <= value <= 1.0:
            fail(f"probabilidad fuera de rango en {key}: {value}")

    double = prediction["double_chance"]
    expected = {
        "home_draw": outcome["home"] + outcome["draw"],
        "no_draw": outcome["home"] + outcome["away"],
        "draw_away": outcome["draw"] + outcome["away"],
    }
    for key, value in expected.items():
        if abs(double[key] - value) > 1e-6:
            fail(f"doble oportunidad {key} incoherente con el 1X2")

    margins = prediction["margins"]
    margin_total = sum(margins.values())
    if abs(margin_total - 1.0) > 1e-4:
        fail(f"los márgenes suman {margin_total:.6f}")
    if abs(margins["draw"] - outcome["draw"]) > 1e-6:
        fail("el empate de los márgenes no coincide con el del 1X2")

    goals = prediction["goals"]
    if goals["home"] <= 0 or goals["away"] <= 0:
        fail(f"goles esperados no positivos: {goals['home']}, {goals['away']}")
    if goals["total"] > 8:
        fail(f"goles totales esperados irreales: {goals['total']:.2f}")

    lines = prediction["goal_lines"]
    for previous, following in zip(lines, lines[1:]):
        if following["over"] > previous["over"] + TOLERANCE:
            fail(
                f"líneas de goles no monótonas: over {previous['line']}="
                f"{previous['over']:.4f} < over {following['line']}={following['over']:.4f}"
            )
    for row in lines:
        if abs(row["over"] + row["under"] - 1.0) > 1e-6:
            fail(f"línea de goles {row['line']} no suma 1")

    if not 0.0 <= prediction["btts"] <= 1.0:
        fail(f"BTTS fuera de rango: {prediction['btts']}")

    scorelines = prediction["scorelines"]
    if len(scorelines) < 3:
        fail("faltan marcadores probables")
    for previous, following in zip(scorelines, scorelines[1:]):
        if following["prob"] > previous["prob"] + TOLERANCE:
            fail("los marcadores probables no están ordenados")

    for name in predictor.SECONDARY_METRICS:
        block = prediction.get(name)
        if block is None:
            continue
        if block["home"] <= 0 or block["away"] <= 0:
            fail(f"{name}: valores esperados no positivos")
        if abs(block["home"] + block["away"] - block["total"]) > 1e-6:
            fail(f"{name}: el total no es la suma de los dos equipos")
        if not 1.0 <= block["dispersion"] <= 3.0:
            fail(f"{name}: dispersión fuera de rango ({block['dispersion']})")
        metric_lines = block["lines"]
        for previous, following in zip(metric_lines, metric_lines[1:]):
            if following["over"] > previous["over"] + TOLERANCE:
                fail(f"{name}: líneas no monótonas en {following['line']}")
        for row in metric_lines:
            if not 0.0 <= row["over"] <= 1.0:
                fail(f"{name}: probabilidad fuera de rango en {row['line']}")

    picks = prediction["picks"]
    if not picks:
        fail("sin pronósticos destacados")
    for pick in picks:
        if not 0.0 < pick["prob"] <= 1.0:
            fail(f"pick con probabilidad inválida: {pick}")
        if not pick["selection"]:
            fail("pick sin texto de selección")
    for previous, following in zip(picks, picks[1:]):
        if following["edge"] > previous["edge"] + TOLERANCE:
            fail("los picks no están ordenados por diferencia con la liga")

    tiers = prediction["tiers"]
    if len(tiers) != len(config.BET_TIERS):
        fail(f"se esperaban {len(config.BET_TIERS)} apuestas por riesgo, hay {len(tiers)}")
    used_markets: set[str] = set()
    for tier in tiers:
        if abs(tier["odds"] * tier["prob"] - 1.0) > 1e-6:
            fail(f"la cuota de '{tier['selection']}' no es la inversa de su probabilidad")
        # La casa siempre paga menos que la cuota justa: esa es su comisión.
        if tier["house_odds"] >= tier["odds"]:
            fail(f"la casa no se queda margen en '{tier['selection']}'")
        # El margen por selección no es plano: las casas cargan mucho más a
        # las opciones poco probables (medido en tools/measure_margin.py: 2,3%
        # a un favorito del 1X2, 23,9% a una opción por debajo del 10%). Aquí
        # solo se acota el disparate; la calibración fina se valida allí.
        margin = tier["odds"] / tier["house_odds"] - 1.0
        if margin > (0.45 if tier["combo"] else 0.25):
            fail(f"margen disparatado ({margin:.1%}) en '{tier['selection']}'")
        if used_markets & set(tier["families"]):
            fail(f"'{tier['selection']}' repite un mercado ya usado en otra apuesta")
        used_markets |= set(tier["families"])
        if tier["combo"]:
            if len(tier["legs"]) != 2:
                fail(f"combinada con {len(tier['legs'])} patas: {tier['selection']}")
            # La combinada nunca puede ser más probable que su pata más floja.
            if tier["prob"] > min(leg["prob"] for leg in tier["legs"]) + TOLERANCE:
                fail(f"combinada más probable que sus patas: {tier['selection']}")
    for previous, following in zip(tiers, tiers[1:]):
        if following["house_odds"] <= previous["house_odds"]:
            fail("las apuestas por riesgo no van de menor a mayor cuota")

    confidence = prediction["confidence"]
    if confidence["label"] not in {"Alta", "Media", "Baja"}:
        fail(f"etiqueta de confianza desconocida: {confidence['label']}")
    if confidence["stars"] not in {1, 2, 3}:
        fail(f"estrellas de confianza inválidas: {confidence['stars']}")

    if prediction["market"].get("available"):
        implied = prediction["market"]["implied"]
        implied_total = sum(implied.values())
        if abs(implied_total - 1.0) > 1e-4:
            fail(f"las probabilidades del mercado suman {implied_total:.6f}")

    # La interfaz usa los diccionarios tal cual, pero la API los serializa: un
    # set o un tipo de numpy pasan desapercibidos en la web y rompen el JSON.
    try:
        json.dumps(prediction, default=_json_default)
    except TypeError as error:
        fail(f"la ficha no se puede serializar a JSON: {error}")

    if not prediction["summary"]:
        fail("resumen vacío")
    for side in ("home", "away"):
        team = prediction[side]
        if not team["name"] or not team["code"] or len(team["code"]) != 3:
            fail(f"metadatos de equipo incompletos en {side}: {team}")

    return problems


def main() -> int:
    engine = predictor.get_engine()
    fixtures = predictor.upcoming_fixtures(engine)
    print(f"Partidos pendientes: {len(fixtures)}")

    problems: list[str] = []
    checked = 0
    missing_time = 0
    with_odds = 0

    for _, fixture in fixtures.iterrows():
        prediction = engine.predict(fixture, full=True)
        problems.extend(check(prediction))
        checked += 1
        if prediction["time"] in {"--:--", "", "nan"}:
            missing_time += 1
        if prediction["market"].get("available"):
            with_odds += 1

    print(f"Comprobados: {checked}")
    print(f"Con cuotas de mercado: {with_odds}")
    print(f"Sin horario asignado todavía: {missing_time}")

    equipos = sorted({f for f in fixtures["home"]} | {f for f in fixtures["away"]})
    print(f"Equipos en el calendario: {len(equipos)}")
    desconocidos = [t for t in equipos if t not in engine.goals_model.attack]
    if desconocidos:
        problems.append(f"equipos sin datos históricos: {desconocidos}")

    from app.data.teams import is_known

    sin_metadatos = [team for team in equipos if not is_known(team)]
    if sin_metadatos:
        print(f"AVISO: equipos sin nombre ni color configurado: {sin_metadatos}")

    if problems:
        print(f"\n{len(problems)} PROBLEMAS:")
        for problem in problems[:40]:
            print(f"  - {problem}")
        return 1

    print("\nTodo coherente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
