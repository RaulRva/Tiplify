"""Motor de predicción: une datos, modelos y features en una ficha por partido."""

from __future__ import annotations

import logging
import math
import threading
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config
from ..data import loader
from ..data.teams import get_team
from . import dixon_coles as dc
from . import features as ft
from . import joint
from . import pricing
from . import rates

log = logging.getLogger(__name__)

# Métricas secundarias: nombre interno -> (columna local, columna visitante, líneas)
SECONDARY_METRICS = {
    "corners": ("home_corners", "away_corners", config.CORNER_LINES),
    "cards": ("home_cards", "away_cards", config.CARD_LINES),
    "shots": ("home_shots", "away_shots", config.SHOT_LINES),
    "shots_target": ("home_shots_target", "away_shots_target", config.SHOT_TARGET_LINES),
    "fouls": ("home_fouls", "away_fouls", []),
}

# Nombre visible de cada métrica secundaria en las apuestas.
METRIC_LABELS = {
    "corners": ("Córners", "córners"),
    "cards": ("Tarjetas", "tarjetas"),
    "shots": ("Tiros", "tiros"),
    "shots_target": ("Tiros a puerta", "tiros a puerta"),
}

# Métricas que entran en la simulación conjunta con el marcador.
JOINT_METRICS = tuple(METRIC_LABELS)


def _team_payload(key: str, league_slug: str | None = None) -> dict:
    team = get_team(key, league_slug)
    return {
        "key": team.key,
        "name": team.name,
        "code": team.code,
        "color": team.color,
        "accent": team.accent,
    }


def _remove_margin(odds: dict[str, float | None]) -> dict[str, float] | None:
    """Convierte cuotas en probabilidades quitando el margen de la casa."""
    values = {}
    for key, value in odds.items():
        if value is None or pd.isna(value) or float(value) <= 1.0:
            return None
        values[key] = 1.0 / float(value)
    total = sum(values.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in values.items()}


@dataclass
class Engine:
    league: config.League
    results: pd.DataFrame
    view: pd.DataFrame
    goals_model: dc.GoalsModel
    xg_model: rates.RateModel | None
    secondary: dict[str, rates.RateModel | None]
    table: pd.DataFrame
    base_rates: dict[str, float]
    joint_model: joint.JointModel
    fitted_at: datetime
    reference_date: pd.Timestamp
    _cache: dict[tuple[str, bool], dict] = field(default_factory=dict)

    # --- Goles -------------------------------------------------------------
    def _blended_goals(self, home: str, away: str) -> tuple[float, float, float]:
        """Mezcla el Dixon-Coles de goles con el modelo de xG. Devuelve pesos."""
        lambda_home, lambda_away = self.goals_model.expected_goals(home, away)
        if self.xg_model is None:
            return lambda_home, lambda_away, 0.0

        xg_home, xg_away = self.xg_model.predict(home, away)
        weight = config.XG_BLEND_MAX_WEIGHT * min(
            1.0, self.xg_model.matches_used / config.XG_BLEND_FULL_AT
        )
        blended_home = (1 - weight) * lambda_home + weight * xg_home
        blended_away = (1 - weight) * lambda_away + weight * xg_away
        return blended_home, blended_away, weight

    def _secondary_payload(self, name: str, home: str, away: str) -> dict | None:
        model = self.secondary.get(name)
        if model is None:
            return None
        expected_home, expected_away = model.predict(home, away)
        total = expected_home + expected_away
        lines = SECONDARY_METRICS[name][2]
        return {
            "home": expected_home,
            "away": expected_away,
            "total": total,
            "dispersion": model.dispersion,
            "lines": rates.line_table(total, lines, model.dispersion),
            "league_average": model.league_home + model.league_away,
            "profile": {
                "home": model.team_profile(home),
                "away": model.team_profile(away),
            },
        }

    # --- Confianza ---------------------------------------------------------
    def _confidence(self, home: str, away: str, outcome: dict[str, float]) -> dict:
        """Cuán marcado está el pronóstico, según el resultado más probable.

        Un 1X2 reparte entre tres opciones, así que un 55% ya es un favorito
        claro. Si alguno de los dos equipos tiene poca historia en los datos,
        la confianza se rebaja aunque las probabilidades salgan marcadas.
        """
        sample = min(
            self.goals_model.weighted_matches.get(home, 0.0),
            self.goals_model.weighted_matches.get(away, 0.0),
        )
        top = max(outcome.values())

        if top >= 0.55:
            label, stars = "Alta", 3
        elif top >= 0.42:
            label, stars = "Media", 2
        else:
            label, stars = "Baja", 1

        if sample < 8.0 and stars > 1:
            label, stars = "Media" if stars == 3 else "Baja", stars - 1

        return {"score": top, "label": label, "stars": stars, "sample": sample}

    # --- Ficha completa ----------------------------------------------------
    def predict(self, fixture: pd.Series, full: bool = True) -> dict:
        """Ficha de un partido.

        Con `full=False` se omiten forma reciente, historial directo y
        clasificación: es lo que necesita el listado y evita recalcular
        agregados por equipo para cada uno de los partidos de la temporada.
        """
        home = str(fixture["home"])
        away = str(fixture["away"])
        match_id = str(fixture.get("match_id") or loader.make_match_id(fixture["date"], home, away))

        cache_key = (match_id, full)
        if cache_key in self._cache:
            return self._cache[cache_key]

        lambda_home, lambda_away, xg_weight = self._blended_goals(home, away)
        matrix = self.goals_model.score_matrix(home, away, lambda_home, lambda_away)
        outcome = dc.outcome_probabilities(matrix)
        totals = dc.total_goals_distribution(matrix)

        goal_lines = [
            {
                "line": line,
                "over": dc.prob_over(totals, line),
                "under": 1.0 - dc.prob_over(totals, line),
            }
            for line in config.GOAL_LINES
        ]

        prediction: dict = {
            "match_id": match_id,
            "date": fixture["date"],
            "time": str(fixture.get("time") or "--:--"),
            "round": str(fixture.get("round") or ""),
            "league": self.league.name,
            "league_slug": self.league.slug,
            "home": _team_payload(home, self.league.slug),
            "away": _team_payload(away, self.league.slug),
            "goals": {
                "home": lambda_home,
                "away": lambda_away,
                "total": lambda_home + lambda_away,
                "xg_weight": xg_weight,
            },
            "outcome": outcome,
            "double_chance": {
                "home_draw": outcome["home"] + outcome["draw"],
                "no_draw": outcome["home"] + outcome["away"],
                "draw_away": outcome["draw"] + outcome["away"],
            },
            "goal_lines": goal_lines,
            "over25": next(row["over"] for row in goal_lines if row["line"] == 2.5),
            "btts": dc.prob_btts(matrix),
            "scorelines": dc.top_scorelines(matrix, limit=6),
            "margins": dc.handicap_probabilities(matrix),
            "strength": {
                "home": self.goals_model.team_strength(home),
                "away": self.goals_model.team_strength(away),
            },
        }

        for name in SECONDARY_METRICS:
            prediction[name] = self._secondary_payload(name, home, away)

        prediction["market"] = self._market(fixture, outcome, goal_lines)
        prediction["confidence"] = self._confidence(home, away, outcome)

        if full:
            slug = self.league.slug
            prediction["form"] = {
                "home": ft.recent_form(self.view, home, league=slug),
                "away": ft.recent_form(self.view, away, league=slug),
                "home_at_home": ft.recent_form(self.view, home, venue="home", league=slug),
                "away_at_away": ft.recent_form(self.view, away, venue="away", league=slug),
            }
            prediction["h2h"] = ft.head_to_head(self.results, home, away, league=slug)
            prediction["standings"] = {
                "home": self._standing_row(home),
                "away": self._standing_row(away),
            }
        else:
            prediction["form"] = {"home": self._mini_form(home), "away": self._mini_form(away)}
            prediction["h2h"] = {"count": 0, "matches": []}
            prediction["standings"] = {"home": None, "away": None}

        prediction["picks"] = self._picks(prediction)
        prediction["tiers"] = self._tiered_bets(prediction, matrix)
        prediction["summary"] = self._summary(prediction)

        self._cache[cache_key] = prediction
        return prediction

    def _mini_form(self, team: str) -> dict:
        """Solo la racha W/D/L, suficiente para el listado."""
        subset = self.view[self.view["team"] == team].tail(config.FORM_MATCHES)
        streak = list(subset["result"])[::-1]
        return {
            "matches": len(streak),
            "streak": streak,
            "wins": streak.count("W"),
            "draws": streak.count("D"),
            "losses": streak.count("L"),
        }

    def _standing_row(self, team: str) -> dict | None:
        if self.table.empty:
            return None
        row = self.table[self.table["team"] == team]
        if row.empty:
            return None
        record = row.iloc[0]
        return {
            "pos": int(record["pos"]),
            "played": int(record["partidos"]),
            "points": int(record["puntos"]),
            "wins": int(record["ganados"]),
            "draws": int(record["empatados"]),
            "losses": int(record["perdidos"]),
            "goals_for": int(record["gf"]),
            "goals_against": int(record["gc"]),
        }

    def _market(self, fixture: pd.Series, outcome: dict, goal_lines: list[dict]) -> dict:
        def value(column: str) -> float | None:
            raw = fixture.get(column)
            if raw is None or pd.isna(raw):
                return None
            return float(raw)

        odds_1x2 = {
            "home": value("odds_home") or value("odds_home_b365"),
            "draw": value("odds_draw") or value("odds_draw_b365"),
            "away": value("odds_away") or value("odds_away_b365"),
        }
        implied = _remove_margin(odds_1x2)
        market: dict = {"available": implied is not None, "odds": odds_1x2}
        if implied:
            market["implied"] = implied
            market["edges"] = {key: outcome[key] - implied[key] for key in implied}

        over_odds = value("odds_over25")
        under_odds = value("odds_under25")
        implied_goals = _remove_margin({"over": over_odds, "under": under_odds})
        if implied_goals:
            model_over = next(row["over"] for row in goal_lines if row["line"] == 2.5)
            market["goals"] = {
                "implied_over25": implied_goals["over"],
                "model_over25": model_over,
                "edge": model_over - implied_goals["over"],
                "odds_over": over_odds,
                "odds_under": under_odds,
            }
        return market

    def _picks(self, prediction: dict) -> list[dict]:
        """Selecciones destacadas, ordenadas por lo que se salen de lo normal.

        Ordenarlas por probabilidad bruta no sirve: "gana uno de los dos"
        siempre ronda el 74% y no dice nada, porque en LaLiga ya pasa el 74% de
        las veces. Aquí cada selección se compara con su frecuencia histórica en
        la liga, así que arriba queda lo que de verdad distingue a este partido.
        """
        outcome = prediction["outcome"]
        home_name = prediction["home"]["name"]
        away_name = prediction["away"]["name"]
        base = self.base_rates

        candidates: list[dict] = []

        def add(market: str, selection: str, prob: float, base_key: str) -> None:
            reference = base.get(base_key, 0.5)
            candidates.append(
                {
                    "market": market,
                    "selection": selection,
                    "prob": prob,
                    "base": reference,
                    "edge": prob - reference,
                }
            )

        best_1x2 = max(outcome, key=outcome.get)
        labels = {"home": f"Gana {home_name}", "draw": "Empate", "away": f"Gana {away_name}"}
        add("Resultado (1X2)", labels[best_1x2], outcome[best_1x2], f"outcome_{best_1x2}")

        double = prediction["double_chance"]
        best_double = max(double, key=double.get)
        double_labels = {
            "home_draw": f"{home_name} o empate",
            "no_draw": "Gana uno de los dos",
            "draw_away": f"Empate o {away_name}",
        }
        add(
            "Doble oportunidad",
            double_labels[best_double],
            double[best_double],
            f"double_{best_double}",
        )

        over25 = prediction["over25"]
        if over25 >= 0.5:
            add("Goles", "Más de 2.5 goles", over25, "goals_over")
        else:
            add("Goles", "Menos de 2.5 goles", 1 - over25, "goals_under")

        btts = prediction["btts"]
        if btts >= 0.5:
            add("Ambos marcan", "Marcan los dos equipos", btts, "btts_yes")
        else:
            add("Ambos marcan", "No marcan los dos equipos", 1 - btts, "btts_no")

        for name, label, unit in (
            ("corners", "Córners", "córners"),
            ("cards", "Tarjetas", "tarjetas"),
        ):
            block = prediction.get(name)
            if not block or not block["lines"]:
                continue
            # Sobre la línea principal del mercado, no sobre la más extrema.
            main = config.MAIN_LINES[name]
            row = next((item for item in block["lines"] if item["line"] == main), None)
            if row is None:
                continue
            if row["over"] >= row["under"]:
                add(label, f"Más de {main:g} {unit}", row["over"], f"{name}_over")
            else:
                add(label, f"Menos de {main:g} {unit}", row["under"], f"{name}_under")

        candidates.sort(key=lambda item: item["edge"], reverse=True)
        return candidates

    # --- Tres apuestas por nivel de riesgo ---------------------------------
    def _bet_candidates(self, prediction: dict, simulation: joint.Simulation) -> list[dict]:
        """Todas las selecciones simples, con su cuota justa y su máscara.

        La `familia` agrupa mercados que se pisan entre sí. Todo lo que depende
        del resultado (1X2, doble oportunidad y hándicap) va en una sola
        familia: "empate" y "gana uno de los dos" son complementarios, así que
        recomendarlos a la vez sería absurdo.

        La máscara dice en cuáles de los partidos simulados gana la apuesta.
        Sirve para calcular combinadas: la probabilidad conjunta es la
        frecuencia con la que ganan las dos patas a la vez.
        """
        home = prediction["home"]["name"]
        away = prediction["away"]["name"]
        outcome = prediction["outcome"]
        double = prediction["double_chance"]
        margins = prediction["margins"]
        difference = simulation.home_goals - simulation.away_goals
        total = simulation.total_goals

        candidates: list[dict] = []

        # Las tres del 1X2 se reparten el margen entre sí; el resto de
        # selecciones van contra su contraria.
        three_way = (outcome["home"], outcome["draw"], outcome["away"])

        def add(
            family: str,
            market: str,
            selection: str,
            probability: float,
            mask: np.ndarray | None,
            siblings: tuple[float, ...] | None = None,
        ) -> None:
            # Fuera de este rango las cuotas dejan de ser útiles como apuesta.
            if mask is None or not 0.05 < probability < 0.92:
                return
            group = siblings or (probability, 1.0 - probability)
            margin = pricing.margin_for(family)
            house = pricing.house_probability(probability, group, margin)
            candidates.append(
                {
                    "family": family,
                    "families": frozenset({family}),
                    "market": market,
                    "selection": selection,
                    "prob": probability,
                    "odds": 1.0 / probability,
                    "house_odds": 1.0 / house,
                    # Cuánto encoge la casa la probabilidad. Sirve para
                    # acumular el margen al combinar dos patas.
                    "margin_factor": house / probability,
                    "combo": False,
                    "legs": [],
                    "penalty": config.BET_FAMILY_PENALTY.get(family, 0.1),
                    "mask": mask,
                }
            )

        add("resultado", "Resultado", f"Gana {home}", outcome["home"], difference > 0, three_way)
        add("resultado", "Resultado", "Empate", outcome["draw"], difference == 0, three_way)
        add("resultado", "Resultado", f"Gana {away}", outcome["away"], difference < 0, three_way)
        add(
            "resultado", "Doble oportunidad", f"{home} o empate",
            double["home_draw"], difference >= 0,
        )
        add(
            "resultado", "Doble oportunidad", f"Empate o {away}",
            double["draw_away"], difference <= 0,
        )
        add(
            "resultado", "Doble oportunidad", "Gana uno de los dos",
            double["no_draw"], difference != 0,
        )
        add(
            "resultado", "Hándicap", f"{home} gana por 2 o más",
            margins["home_by_2plus"], difference >= 2,
        )
        add(
            "resultado", "Hándicap", f"{away} gana por 2 o más",
            margins["away_by_2plus"], difference <= -2,
        )

        for row in prediction["goal_lines"]:
            over = total > row["line"]
            add("goles", "Goles", f"Más de {row['line']:g} goles", row["over"], over)
            add("goles", "Goles", f"Menos de {row['line']:g} goles", row["under"], ~over)

        both_score = (simulation.home_goals > 0) & (simulation.away_goals > 0)
        add("ambos", "Ambos marcan", "Marcan los dos equipos", prediction["btts"], both_score)
        add("ambos", "Ambos marcan", "No marcan los dos equipos", 1 - prediction["btts"], ~both_score)

        for name, (label, unit) in METRIC_LABELS.items():
            block = prediction.get(name)
            if not block:
                continue
            for row in block["lines"]:
                over = simulation.metric_over(name, row["under"])
                add(name, label, f"Más de {row['line']:g} {unit}", row["over"], over)
                add(
                    name, label, f"Menos de {row['line']:g} {unit}",
                    row["under"], None if over is None else ~over,
                )

        return candidates

    def _combined_candidates(
        self, singles: list[dict], simulation: joint.Simulation
    ) -> list[dict]:
        """Combinadas de dos patas, con la cuota que corrige la correlación.

        El producto de las cuotas solo sirve como filtro rápido para no
        evaluar las miles de parejas posibles. La cuota que se publica sale de
        contar en cuántos partidos simulados ganan las dos patas a la vez, así
        que recoge que "gana el Barça" y "más de 2.5 goles" suelen ir juntas.
        """
        low, high = config.COMBO_ODDS_RANGE
        targets = config.BET_TIERS

        pairs: list[tuple[float, dict, dict]] = []
        for i, first in enumerate(singles):
            for second in singles[i + 1 :]:
                if first["families"] & second["families"]:
                    continue
                product = first["house_odds"] * second["house_odds"]
                if not low <= product <= high:
                    continue
                distance = min(abs(math.log(product / target)) for target in targets)
                pairs.append((distance, first, second))

        pairs.sort(key=lambda item: item[0])

        combos: list[dict] = []
        for _distance, first, second in pairs[: config.COMBO_MAX_EVALUATED]:
            probability = float(np.count_nonzero(first["mask"] & second["mask"])) / simulation.draws
            if not 0.05 < probability < 0.92:
                continue
            # Si una pata implica a la otra no hay combinada que valga: "más de
            # 1.5 goles + marcan los dos" es exactamente "marcan los dos", y
            # venderlo como combinada engaña sobre el riesgo que se asume.
            if probability > config.COMBO_REDUNDANT_RATIO * min(first["prob"], second["prob"]):
                continue
            # La pata del mercado más relevante va primero.
            legs = sorted([first, second], key=lambda item: item["penalty"])
            house = pricing.combo_house_odds(
                probability, (first["margin_factor"], second["margin_factor"])
            )
            independent = first["house_odds"] * second["house_odds"]
            combos.append(
                {
                    "family": "combinada",
                    "families": first["families"] | second["families"],
                    "market": " + ".join(leg["market"] for leg in legs),
                    "selection": " + ".join(leg["selection"] for leg in legs),
                    "prob": probability,
                    "odds": 1.0 / probability,
                    "house_odds": house,
                    "combo": True,
                    "legs": [
                        {
                            "market": leg["market"],
                            "selection": leg["selection"],
                            "prob": leg["prob"],
                            "odds": leg["odds"],
                            "house_odds": leg["house_odds"],
                        }
                        for leg in legs
                    ],
                    "independent_odds": independent,
                    # Cuánto se aleja la cuota real de multiplicar las patas.
                    # Positivo = las dos cosas suelen pasar juntas. El margen
                    # se cancela en la división, así que mide solo correlación.
                    "correlation_gap": independent / house - 1.0,
                    # La pata más rara manda: una combinada es tan poco
                    # apetecible como su selección más oscura.
                    "penalty": max(first["penalty"], second["penalty"]) + config.COMBO_PENALTY,
                    "mask": None,
                }
            )
        return combos

    @staticmethod
    def _bet_cost(item: dict, target: float) -> float:
        """Lo lejos que queda una apuesta de un objetivo, más su penalización.

        El objetivo se mide sobre el **precio de casa**, no sobre la cuota
        justa: así el 1.5 que muestra la app es el que se ve en la casa de
        apuestas, no uno que nadie paga.

        La distancia va en escala logarítmica para que acercarse a 3 cuente lo
        mismo que acercarse a 1.5. La penalización existe porque cualquier
        métrica puede clavar la cuota eligiendo la línea adecuada: sin ella
        saldrían siempre los mercados más oscuros en lugar del resultado.
        """
        return abs(math.log(item["house_odds"] / target)) + item["penalty"]

    def _tiered_bets(self, prediction: dict, matrix: np.ndarray) -> list[dict]:
        """Una apuesta por nivel de riesgo, simple o combinada.

        Las tres no pueden compartir ningún mercado, ni siquiera a través de
        una pata de una combinada, y sus cuotas van de menor a mayor.
        """
        seed = zlib.crc32(prediction["match_id"].encode("utf-8"))
        simulation = self.joint_model.simulate(matrix, seed=seed)

        singles = self._bet_candidates(prediction, simulation)
        if not singles:
            return []
        pool = singles + self._combined_candidates(singles, simulation)

        targets = config.BET_TIERS
        shortlists = [
            sorted(pool, key=lambda item: self._bet_cost(item, target))[: config.TIER_SHORTLIST]
            for target in targets
        ]

        best: list[dict] | None = None
        best_cost = math.inf
        for first in shortlists[0]:
            cost_first = self._bet_cost(first, targets[0])
            if cost_first >= best_cost:
                break
            for second in shortlists[1]:
                if (
                    second["house_odds"] <= first["house_odds"]
                    or second["families"] & first["families"]
                ):
                    continue
                cost_second = cost_first + self._bet_cost(second, targets[1])
                if cost_second >= best_cost:
                    continue
                used = first["families"] | second["families"]
                for third in shortlists[2]:
                    if third["house_odds"] <= second["house_odds"] or third["families"] & used:
                        continue
                    total = cost_second + self._bet_cost(third, targets[2])
                    if total < best_cost:
                        best_cost, best = total, [first, second, third]

        if best is None:
            return []

        return [
            {key: value for key, value in item.items() if key != "mask"}
            # `families` se usa como conjunto al elegir el trío, pero sale como
            # lista para que la respuesta de la API sea JSON válido.
            | {"target": target, "families": sorted(item["families"])}
            for item, target in zip(best, targets)
        ]

    def _summary(self, prediction: dict) -> str:
        home = prediction["home"]["name"]
        away = prediction["away"]["name"]
        outcome = prediction["outcome"]
        goals = prediction["goals"]
        form_home = prediction["form"]["home"]
        form_away = prediction["form"]["away"]

        favourite, probability = max(outcome.items(), key=lambda item: item[1])
        if favourite == "home":
            head = f"{home} parte como favorito ({probability:.0%})"
        elif favourite == "away":
            head = f"{away} parte como favorito ({probability:.0%})"
        else:
            head = f"Partido muy igualado, con el empate como resultado más probable ({probability:.0%})"

        parts = [
            f"{head}. El modelo espera {goals['home']:.1f}-{goals['away']:.1f} "
            f"({goals['total']:.1f} goles en total)."
        ]

        if form_home["matches"] and form_away["matches"]:
            parts.append(
                f"Forma reciente: {home} {form_home['wins']}V-{form_home['draws']}E-{form_home['losses']}D "
                f"y {away} {form_away['wins']}V-{form_away['draws']}E-{form_away['losses']}D "
                f"en sus últimos {config.FORM_MATCHES} partidos."
            )

        corners = prediction.get("corners")
        cards = prediction.get("cards")
        extra = []
        if corners:
            extra.append(f"{corners['total']:.1f} córners")
        if cards:
            extra.append(f"{cards['total']:.1f} tarjetas")
        if extra:
            parts.append("Se esperan " + " y ".join(extra) + " en el partido.")

        h2h = prediction["h2h"]
        if h2h["count"]:
            parts.append(
                f"En los últimos {h2h['count']} enfrentamientos directos: "
                f"{h2h['home_wins']} victorias de {home}, {h2h['draws']} empates "
                f"y {h2h['away_wins']} de {away}."
            )
        return " ".join(parts)


def league_base_rates(results: pd.DataFrame) -> dict[str, float]:
    """Frecuencia histórica de cada mercado en la liga.

    Es la referencia contra la que se mide si un pronóstico dice algo o no:
    si el modelo da un 74% a "gana uno de los dos" y en la liga pasa el 74% de
    las veces, ese pronóstico no aporta información sobre este partido.
    """
    home_goals = results["home_goals"]
    away_goals = results["away_goals"]
    total_goals = home_goals + away_goals

    home_win = float((home_goals > away_goals).mean())
    draw = float((home_goals == away_goals).mean())
    away_win = float((home_goals < away_goals).mean())
    btts = float(((home_goals > 0) & (away_goals > 0)).mean())
    over25 = float((total_goals > 2.5).mean())

    rates = {
        "outcome_home": home_win,
        "outcome_draw": draw,
        "outcome_away": away_win,
        "double_home_draw": home_win + draw,
        "double_no_draw": home_win + away_win,
        "double_draw_away": draw + away_win,
        "goals_over": over25,
        "goals_under": 1 - over25,
        "btts_yes": btts,
        "btts_no": 1 - btts,
    }

    for name, (home_column, away_column, _lines) in SECONDARY_METRICS.items():
        line = config.MAIN_LINES.get(name)
        if line is None:
            continue
        totals = (results[home_column] + results[away_column]).dropna()
        if totals.empty:
            continue
        over = float((totals > line).mean())
        rates[f"{name}_over"] = over
        rates[f"{name}_under"] = 1 - over

    return rates


# --- Construcción y caché del motor -----------------------------------------

_lock = threading.Lock()
_engines: dict[str, Engine] = {}
_engine_built_at: dict[str, float] = {}
ENGINE_TTL_SECONDS = 3 * 3600


def build_engine(league: config.League | str | None = None) -> Engine:
    league = config.get_league(league)
    results = loader.load_results(league)
    if results.empty:
        raise RuntimeError(
            "No se pudieron cargar datos históricos. Comprueba la conexión a internet."
        )

    reference = results["date"].max()
    goals_model = dc.fit_goals_model(results, reference_date=reference)

    def fit(home_col: str, away_col: str, name: str) -> rates.RateModel | None:
        params = config.METRIC_PARAMS.get(name, {"half_life": 250.0, "prior": 12.0})
        return rates.fit_rate_model(
            results,
            home_col,
            away_col,
            name,
            reference_date=reference,
            xi=config.decay(params["half_life"]),
            prior_matches=params["prior"],
        )

    xg_model = fit("home_xg", "away_xg", "xg")
    secondary: dict[str, rates.RateModel | None] = {
        name: fit(home_col, away_col, name)
        for name, (home_col, away_col, _lines) in SECONDARY_METRICS.items()
    }

    joint_model = joint.fit_joint_model(results, goals_model, secondary, JOINT_METRICS)

    engine = Engine(
        league=league,
        results=results,
        view=ft.long_view(results),
        goals_model=goals_model,
        xg_model=xg_model,
        secondary=secondary,
        table=ft.standings(results),
        base_rates=league_base_rates(results),
        joint_model=joint_model,
        fitted_at=datetime.now(),
        reference_date=reference,
    )
    log.info(
        "Modelo %s entrenado con %s partidos (%s equipos, localía %.3f, rho %.3f)",
        league.name,
        goals_model.matches_used,
        len(goals_model.teams),
        goals_model.home_advantage,
        goals_model.rho,
    )
    log.info(
        "Correlación goles-córners %.2f, goles-tiros %.2f",
        joint_model.correlation[0, 1],
        joint_model.correlation[0, 3],
    )
    return engine


def get_engine(league: config.League | str | None = None, force: bool = False) -> Engine:
    league = config.get_league(league)
    with _lock:
        built_at = _engine_built_at.get(league.slug, 0.0)
        expired = (time.time() - built_at) > ENGINE_TTL_SECONDS
        current = _engines.get(league.slug)
        if force or current is None or expired:
            current = build_engine(league)
            _engines[league.slug] = current
            _engine_built_at[league.slug] = time.time()
        return current


def upcoming_fixtures(engine: Engine) -> pd.DataFrame:
    return loader.load_fixtures(engine.league, known_teams=set(engine.goals_model.teams))


def upcoming_predictions(
    engine: Engine,
    limit: int | None = None,
    full: bool = False,
) -> list[dict]:
    fixtures = upcoming_fixtures(engine)
    if fixtures.empty:
        return []
    if limit:
        fixtures = fixtures.head(limit)
    return [engine.predict(row, full=full) for _, row in fixtures.iterrows()]


def find_prediction(engine: Engine, match_id: str) -> dict | None:
    fixtures = upcoming_fixtures(engine)
    if fixtures.empty:
        return None
    match = fixtures[fixtures["match_id"] == match_id]
    if match.empty:
        return None
    return engine.predict(match.iloc[0], full=True)


def np_safe(value) -> float | None:
    """Convierte tipos numpy/pandas a float de Python (o None)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)
