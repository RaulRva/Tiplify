"""Configuración central de Tiplify."""

from __future__ import annotations

import math
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# En Vercel el sistema de archivos del paquete es de solo lectura; /tmp sí se puede escribir.
CACHE_DIR = (
    Path("/tmp/tiplify-cache") if os.environ.get("VERCEL") else BASE_DIR / ".cache"
)

# --- Liga -------------------------------------------------------------------
# Código de football-data.co.uk: SP1 = LaLiga (Primera División).
LEAGUE_CODE = "SP1"
LEAGUE_NAME = "LaLiga"
LEAGUE_COUNTRY = "España"

# Temporadas usadas para entrenar (la más reciente primero).
# El decaimiento temporal reduce automáticamente el peso de las antiguas.
SEASONS = ["2627", "2526", "2425", "2324", "2223"]

RESULTS_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
# Cuotas de casas de apuestas de los próximos días (solo cubre ~1 semana).
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
# Calendario completo de las 38 jornadas.
CALENDAR_URL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/{label}/es.1.json"
)

# --- Caché ------------------------------------------------------------------
# Los resultados cambian poco; el calendario y las cuotas, más a menudo.
RESULTS_TTL_SECONDS = 6 * 3600
FIXTURES_TTL_SECONDS = 1800
CALENDAR_TTL_SECONDS = 12 * 3600
HTTP_TIMEOUT = 25


def season_label(season: str = SEASONS[0]) -> str:
    """'2627' -> '2026-27' (formato de openfootball)."""
    start = int(season[:2])
    century = 2000 if start < 90 else 1900
    return f"{century + start}-{season[2:]}"

# --- Modelo -----------------------------------------------------------------
# Todos estos valores salen del backtest walk-forward (tools/backtest.py y
# tools/tune_rates.py): son los que minimizan el logloss sobre partidos que el
# modelo no había visto. No los cambies "a ojo", vuelve a correr el backtest.


def decay(half_life_days: float) -> float:
    """Convierte una vida media en días en el factor de decaimiento xi."""
    return math.log(2) / half_life_days


# Goles: la fuerza de un equipo persiste, así que conviene memoria larga.
GOALS_HALF_LIFE_DAYS = 600.0
TIME_DECAY_XI = decay(GOALS_HALF_LIFE_DAYS)

# Encogimiento del Dixon-Coles hacia la media de la liga. Evita que un equipo
# recién ascendido con 4 partidos buenos aparezca como el mejor ataque.
GOALS_RIDGE = 0.12

# Las estadísticas secundarias son más volátiles y quieren memoria más corta:
# los tiros dependen del estilo actual del equipo mucho más que de su historia.
# `prior` = partidos equivalentes de encogimiento hacia la media de la liga.
METRIC_PARAMS = {
    "corners": {"half_life": 250.0, "prior": 12.0},
    "cards": {"half_life": 250.0, "prior": 12.0},
    "shots": {"half_life": 150.0, "prior": 6.0},
    "shots_target": {"half_life": 150.0, "prior": 10.0},
    "fouls": {"half_life": 250.0, "prior": 12.0},
    "xg": {"half_life": 150.0, "prior": 6.0},
}

RATE_MODEL_PRIOR_MATCHES = 12.0
RATE_MODEL_ITERATIONS = 80

# Peso máximo del modelo basado en xG al mezclarlo con el de goles.
XG_BLEND_MAX_WEIGHT = 0.35
# Partidos con xG necesarios para alcanzar ese peso máximo.
XG_BLEND_FULL_AT = 120

# Máximo de goles considerado al construir la matriz de marcadores.
MAX_GOALS = 10

# Nº de partidos que definen "la forma reciente".
FORM_MATCHES = 5
H2H_MATCHES = 8

# Líneas de apuesta mostradas en la ficha del partido.
GOAL_LINES = [1.5, 2.5, 3.5]
CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5]
CARD_LINES = [2.5, 3.5, 4.5, 5.5]
SHOT_LINES = [20.5, 22.5, 24.5]
SHOT_TARGET_LINES = [7.5, 8.5, 9.5]

# Línea principal de cada mercado (la que usan normalmente las casas).
MAIN_LINES = {
    "goals": 2.5,
    "corners": 9.5,
    "cards": 4.5,
    "shots": 22.5,
    "shots_target": 8.5,
}

# Margen mínimo sobre la probabilidad implícita del mercado para marcar valor.
VALUE_EDGE_THRESHOLD = 0.04

# --- Precio estimado de casa ------------------------------------------------
# Medidos sobre los 1.579 partidos del histórico con cuotas (tools/measure_margin.py).
# No son de ninguna casa concreta: son la media del mercado, y casas como
# Winamax o Bet365 se mueven en ese rango. Winamax no publica sus cuotas por
# ninguna vía accesible (su servidor responde 403 a cualquier petición
# automática), así que no se pueden leer directamente.
MARGIN_1X2 = 0.050
# Medido en el más/menos 2.5 goles. Se reutiliza para córners, tarjetas y
# tiros, donde no hay cuotas en los datos: ahí es una aproximación.
MARGIN_BINARY = 0.055

# Niveles de riesgo: para cada partido se busca una apuesta cuya cuota justa
# se acerque a cada uno de estos valores (segura, media y arriesgada).
BET_TIERS = (1.5, 2.0, 3.0)
BET_TIER_LABELS = {1.5: "Segura", 2.0: "Equilibrada", 3.0: "Arriesgada"}

# Simulación conjunta del partido (app/model/joint.py). Con 20.000 partidos
# simulados el error típico de una probabilidad del 33% es de 0.3 puntos, que
# en cuota son unas 3 centésimas: suficiente para mostrarla con dos decimales.
SIM_DRAWS = 20_000
CORRELATION_MIN_MATCHES = 200

# Penalización extra de una combinada frente a una apuesta simple: son menos
# transparentes y fallan enteras si falla una pata, así que solo deberían
# ganar cuando se acercan claramente mejor a la cuota objetivo.
COMBO_PENALTY = 0.035

# Rango de cuotas (asumiendo independencia, solo para descartar rápido) que
# merece la pena evaluar como combinada.
COMBO_ODDS_RANGE = (1.2, 6.0)
COMBO_MAX_EVALUATED = 400

# Si la combinada ocurre casi tan a menudo como su pata menos probable, es que
# una pata implica a la otra y la combinada es humo.
COMBO_REDUNDANT_RATIO = 0.95

# Cuántas apuestas se preseleccionan por nivel antes de buscar el trío que no
# comparte ningún mercado.
TIER_SHORTLIST = 14

# Sin esto, el buscador elegiría siempre el mercado más oscuro que clave la
# cuota: en un Sevilla-Barcelona acabaría recomendando "menos de 8.5 tiros a
# puerta" en vez de hablar del resultado. La penalización ordena por interés
# para quien apuesta y por fiabilidad medida en el backtest (las tarjetas son
# las que menos le ganan a la media de la liga; los tiros, las que más).
# Se compara con la distancia logarítmica a la cuota objetivo: 0.065 equivale
# a desviarse de 1.50 a 1.60.
BET_FAMILY_PENALTY = {
    "resultado": 0.00,
    "goles": 0.03,
    "ambos": 0.06,
    "corners": 0.09,
    "shots": 0.11,
    "shots_target": 0.13,
    "cards": 0.16,
}
