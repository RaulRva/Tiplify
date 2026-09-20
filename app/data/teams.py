"""Nombres, abreviaturas y colores de los equipos.

football-data.co.uk usa nombres cortos ("Ath Madrid", "Vallecano"). Aquí los
traducimos a nombre completo, código de 3 letras y color de club para la UI.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    key: str  # nombre tal y como aparece en el CSV
    name: str  # nombre para mostrar
    code: str  # abreviatura de 3 letras
    color: str  # color principal del club
    accent: str  # color secundario (texto sobre el principal)


_TEAMS: dict[str, tuple[str, str, str, str]] = {
    # key CSV              nombre                    código  color      texto
    "Alaves": ("Deportivo Alavés", "ALA", "#0761AF", "#ffffff"),
    "Almeria": ("UD Almería", "ALM", "#CE2B2E", "#ffffff"),
    "Ath Bilbao": ("Athletic Club", "ATH", "#EE2523", "#ffffff"),
    "Ath Madrid": ("Atlético de Madrid", "ATM", "#CB3524", "#ffffff"),
    "Barcelona": ("FC Barcelona", "BAR", "#A50044", "#ffffff"),
    "Betis": ("Real Betis", "BET", "#00954C", "#ffffff"),
    "Cadiz": ("Cádiz CF", "CAD", "#F2E500", "#10233f"),
    "Celta": ("RC Celta", "CEL", "#8AC3EE", "#10233f"),
    "Cordoba": ("Córdoba CF", "COR", "#00843D", "#ffffff"),
    "Eibar": ("SD Eibar", "EIB", "#0B4EA2", "#ffffff"),
    "Elche": ("Elche CF", "ELC", "#00913F", "#ffffff"),
    "Espanol": ("RCD Espanyol", "ESP", "#0067B1", "#ffffff"),
    "Getafe": ("Getafe CF", "GET", "#005999", "#ffffff"),
    "Girona": ("Girona FC", "GIR", "#CD2534", "#ffffff"),
    "Granada": ("Granada CF", "GRA", "#C4122E", "#ffffff"),
    "Huesca": ("SD Huesca", "HUE", "#005CA9", "#ffffff"),
    "La Coruna": ("RC Deportivo", "DEP", "#0055A5", "#ffffff"),
    "Las Palmas": ("UD Las Palmas", "LPA", "#FFE400", "#10233f"),
    "Leganes": ("CD Leganés", "LEG", "#005BAC", "#ffffff"),
    "Levante": ("Levante UD", "LEV", "#004B9D", "#ffffff"),
    "Mallorca": ("RCD Mallorca", "MLL", "#E20613", "#ffffff"),
    "Malaga": ("Málaga CF", "MAL", "#0080C8", "#ffffff"),
    "Osasuna": ("CA Osasuna", "OSA", "#0A346F", "#ffffff"),
    "Oviedo": ("Real Oviedo", "OVI", "#0057A8", "#ffffff"),
    "Real Madrid": ("Real Madrid", "RMA", "#FEBE10", "#10233f"),
    "Santander": ("Racing de Santander", "RAC", "#00A650", "#ffffff"),
    "Sevilla": ("Sevilla FC", "SEV", "#D81920", "#ffffff"),
    "Sociedad": ("Real Sociedad", "RSO", "#0067B2", "#ffffff"),
    "Sp Gijon": ("Sporting de Gijón", "SPG", "#D2232A", "#ffffff"),
    "Valencia": ("Valencia CF", "VAL", "#EE7203", "#ffffff"),
    "Valladolid": ("Real Valladolid", "VLL", "#921E8C", "#ffffff"),
    "Vallecano": ("Rayo Vallecano", "RAY", "#E53027", "#ffffff"),
    "Villarreal": ("Villarreal CF", "VIL", "#FFE667", "#10233f"),
}

_DEFAULT_COLOR = "#4b5563"
_DEFAULT_ACCENT = "#ffffff"

# El calendario (openfootball) usa el nombre oficial largo; el histórico
# (football-data.co.uk) usa nombres cortos. Esta tabla los une.
_ALIASES: dict[str, str] = {
    "athletic club": "Ath Bilbao",
    "athletic club de bilbao": "Ath Bilbao",
    "ca osasuna": "Osasuna",
    "club atletico de madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid",
    "atletico madrid": "Ath Madrid",
    "deportivo alaves": "Alaves",
    "elche cf": "Elche",
    "fc barcelona": "Barcelona",
    "getafe cf": "Getafe",
    "girona fc": "Girona",
    "levante ud": "Levante",
    "malaga cf": "Malaga",
    "rc celta de vigo": "Celta",
    "rc celta": "Celta",
    "celta de vigo": "Celta",
    "rc deportivo la coruna": "La Coruna",
    "rc deportivo": "La Coruna",
    "deportivo la coruna": "La Coruna",
    "rcd espanyol de barcelona": "Espanol",
    "rcd espanyol": "Espanol",
    "espanyol": "Espanol",
    "rcd mallorca": "Mallorca",
    "rayo vallecano de madrid": "Vallecano",
    "rayo vallecano": "Vallecano",
    "real betis balompie": "Betis",
    "real betis": "Betis",
    "real madrid cf": "Real Madrid",
    "real oviedo": "Oviedo",
    "real racing club de santander": "Santander",
    "racing de santander": "Santander",
    "racing club de santander": "Santander",
    "real sociedad de futbol": "Sociedad",
    "real sociedad": "Sociedad",
    "real valladolid cf": "Valladolid",
    "real valladolid": "Valladolid",
    "sevilla fc": "Sevilla",
    "ud almeria": "Almeria",
    "ud las palmas": "Las Palmas",
    "cd leganes": "Leganes",
    "sd eibar": "Eibar",
    "sd huesca": "Huesca",
    "valencia cf": "Valencia",
    "villarreal cf": "Villarreal",
    "cadiz cf": "Cadiz",
    "granada cf": "Granada",
    "real sporting de gijon": "Sp Gijon",
    "cordoba cf": "Cordoba",
}


def _simplify(value: str) -> str:
    """'Real Betis Balompié' -> 'real betis balompie'."""
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", ascii_only.lower())).strip()


def resolve_key(name: str) -> str | None:
    """Traduce cualquier variante de nombre a la clave usada en el histórico."""
    name = (name or "").strip()
    if not name:
        return None
    if name in _TEAMS:
        return name

    simple = _simplify(name)
    if simple in _ALIASES:
        return _ALIASES[simple]

    by_simple = {_simplify(key): key for key in _TEAMS}
    if simple in by_simple:
        return by_simple[simple]

    # Último recurso: comparar contra el nombre para mostrar.
    by_display = {_simplify(entry[0]): key for key, entry in _TEAMS.items()}
    return by_display.get(simple)


def slugify(value: str) -> str:
    """'Ath Madrid' -> 'ath-madrid' (sin acentos, apto para URLs)."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")


def _fallback_code(key: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", key).upper()
    return (letters[:3] or "???").ljust(3, "-")


def get_team(key: str) -> Team:
    """Devuelve los metadatos de un equipo, con valores razonables si no está."""
    key = (key or "").strip()
    entry = _TEAMS.get(key)
    if entry is None:
        return Team(
            key=key,
            name=key or "Desconocido",
            code=_fallback_code(key),
            color=_DEFAULT_COLOR,
            accent=_DEFAULT_ACCENT,
        )
    name, code, color, accent = entry
    return Team(key=key, name=name, code=code, color=color, accent=accent)


def display_name(key: str) -> str:
    return get_team(key).name


def is_known(name: str) -> bool:
    """True si el equipo tiene nombre y colores configurados."""
    return resolve_key(name) is not None
