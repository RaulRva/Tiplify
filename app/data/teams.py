"""Nombres, abreviaturas y colores de los equipos.

football-data.co.uk usa nombres cortos ("Ath Madrid", "Man City"). Aquí los
traducimos a nombre completo, código de 3 letras y color de club para la UI.
Cada liga tiene su propio diccionario para no mezclar homónimos.
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


# key CSV -> (nombre, código, color, texto)
_TEAMS: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "laliga": {
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
    },
    "premier": {
        "Arsenal": ("Arsenal", "ARS", "#EF0107", "#ffffff"),
        "Aston Villa": ("Aston Villa", "AVL", "#670E36", "#ffffff"),
        "Bournemouth": ("AFC Bournemouth", "BOU", "#DA291C", "#ffffff"),
        "Brentford": ("Brentford", "BRE", "#E30613", "#ffffff"),
        "Brighton": ("Brighton & Hove Albion", "BHA", "#0057B8", "#ffffff"),
        "Burnley": ("Burnley", "BUR", "#6C1D45", "#ffffff"),
        "Chelsea": ("Chelsea", "CHE", "#034694", "#ffffff"),
        "Coventry": ("Coventry City", "COV", "#1E90FF", "#10233f"),
        "Crystal Palace": ("Crystal Palace", "CRY", "#1B458F", "#ffffff"),
        "Everton": ("Everton", "EVE", "#003399", "#ffffff"),
        "Fulham": ("Fulham", "FUL", "#000000", "#ffffff"),
        "Hull": ("Hull City", "HUL", "#F5A12D", "#10233f"),
        "Ipswich": ("Ipswich Town", "IPS", "#0044A9", "#ffffff"),
        "Leeds": ("Leeds United", "LEE", "#FFCD00", "#1D428A"),
        "Leicester": ("Leicester City", "LEI", "#003090", "#ffffff"),
        "Liverpool": ("Liverpool", "LIV", "#C8102E", "#ffffff"),
        "Luton": ("Luton Town", "LUT", "#F78F1E", "#10233f"),
        "Man City": ("Manchester City", "MCI", "#6CABDD", "#10233f"),
        "Man United": ("Manchester United", "MUN", "#DA291C", "#ffffff"),
        "Newcastle": ("Newcastle United", "NEW", "#241F20", "#ffffff"),
        "Nott'm Forest": ("Nottingham Forest", "NFO", "#DD0000", "#ffffff"),
        "Sheffield United": ("Sheffield United", "SHU", "#EE2737", "#ffffff"),
        "Southampton": ("Southampton", "SOU", "#D71920", "#ffffff"),
        "Sunderland": ("Sunderland", "SUN", "#EB172B", "#ffffff"),
        "Tottenham": ("Tottenham Hotspur", "TOT", "#132257", "#ffffff"),
        "West Ham": ("West Ham United", "WHU", "#7A263A", "#ffffff"),
        "Wolves": ("Wolverhampton", "WOL", "#FDB913", "#10233f"),
    },
    "serie-a": {
        "Atalanta": ("Atalanta", "ATA", "#1E71B8", "#ffffff"),
        "Bologna": ("Bologna", "BOL", "#1A2F6B", "#ffffff"),
        "Cagliari": ("Cagliari", "CAG", "#A91C1C", "#ffffff"),
        "Como": ("Como", "COM", "#003DA5", "#ffffff"),
        "Cremonese": ("Cremonese", "CRE", "#8B0000", "#ffffff"),
        "Empoli": ("Empoli", "EMP", "#0054A6", "#ffffff"),
        "Fiorentina": ("Fiorentina", "FIO", "#482E92", "#ffffff"),
        "Frosinone": ("Frosinone", "FRO", "#FFDE00", "#10233f"),
        "Genoa": ("Genoa", "GEN", "#C8102E", "#ffffff"),
        "Inter": ("Inter", "INT", "#010E80", "#ffffff"),
        "Juventus": ("Juventus", "JUV", "#111111", "#ffffff"),
        "Lazio": ("Lazio", "LAZ", "#87D8F7", "#10233f"),
        "Lecce": ("Lecce", "LEC", "#E30613", "#ffffff"),
        "Milan": ("AC Milan", "MIL", "#FB090B", "#ffffff"),
        "Monza": ("Monza", "MON", "#E30613", "#ffffff"),
        "Napoli": ("Napoli", "NAP", "#12A0D7", "#10233f"),
        "Parma": ("Parma", "PAR", "#003DA5", "#ffffff"),
        "Pisa": ("Pisa", "PIS", "#001489", "#ffffff"),
        "Roma": ("Roma", "ROM", "#8E1F2F", "#ffffff"),
        "Salernitana": ("Salernitana", "SAL", "#8B1A1A", "#ffffff"),
        "Sampdoria": ("Sampdoria", "SAM", "#1B4F9C", "#ffffff"),
        "Sassuolo": ("Sassuolo", "SAS", "#00A650", "#ffffff"),
        "Spezia": ("Spezia", "SPE", "#EEEEEE", "#10233f"),
        "Torino": ("Torino", "TOR", "#8B1A1A", "#ffffff"),
        "Udinese": ("Udinese", "UDI", "#111111", "#ffffff"),
        "Venezia": ("Venezia", "VEN", "#F7A600", "#10233f"),
        "Verona": ("Hellas Verona", "VER", "#003DA5", "#ffffff"),
    },
}

_DEFAULT_COLOR = "#4b5563"
_DEFAULT_ACCENT = "#ffffff"

# El calendario (openfootball) usa el nombre oficial largo; el histórico
# (football-data.co.uk) usa nombres cortos. Esta tabla los une, por liga.
_ALIASES: dict[str, dict[str, str]] = {
    "laliga": {
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
    },
    "premier": {
        "arsenal fc": "Arsenal",
        "afc bournemouth": "Bournemouth",
        "bournemouth": "Bournemouth",
        "aston villa fc": "Aston Villa",
        "brentford fc": "Brentford",
        "brighton hove albion fc": "Brighton",
        "brighton and hove albion": "Brighton",
        "brighton hove albion": "Brighton",
        "burnley fc": "Burnley",
        "chelsea fc": "Chelsea",
        "coventry city fc": "Coventry",
        "coventry city": "Coventry",
        "crystal palace fc": "Crystal Palace",
        "everton fc": "Everton",
        "fulham fc": "Fulham",
        "hull city afc": "Hull",
        "hull city": "Hull",
        "ipswich town fc": "Ipswich",
        "ipswich town": "Ipswich",
        "leeds united fc": "Leeds",
        "leeds united": "Leeds",
        "leicester city": "Leicester",
        "liverpool fc": "Liverpool",
        "luton town": "Luton",
        "manchester city fc": "Man City",
        "manchester city": "Man City",
        "man city": "Man City",
        "manchester united fc": "Man United",
        "manchester united": "Man United",
        "man utd": "Man United",
        "man united": "Man United",
        "newcastle united fc": "Newcastle",
        "newcastle united": "Newcastle",
        "nottingham forest fc": "Nott'm Forest",
        "nottingham forest": "Nott'm Forest",
        "nottm forest": "Nott'm Forest",
        "sheffield united": "Sheffield United",
        "southampton fc": "Southampton",
        "sunderland afc": "Sunderland",
        "tottenham hotspur fc": "Tottenham",
        "tottenham hotspur": "Tottenham",
        "west ham united fc": "West Ham",
        "west ham united": "West Ham",
        "wolverhampton wanderers fc": "Wolves",
        "wolverhampton wanderers": "Wolves",
        "wolverhampton": "Wolves",
    },
    "serie-a": {
        "ac milan": "Milan",
        "ac monza": "Monza",
        "acf fiorentina": "Fiorentina",
        "as roma": "Roma",
        "atalanta bc": "Atalanta",
        "bologna fc 1909": "Bologna",
        "bologna fc": "Bologna",
        "cagliari calcio": "Cagliari",
        "como 1907": "Como",
        "us cremonese": "Cremonese",
        "empoli fc": "Empoli",
        "fc internazionale milano": "Inter",
        "internazionale": "Inter",
        "inter milan": "Inter",
        "frosinone calcio": "Frosinone",
        "genoa cfc": "Genoa",
        "hellas verona": "Verona",
        "hellas verona fc": "Verona",
        "juventus fc": "Juventus",
        "parma calcio 1913": "Parma",
        "parma calcio": "Parma",
        "pisa sc": "Pisa",
        "ss lazio": "Lazio",
        "ssc napoli": "Napoli",
        "torino fc": "Torino",
        "us lecce": "Lecce",
        "us salernitana 1919": "Salernitana",
        "us sassuolo calcio": "Sassuolo",
        "uc sampdoria": "Sampdoria",
        "spezia calcio": "Spezia",
        "udinese calcio": "Udinese",
        "venezia fc": "Venezia",
    },
}


def _simplify(value: str) -> str:
    """'Real Betis Balompié' -> 'real betis balompie'."""
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", ascii_only.lower())).strip()


def _league_slugs(league: str | None) -> list[str]:
    if league:
        return [league] if league in _TEAMS else []
    return list(_TEAMS)


def resolve_key(name: str, league: str | None = None) -> str | None:
    """Traduce cualquier variante de nombre a la clave usada en el histórico."""
    name = (name or "").strip()
    if not name:
        return None

    for slug in _league_slugs(league):
        teams = _TEAMS[slug]
        if name in teams:
            return name

        simple = _simplify(name)
        aliases = _ALIASES.get(slug, {})
        if simple in aliases:
            return aliases[simple]

        by_simple = {_simplify(key): key for key in teams}
        if simple in by_simple:
            return by_simple[simple]

        by_display = {_simplify(entry[0]): key for key, entry in teams.items()}
        found = by_display.get(simple)
        if found:
            return found
    return None


def slugify(value: str) -> str:
    """'Ath Madrid' -> 'ath-madrid' (sin acentos, apto para URLs)."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")


def _fallback_code(key: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", key).upper()
    return (letters[:3] or "???").ljust(3, "-")


def get_team(key: str, league: str | None = None) -> Team:
    """Devuelve los metadatos de un equipo, con valores razonables si no está."""
    key = (key or "").strip()
    for slug in _league_slugs(league):
        entry = _TEAMS[slug].get(key)
        if entry is not None:
            name, code, color, accent = entry
            return Team(key=key, name=name, code=code, color=color, accent=accent)
    return Team(
        key=key,
        name=key or "Desconocido",
        code=_fallback_code(key),
        color=_DEFAULT_COLOR,
        accent=_DEFAULT_ACCENT,
    )


def display_name(key: str, league: str | None = None) -> str:
    return get_team(key, league).name


def is_known(name: str, league: str | None = None) -> bool:
    """True si el equipo tiene nombre y colores configurados."""
    return resolve_key(name, league) is not None
