"""Traduce la cuota justa del modelo al precio que pondría una casa.

Una casa no publica la cuota justa: le quita un margen. Pero no lo reparte por
igual. Medido sobre 1.579 partidos del histórico (`tools/measure_margin.py`),
cuando se normalizan las cuotas proporcionalmente los no favoritos ganan mucho
menos de lo que dicen (7,7% estimado frente a 3,1% real en el tramo 0-10%) y
los favoritos bastante más (65,8% frente a 71,6%). O sea: el margen se carga
sobre todo en las opciones poco probables.

Eso se reproduce con el **método de potencia**: la casa publica
`p_casa = p_justa ** e`, con un exponente `e < 1` que hace que la suma de todas
las opciones del mercado se pase de 1 justo en el margen. Elevar a una potencia
menor que 1 sube más, en términos relativos, las probabilidades pequeñas, que
es exactamente el sesgo observado. El ajuste sobre los datos da un exponente
inverso de 1.048 para el 1X2, coherente con el margen del 5%.

Ojo con lo que hay aquí: estos precios son una **estimación** del mercado a
partir del margen medido, no las cuotas reales de ninguna casa concreta.
"""

from __future__ import annotations

from functools import lru_cache

from scipy.optimize import brentq

from .. import config


@lru_cache(maxsize=8192)
def _exponent(probabilities: tuple[float, ...], margin: float) -> float:
    """Exponente `e` tal que la suma de `p ** e` se pasa de 1 en el margen."""
    target = 1.0 + margin

    def excess(exponent: float) -> float:
        return sum(p**exponent for p in probabilities) - target

    # Con un solo resultado (p=1) o probabilidades degeneradas no hay solución.
    try:
        return brentq(excess, 0.05, 1.0, xtol=1e-6)
    except ValueError:
        return 1.0


def house_probability(probability: float, market: tuple[float, ...], margin: float) -> float:
    """Probabilidad implícita en la cuota que publicaría la casa.

    `market` son las probabilidades justas de todas las opciones excluyentes de
    ese mercado: las tres del 1X2, o la selección y su contraria en los
    más/menos. El margen se reparte entre todas ellas.
    """
    rounded = tuple(round(value, 5) for value in market)
    exponent = _exponent(rounded, margin)
    return min(probability**exponent, 0.999)


def house_odds(probability: float, market: tuple[float, ...], margin: float) -> float:
    return 1.0 / house_probability(probability, market, margin)


def combo_house_odds(fair_probability: float, leg_margin_factors: tuple[float, ...]) -> float:
    """Precio de una combinada del mismo partido.

    La casa parte de la probabilidad conjunta (ya corregida por correlación) y
    le aplica el margen de cada pata, que se acumula. No hace falta inventar un
    margen de combinada: sale de multiplicar los de las patas, que sí están
    medidos. Con dos patas al 5,5% el margen total ronda el 11%.
    """
    factor = 1.0
    for value in leg_margin_factors:
        factor *= value
    return 1.0 / min(fair_probability * factor, 0.999)


def margin_for(family: str) -> float:
    """Margen aplicable a un mercado.

    Solo hay medición para el 1X2 y el más/menos 2.5 goles; para córners,
    tarjetas y tiros se usa el de los mercados binarios como aproximación.
    """
    if family == "resultado":
        return config.MARGIN_1X2
    return config.MARGIN_BINARY
