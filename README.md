# Tiplify

Pronósticos de **LaLiga** con modelos estadísticos: probabilidad de victoria,
empate y derrota, goles esperados, y estimaciones de córners, tarjetas y tiros
para cada partido. Se calculan a partir de la forma reciente de cada equipo, de
lo que genera y concede, y del historial de enfrentamientos directos.

Abres la app, ves la lista de próximos partidos y al entrar en uno tienes la
ficha completa.

## Arrancar

Doble clic en **`run.bat`**, o desde PowerShell:

```powershell
.\run.ps1
```

Luego abre <http://127.0.0.1:8000>. La primera carga descarga los datos y
entrena el modelo (unos 15 segundos); después queda en caché.

Si no tienes el entorno creado, `run.ps1` lo crea e instala las dependencias
solo. A mano sería:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

## Qué muestra

**Listado de partidos** — Agrupados por día, con la barra de probabilidades
1X2, el marcador esperado, córners, tarjetas y tiros previstos, la racha de los
últimos 5 partidos de cada equipo y las tres apuestas por nivel de riesgo.

**Ficha del partido** — Todo lo anterior más:

- **Tres apuestas, tres niveles de riesgo**: una apuesta con cuota cercana a
  1.5, otra a 2 y otra a 3, cada una de mercados distintos. Pueden ser simples
  o combinadas de dos selecciones. La cuota que se muestra es el precio
  estimado de casa, con la cuota justa del modelo al lado.
- Probabilidad de 1, X y 2 con su *cuota justa*, doble oportunidad y márgenes
  de victoria.
- Goles: total esperado, líneas de 1.5 / 2.5 / 3.5, ambos equipos marcan y los
  marcadores exactos más probables.
- Córners, tarjetas, tiros y tiros a puerta: reparto esperado entre los dos
  equipos y probabilidad de cada línea.
- Pronósticos destacados ordenados por probabilidad.
- **Modelo vs mercado**: compara las probabilidades del modelo con las cuotas
  reales sin margen de la casa, para ver dónde discrepan.
- Forma reciente de cada equipo (global y en su condición de local/visitante)
  con sus últimos partidos, y el historial de enfrentamientos directos.

**Clasificación** — La tabla real de la temporada junto a la fuerza ofensiva y
defensiva que el modelo estima para cada equipo.

## Cómo funciona el modelo

**Resultado y goles: Dixon-Coles.** Estima por máxima verosimilitud un
parámetro de ataque y otro de defensa para cada equipo, más la ventaja de jugar
en casa y un parámetro `rho` que corrige la frecuencia real de los marcadores
bajos (0-0, 1-0, 0-1, 1-1), donde un Poisson puro se queda corto. Con eso se
construye la matriz de probabilidad de cada marcador y de ahí salen el 1X2, las
líneas de goles, el "ambos marcan" y los marcadores probables. Los partidos
antiguos pesan menos (decaimiento exponencial). Cuando hay datos de xG de la
temporada en curso se mezclan con el modelo de goles.

**Córners, tarjetas, tiros y faltas: modelo multiplicativo.** Para cada
estadística se estima cuánto *genera* y cuánto *concede* cada equipo respecto a
la media de la liga, con la localía separada:

```
esperado(local) = media_local × genera_local × concede_visitante
```

Las probabilidades de cada línea usan una **binomial negativa** cuya
sobredispersión se estima de los residuos del histórico, porque córners y
tarjetas varían más de lo que predice un Poisson.

Los equipos con pocos partidos se encogen hacia la media de la liga, para que
un recién ascendido con cuatro buenos partidos no aparezca como el mejor ataque
de la liga.

**Las tres apuestas por nivel de riesgo.** Se generan todas las selecciones
posibles del partido (resultado, doble oportunidad, hándicap, líneas de goles,
ambos marcan, córners, tarjetas, tiros y tiros a puerta) con su cuota justa
`1 / probabilidad`, más las **combinadas** de dos de ellas, y se busca el trío
cuyas cuotas más se acercan a 1.5, 2 y 3. Tres detalles importantes:

- Las tres vienen de **mercados distintos**, contando también las patas de las
  combinadas. Todo lo que depende del resultado cuenta como un solo mercado,
  porque "empate" y "gana uno de los dos" son complementarios y recomendarlos
  a la vez no tendría sentido.
- Cada mercado lleva una **penalización** (`BET_FAMILY_PENALTY`) que ordena por
  interés y por fiabilidad medida en el backtest. Sin ella siempre ganaría el
  mercado más oscuro que clave la cuota, porque cualquier métrica la alcanza
  eligiendo la línea adecuada: en un Sevilla-Barcelona saldría "menos de 8.5
  tiros a puerta" en lugar de hablar del resultado. Una combinada hereda la
  penalización de su pata más oscura, más un extra por ser combinada.
- Se descartan las combinadas en las que **una pata implica a la otra**. "Más
  de 1.5 goles + marcan los dos" es exactamente "marcan los dos", y venderlo
  como combinada engañaría sobre el riesgo que se asume.

**Por qué una combinada no es multiplicar cuotas.** "Gana el Barça" y "más de
2.5 goles" van de la mano: si el Barça gana suele ser 3-0 o 3-1, así que la
combinada ocurre más veces de lo que diría el producto y su cuota justa es más
baja. En los partidos de la temporada la corrección llega al **±60 %**, o sea
que multiplicar sería un error grave, no un matiz.

Para calcularla bien se simula el partido 20.000 veces (`app/model/joint.py`):

1. El marcador sale de la matriz Dixon-Coles, así que cualquier mercado de
   resultado o de goles queda exacto por construcción.
2. Córners, tarjetas y tiros se enganchan al marcador con una **cópula
   gaussiana**. La correlación se mide sobre los **residuos** del histórico, no
   sobre los valores brutos: interesa cuánto se mueven juntos *más allá* de lo
   que ya predicen los modelos. Sale, por ejemplo, 0.51 entre goles y tiros a
   puerta, 0.32 entre córners y tiros, y prácticamente 0 en las tarjetas.

Las probabilidades individuales no cambian: la cópula solo reparte el conjunto.
`tools/check_combos.py` comprueba precisamente eso, que cada marginal simulada
coincide con la analítica.

**Del precio justo al precio de casa.** La cuota justa (`1 / probabilidad`) es
la que haría la apuesta neutra a largo plazo, y no la paga nadie: la casa se
queda un margen. Como el objetivo es que el 1.5 que muestra la app sea el que
se ve al apostar, los niveles se fijan sobre el **precio de casa estimado**, y
la cuota justa queda al lado para ver cuánto se lleva la casa.

El margen no se reparte por igual, y eso importa mucho. Midiendo las cuotas
reales del histórico (`tools/measure_margin.py`), al normalizarlas de forma
proporcional los no favoritos ganan mucho menos de lo que dicen y los
favoritos bastante más:

| prob. estimada | frecuencia real | diferencia |
|---|---|---|
| 0-10 % | 3,1 % | −4,6 pts |
| 20-30 % | 24,1 % | −1,8 pts |
| 55 %+ | 71,6 % | +5,7 pts |

O sea: **el margen se carga sobre las opciones poco probables**. Se reproduce
con el método de potencia, `p_casa = p_justa ** e`, con el exponente que hace
que la suma del mercado se pase de 1 justo en el margen medido (5,0 % en el
1X2, 5,5 % en los más/menos). El margen por selección que sale de ahí coincide
con el real:

| prob. justa | margen real (más/menos 2.5) |
|---|---|
| 20-30 % | 14,5 % |
| 30-40 % | 8,9 % |
| 40-55 % | 5,8 % |
| 55 %+ | 4,0 % |

En las combinadas el margen se acumula solo: se aplica el de cada pata sobre
la probabilidad conjunta, lo que da un 11-13 % en total sin tener que inventar
ningún número.

> **No son las cuotas de ninguna casa concreta.** Son una estimación a partir
> del margen medio del mercado. Winamax no publica sus cuotas por ninguna vía
> accesible: no tiene API pública, las sirve por un Socket.IO sin documentar y
> su servidor responde 403 a cualquier petición automática. El margen de
> córners, tarjetas y tiros es además una aproximación, porque los datos
> abiertos no traen cuotas de esos mercados.

## Validación

Los parámetros no están puestos a ojo: salen de un backtest *walk-forward*, que
entrena solo con los partidos anteriores a cada bloque y predice los
siguientes, igual que en producción.

```powershell
.\.venv\Scripts\python.exe tools\backtest.py           # goles y 1X2
.\.venv\Scripts\python.exe tools\tune_rates.py         # córners, tarjetas, tiros
.\.venv\Scripts\python.exe tools\smoke_test.py         # comprobación rápida
.\.venv\Scripts\python.exe tools\check_consistency.py  # coherencia de las fichas
```

Resultado sobre los últimos 380 partidos de LaLiga, con la configuración que
trae `config.py` (vida media 600 días, ridge 0.12):

| | logloss 1X2 | Brier | acierto |
|---|---|---|---|
| Tiplify | 0.975 | 0.578 | 52.3 % |
| Mercado de apuestas | 0.966 | 0.573 | 54.6 % |

El modelo queda a 0.009 de logloss del mercado, que incorpora alineaciones,
lesiones y dinero real. Es un margen pequeño, pero el mercado sigue siendo
mejor: no esperes ganarle sistemáticamente.

En córners, tarjetas y tiros, el modelo bate a la línea base de "predecir
siempre la media de la liga" en las cuatro métricas:

| Métrica | MAE modelo | MAE media liga | Mejora en logloss |
|---|---|---|---|
| Córners | 2.66 | 2.75 | 0.015 |
| Tarjetas | 1.92 | 1.97 | 0.014 |
| Tiros | 4.53 | 4.84 | 0.039 |
| Tiros a puerta | 2.38 | 2.49 | 0.026 |

La ganancia en tarjetas es pequeña: dependen mucho del árbitro, un dato que el
modelo todavía no usa.

Una salvedad honesta: la mezcla con el modelo de xG **no está validada**.
football-data solo publica xG desde la temporada 2026/27, así que no hay
histórico suficiente para medirla en el backtest. Por eso su peso es pequeño y
crece con los partidos disponibles, hasta un máximo del 35 %
(`XG_BLEND_MAX_WEIGHT`). Si prefieres quedarte solo con lo medido, ponlo a 0.

## Datos

- **Resultados y estadísticas** (goles, xG, tiros, córners, faltas, tarjetas y
  cuotas): [football-data.co.uk](https://www.football-data.co.uk/spainm.php),
  últimas 5 temporadas.
- **Calendario completo** de las 38 jornadas:
  [openfootball/football.json](https://github.com/openfootball/football.json).

Todo se guarda en `.cache/`, así que la app sigue funcionando sin conexión con
los últimos datos descargados. El botón **Actualizar datos** fuerza una
descarga limpia y reentrena.

## API

| Ruta | Qué devuelve |
|---|---|
| `GET /api/partidos?n=20&completo=false` | Próximos partidos con su pronóstico |
| `GET /api/partidos/{match_id}` | Ficha completa de un partido |
| `GET /api/estado` | Estado del modelo y frescura de los datos |

## Estructura

```
app/
  config.py            parámetros del modelo (validados por backtest)
  main.py              rutas web y API
  data/
    loader.py          descarga, caché y normalización
    teams.py           nombres, escudos y equivalencias entre fuentes
  model/
    dixon_coles.py     modelo de goles y derivados (1X2, líneas, marcadores)
    rates.py           córners, tarjetas, tiros, faltas y xG
    joint.py           simulación conjunta del partido (precio de combinadas)
    pricing.py         de la cuota justa al precio estimado de casa
    features.py        forma reciente, clasificación y head-to-head
    predictor.py       une todo y cachea las fichas
  templates/, static/  interfaz
tools/                 backtest y comprobaciones
```

## Añadir otra liga

En `app/config.py` cambia `LEAGUE_CODE` por el código de football-data
(`E0` Premier League, `I1` Serie A, `D1` Bundesliga, `F1` Ligue 1, `SP2`
Segunda) y ajusta `CALENDAR_URL` al fichero correspondiente de openfootball
(`es.1.json`, `en.1.json`, `it.1.json`...). Luego añade los equipos nuevos a
`app/data/teams.py` y vuelve a correr el backtest.

## Aviso

Son estimaciones estadísticas, no certezas. El modelo no conoce lesiones,
alineaciones, sanciones ni rotaciones por competición europea. Si lo usas para
apostar, hazlo con responsabilidad.
