"""Penales de la temporada, cargados a mano desde Transfermarkt (data/penales_<año>.csv).

FPL no dice qué parte del xG de un jugador vino de penales. Con esta lista, el xG de cada penal pateado
sale de la producción de juego del jugador, y el xG de penales se proyecta aparte: cada equipo recibe
penales a su ritmo (encogido hacia el promedio de la liga) y los patea su lanzador designado en FPL.

Columnas: gw, element (id de FPL), jugador, equipo, convertido (1/0). Los fallados también cuentan:
el xG de un penal es el mismo lo marque o no.
"""
import os

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cargar(t, cfg, gw_actual):
    cal = cfg.get("calibracion", {})
    if not cal.get("penales", False):
        return {"activo": False}
    ruta = os.path.join(RAIZ, cal.get("archivo_penales", "data/penales_2026.csv"))
    if not os.path.exists(ruta):
        return {"activo": False, "error": "no existe %s" % os.path.basename(ruta)}
    p = pd.read_csv(ruta)
    j = t["jugadores"].set_index("id")
    p = p[p["element"].isin(j.index) & (p["gw"] <= gw_actual)].copy()
    p["team"] = p["element"].map(j["team"])
    intentos = p.groupby(["element", "gw"]).size()
    t["pen_intentos"] = {(int(e), int(g)): int(n) for (e, g), n in intentos.items()}

    # Ritmo de penales por equipo: penales / partidos jugados, encogido hacia la liga con `prior` partidos
    partidos = t["eq_gw"][t["eq_gw"]["gw"] <= gw_actual].groupby("team")["n_part"].sum()
    por_eq = p.groupby("team").size().reindex(partidos.index, fill_value=0)
    liga = float(por_eq.sum() / max(partidos.sum(), 1))
    prior = cal.get("prior_partidos_penales", 12)
    tasa = (por_eq + prior * liga) / (partidos + prior)
    t["pen_tasa"] = ({int(k): float(v) for k, v in tasa.items()}, liga)
    return {"activo": True, "penales": int(len(p)), "hasta_gw": int(p["gw"].max()) if len(p) else None,
            "por_partido_liga": round(liga, 3), "archivo": os.path.basename(ruta)}
