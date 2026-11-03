"""Penales: emparejar los datos de Understat con FPL y separar el xG de penales del de juego.

FPL no distingue qué parte del xG de un jugador vino de penales. Con Understat se sabe cuántos penales
pateó cada jugador en cada partido; ese xG se resta de su producción y el xG de penales esperado del
equipo se asigna al lanzador designado (penalties_order de FPL).
"""
import difflib
import json
import os
import unicodedata

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALIAS_EQUIPOS = {
    "manchester city": "man city", "manchester united": "man utd", "tottenham": "spurs",
    "tottenham hotspur": "spurs", "wolverhampton wanderers": "wolves", "nottingham forest": "nott'm forest",
    "newcastle united": "newcastle", "west ham": "west ham", "brighton": "brighton",
    "leeds": "leeds", "sheffield united": "sheffield utd", "ipswich": "ipswich", "hull city": "hull",
    "coventry city": "coventry", "leicester": "leicester", "luton": "luton",
}


def _n(s):
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(".", " ").replace("'", "").split())


def equipos_map(understat_nombres, equipos):
    """equipos: DataFrame con columnas team, nombre (nombre largo de FPL), abrev."""
    fpl = {_n(r.nombre): int(r.team) for r in equipos.itertuples()}
    out = {}
    for u in understat_nombres:
        k = _n(u)
        k2 = _n(ALIAS_EQUIPOS.get(k, k))
        if k in fpl:
            out[u] = fpl[k]
        elif k2 in fpl:
            out[u] = fpl[k2]
        else:
            m = difflib.get_close_matches(k2, list(fpl), n=1, cutoff=0.6)
            out[u] = fpl[m[0]] if m else None
    return out


def jugador_map(nombre, team, jugadores):
    """jugadores: DataFrame con id, first_name, second_name, web_name, team."""
    q = _n(nombre)
    tokens = set(q.split())

    def puntaje(r):
        full = _n("%s %s" % (r.first_name, r.second_name))
        web = _n(r.web_name)
        if q == full or q == web:
            return 1.0
        ft = set(full.split())
        if tokens and tokens <= ft:
            return 0.95
        if web and web in q.split():
            return 0.9
        return difflib.SequenceMatcher(None, q, full).ratio() * 0.9
    mejores = []
    for cand in (jugadores[jugadores["team"] == team], jugadores):
        if not len(cand):
            continue
        sc = cand.apply(puntaje, axis=1)
        i = sc.idxmax()
        mejores.append((sc[i], int(cand.loc[i, "id"])))
        if sc[i] >= 0.85:
            return int(cand.loc[i, "id"]), float(sc[i])
    s, pid = max(mejores) if mejores else (0, None)
    return (pid, float(s)) if s >= 0.75 else (None, float(s))


def preparar(anio, jugadores, equipos, fixtures):
    """Devuelve (intentos por jugador y jornada, partidos por equipo, reporte de emparejamiento).

    fixtures: DataFrame con event, team_h, team_a."""
    ruta = os.path.join(RAIZ, "data", "understat", "penales_%d.json" % anio)
    if not os.path.exists(ruta):
        return None, None, {"error": "sin datos de Understat para %d" % anio}
    d = json.load(open(ruta))
    partidos = pd.DataFrame.from_dict(d["partidos"], orient="index")
    if not len(partidos):
        return None, None, {"error": "sin partidos procesados"}
    eq = equipos_map(set(partidos["local"]) | set(partidos["visita"]), equipos)
    partidos["th"] = partidos["local"].map(eq)
    partidos["ta"] = partidos["visita"].map(eq)
    fx = fixtures.dropna(subset=["event"])
    gw_de = {(int(r.team_h), int(r.team_a)): int(r.event) for r in fx.itertuples()}
    partidos["gw"] = [gw_de.get((h, a)) if h and a else None for h, a in zip(partidos["th"], partidos["ta"])]
    pen = pd.DataFrame(d["penales"])
    reporte = {"equipos_sin_emparejar": sorted([k for k, v in eq.items() if v is None]),
               "partidos": int(len(partidos)), "partidos_sin_jornada": int(partidos["gw"].isna().sum()),
               "penales": int(len(pen))}
    if len(pen):
        pen["team"] = np.where(pen["lado"] == "h", pen["local"].map(eq), pen["visita"].map(eq))
        pen["gw"] = pen["partido"].map(partidos["gw"])
        res = [jugador_map(r.jugador, r.team, jugadores) for r in pen.itertuples()]
        pen["element"] = [x[0] for x in res]
        pen["confianza"] = [x[1] for x in res]
        dudosos = pen[(pen["confianza"] < 0.9)][["jugador", "team", "element", "confianza"]].drop_duplicates()
        nombres = jugadores.set_index("id")["web_name"]
        reporte["jugadores_sin_emparejar"] = sorted(set(pen.loc[pen["element"].isna(), "jugador"]))
        reporte["emparejamientos_dudosos"] = [
            {"understat": r.jugador, "fpl": nombres.get(r.element), "confianza": round(r.confianza, 2)}
            for r in dudosos.itertuples() if r.element is not None]
        intentos = pen.dropna(subset=["element", "gw"]).groupby(["element", "gw"]).size().rename("intentos")
        intentos.index = intentos.index.set_levels([intentos.index.levels[0].astype(int),
                                                    intentos.index.levels[1].astype(int)])
    else:
        intentos = pd.Series(dtype=float, name="intentos")
    # Penales por equipo y partido, para la tasa de cada equipo
    por_eq = pd.concat([
        partidos[["th", "gw"]].rename(columns={"th": "team"}).assign(lado="h"),
        partidos[["ta", "gw"]].rename(columns={"ta": "team"}).assign(lado="a")]).reset_index()
    por_eq = por_eq.rename(columns={"index": "partido"})
    if len(pen):
        cnt = pen.groupby(["partido", "lado"]).size()
        por_eq["pen"] = [cnt.get((p, l), 0) for p, l in zip(por_eq["partido"], por_eq["lado"])]
    else:
        por_eq["pen"] = 0
    return intentos, por_eq.dropna(subset=["team"]), reporte


def tasa_equipos(por_eq, hasta_gw, prior_partidos=12):
    """Penales por partido de cada equipo hasta una jornada, encogidos hacia el promedio de la liga."""
    p = por_eq[por_eq["gw"].notna() & (por_eq["gw"] <= hasta_gw)]
    if not len(p):
        return {}, 0.12
    liga = p["pen"].sum() / len(p)
    a = p.groupby("team")["pen"].agg(["sum", "count"])
    tasa = (a["sum"] + prior_partidos * liga) / (a["count"] + prior_partidos)
    return {int(k): float(v) for k, v in tasa.items()}, float(liga)
