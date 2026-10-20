"""Señales para leer a un jugador: hauls aislados, sobre-rendimiento, forma real, producción reciente."""
import numpy as np
import pandas as pd


def diagnosticos(t, cfg):
    g = t["jug_gw"]
    jugados = g[g["minutos"] > 0].copy()
    gw_ref = int(g["gw"].max())
    v = cfg["forma"]["ventana_forma"]

    a = jugados.groupby("element").agg(
        pj=("gw", "count"), pts=("puntos", "sum"), pts_max=("puntos", "max"),
        pts_mediana=("puntos", "median"), goles=("goles", "sum"), asist=("asist", "sum"),
        xg=("xg", "sum"), xa=("xa", "sum"), minutos=("minutos", "sum"))
    a["pts_media"] = a["pts"] / a["pj"]
    a["concentracion"] = np.where(a["pts"] >= 10, a["pts_max"] / a["pts"].clip(lower=1), np.nan)
    a["pts_sin_mejor"] = np.where(a["pj"] > 1, (a["pts"] - a["pts_max"]) / (a["pj"] - 1).clip(lower=1), np.nan)
    a["brecha_z"] = (a["goles"] - a["xg"]) / np.sqrt(a["xg"].clip(lower=0.3))
    a["xgi90_temp"] = np.where(a["minutos"] > 0, (a["xg"] + a["xa"]) * 90 / a["minutos"], np.nan)

    # El mejor partido: en qué jornada fue y qué parte de su xG+xA de la temporada salió de ahí
    jugados["xgi"] = jugados["xg"] + jugados["xa"]
    idx = jugados.sort_values(["puntos", "gw"], ascending=[False, False]).groupby("element").head(1)
    idx = idx.set_index("element")
    a["gw_mejor"] = idx["gw"]
    a["xgi_mejor"] = idx["xgi"]
    a["conc_xgi"] = idx["xgi"] / (a["xg"] + a["xa"]).clip(lower=0.05)
    a["hace"] = gw_ref - a["gw_mejor"]

    # Percentil de concentración entre jugadores con un número parecido de partidos (±1)
    base = a[a["pts"] >= 15]
    pct = pd.Series(np.nan, index=a.index)
    for pid, r in base.iterrows():
        pares = base[(base["pj"] - r["pj"]).abs() <= 1]["concentracion"]
        pct[pid] = (pares <= r["concentracion"]).mean()
    a["conc_pct"] = pct

    titu60 = g[(g["minutos"] / g["n_part"].clip(lower=1)) >= 60]
    reg = titu60.assign(ok=(titu60["puntos"] >= 4 * titu60["n_part"]).astype(float)) \
        .groupby("element")["ok"].mean()
    a["regularidad"] = reg

    rec = g[g["gw"] > gw_ref - v].groupby("element").agg(
        min_rec=("minutos", "sum"), xg_rec=("xg", "sum"), xa_rec=("xa", "sum"), pts_rec=("puntos", "sum"))
    a = a.join(rec, how="outer")
    a["xgi90_rec"] = np.where(a["min_rec"] > 0, (a["xg_rec"] + a["xa_rec"]) * 90 / a["min_rec"], np.nan)
    a["tendencia"] = a["xgi90_rec"] - a["xgi90_temp"]

    # Haul aislado: muy concentrado para sus partidos Y su producción también salió de ese partido.
    # (En 2025-26, la concentración de puntos sola no anticipó caídas; con xG+xA concentrado sí.)
    a["alerta_haul"] = (a["conc_pct"] >= 0.90) & (a["conc_xgi"] >= 0.35) & (a["pts"] >= 15)
    a["alerta_suerte"] = a["brecha_z"] >= 2.0
    a["en_forma"] = (a["min_rec"] >= 60 * v * 0.75) & (a["xgi90_rec"] >= 0.25) & \
                    (a["xgi90_rec"] >= 1.3 * a["xgi90_temp"].clip(lower=0.05))
    return a


def ultimas(t, n=5):
    """Producción real de las últimas n jornadas jugadas, jugador por jugador."""
    g = t["jug_gw"]
    gws = sorted(g["gw"].unique())[-n:]
    u = g[g["gw"].isin(gws)].copy()
    u["xgi"] = u["xg"] + u["xa"]
    pts = u.pivot_table(index="element", columns="gw", values="puntos", aggfunc="sum")
    mins = u.pivot_table(index="element", columns="gw", values="minutos", aggfunc="sum")
    tot = u.groupby("element").agg(min=("minutos", "sum"), xgi=("xgi", "sum"), goles=("goles", "sum"),
                                   asist=("asist", "sum"), tit=("titular", "sum"), dc=("dc_hits", "sum"),
                                   partidos=("n_part", "sum"))
    return [int(x) for x in gws], pts.reindex(columns=gws), mins.reindex(columns=gws), tot
