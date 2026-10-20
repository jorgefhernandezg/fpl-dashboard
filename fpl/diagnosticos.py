"""Señales para leer a un jugador: hauls aislados, sobre-rendimiento, forma real."""
import numpy as np
import pandas as pd


def diagnosticos(t, cfg):
    g = t["jug_gw"]
    jugados = g[g["minutos"] > 0]
    gw_ref = g["gw"].max()
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

    titu60 = g[(g["minutos"] / g["n_part"].clip(lower=1)) >= 60]
    reg = titu60.assign(ok=(titu60["puntos"] >= 4 * titu60["n_part"]).astype(float)) \
        .groupby("element")["ok"].mean()
    a["regularidad"] = reg

    rec = g[g["gw"] > gw_ref - v].groupby("element").agg(
        min_rec=("minutos", "sum"), xg_rec=("xg", "sum"), xa_rec=("xa", "sum"), pts_rec=("puntos", "sum"))
    a = a.join(rec, how="outer")
    a["xgi90_rec"] = np.where(a["min_rec"] > 0, (a["xg_rec"] + a["xa_rec"]) * 90 / a["min_rec"], np.nan)
    a["tendencia"] = a["xgi90_rec"] - a["xgi90_temp"]

    a["alerta_haul"] = (a["concentracion"] >= 0.45) & (a["pts"] >= 20)
    a["alerta_suerte"] = a["brecha_z"] >= 2.0
    a["en_forma"] = (a["min_rec"] >= 60 * v * 0.75) & (a["xgi90_rec"] >= 0.25) & \
                    (a["xgi90_rec"] >= 1.3 * a["xgi90_temp"].clip(lower=0.05))
    return a
