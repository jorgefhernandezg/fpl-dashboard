"""Registro de lo que el modelo esperaba antes de cada deadline, y su comparación con lo ocurrido.

Es lo único que se guarda entre corridas: la API no permite reconstruirlo después.
"""
import os

import numpy as np
import pandas as pd

COLS = ["gw", "element", "xpts", "xmin", "ep_next", "precio", "propiedad", "generado"]


def actualizar_registro(ruta, gw_sig, deadline, ahora, matriz, det, jugadores):
    reg = pd.read_csv(ruta) if os.path.exists(ruta) else pd.DataFrame(columns=COLS)
    if ahora >= deadline:
        return reg, False  # deadline pasado: la predicción de esta GW ya quedó congelada
    j = jugadores.set_index("id")
    xmin = det[det["gw"] == gw_sig].groupby("id")["xmin"].sum()
    nueva = pd.DataFrame({
        "gw": gw_sig, "element": matriz.index, "xpts": matriz[gw_sig].round(3).values,
        "xmin": xmin.reindex(matriz.index).fillna(0).round(1).values,
        "ep_next": j["ep_next"].reindex(matriz.index).values,
        "precio": j["precio"].reindex(matriz.index).values,
        "propiedad": j["propiedad"].reindex(matriz.index).values,
        "generado": ahora.isoformat(),
    })
    nueva = nueva[(nueva["xpts"] > 0.05) | (nueva["ep_next"] > 0)]
    reg = pd.concat([reg[reg["gw"] != gw_sig], nueva], ignore_index=True)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    reg.to_csv(ruta, index=False)
    return reg, True


def validar(reg, t, picks_por_gw=None):
    """Error y correlación de rangos por jornada cerrada, del modelo y de ep_next de FPL."""
    if len(reg) == 0:
        return []
    reales = t["jug_gw"].groupby(["gw", "element"])["puntos"].sum().rename("real").reset_index()
    cerradas = set(t["eventos"].loc[t["eventos"]["finished"] == True, "id"])
    m = reg[reg["gw"].isin(cerradas)].merge(reales, on=["gw", "element"], how="left")
    m["real"] = m["real"].fillna(0)
    salida = []
    pos = t["jugadores"].set_index("id")["pos"]
    for gw, d in m.groupby("gw"):
        d = d.assign(pos=d["element"].map(pos))
        rel = d[(d["xmin"] >= 45) | (d["real"] != 0)]
        fila = {"gw": int(gw), "n": int(len(rel)),
                "mae_modelo": float((rel["xpts"] - rel["real"]).abs().mean()),
                "mae_fpl": float((rel["ep_next"] - rel["real"]).abs().mean()),
                "rho_modelo": float(rel["xpts"].corr(rel["real"], method="spearman")),
                "rho_fpl": float(rel["ep_next"].corr(rel["real"], method="spearman"))}
        por_pos = {}
        for p, dp in rel.groupby("pos"):
            por_pos[p] = {"rho_modelo": round(float(dp["xpts"].corr(dp["real"], method="spearman")), 3),
                          "rho_fpl": round(float(dp["ep_next"].corr(dp["real"], method="spearman")), 3)}
        fila["por_pos"] = por_pos
        salida.append({k: (round(v, 3) if isinstance(v, float) and not np.isnan(v) else v)
                       for k, v in fila.items()})
    return salida
