"""Corrida completa: descarga, proyecta, evalúa transferencias y escribe los resultados.

Uso:
    python -m fpl.main                 # contra la API real
    python -m fpl.main --local carpeta # contra respuestas guardadas (pruebas)
"""
import argparse
import json
import os
import time
import tomllib

import numpy as np
import pandas as pd

from . import chips, datos, diagnosticos, modelo, momento, predicciones, transferencias
from .api import Fuente

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAM_BLOQUE = 200


def ni(x):
    return 0 if x is None or pd.isna(x) else int(x)


def r2(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), d)


def ejecutar(local=None, salida=None, ahora=None):
    t0 = time.time()
    with open(os.path.join(RAIZ, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    salida = salida or os.path.join(RAIZ, "data")
    fuente = Fuente(local=local)

    raw = datos.descargar(fuente)
    t = datos.tablas(raw)
    ev = t["eventos"]
    ahora = ahora or pd.Timestamp.now(tz="UTC")
    gw_actual = int(ev.loc[ev["is_current"] == True, "id"].iloc[0]) if ev["is_current"].any() else 0
    gw_sig = int(ev.loc[ev["is_next"] == True, "id"].iloc[0]) if ev["is_next"].any() else gw_actual + 1
    H = cfg["horizonte"]["jornadas"]
    gws = [g for g in range(gw_sig, gw_sig + H) if g <= 38]
    H = len(gws)
    deadline = ev.set_index("id")["deadline"][gw_sig]

    proy = modelo.proyectar(t, cfg, gws)
    X = proy["matriz"]
    pesos_gw = np.power(cfg["horizonte"]["descuento"], np.arange(H))
    diag = diagnosticos.diagnosticos(t, cfg)

    # --- tu equipo ---
    team_id = int(os.environ.get("FPL_TEAM_ID") or cfg["equipo"]["team_id"])
    cfg["equipo"]["team_id"] = team_id
    est = transferencias.estado_equipo(fuente, team_id, t, gw_actual, gw_sig)
    ft_cfg = cfg["equipo"]["transferencias_libres"]
    ft = int(ft_cfg) if ft_cfg else est["ft_estimadas"]
    eo = propiedad_efectiva(t, gw_actual, cfg)
    evalu = transferencias.Evaluador(X, t["jugadores"], cfg, eo)
    busq = transferencias.buscar(evalu, est["plantilla"], est["banco"], est["venta"], cfg)
    planes = transferencias.planificar(X, t["jugadores"], cfg, eo, est, ft, busq)
    gan_plan = sum(p["mov"]["hT"] for p in planes[0]["pasos"] if p["mov"])
    rchips = chips.evaluar(X, t["jugadores"], est, gws, gw_sig, raw["boot"], cfg, gan_plan)

    # --- registro de predicciones y validación ---
    ruta_reg = os.path.join(salida, "predicciones.csv")
    reg, escrito = predicciones.actualizar_registro(ruta_reg, gw_sig, deadline, ahora, X,
                                                    proy["detalle"], t["jugadores"])
    valid = predicciones.validar(reg, t)
    auditoria = auditar_capitania(fuente, team_id, reg, t)

    # --- salida ---
    res = armar_salida(t, cfg, proy, diag, est, busq, planes, ft, gws, gw_actual, gw_sig, deadline,
                       ahora, valid, auditoria, pesos_gw, eo, rchips, raw["boot"])
    res["meta"]["segundos"] = round(time.time() - t0, 1)
    res["meta"]["llamadas_api"] = fuente.llamadas
    escribir(res, salida)
    return res


def propiedad_efectiva(t, gw_actual, cfg):
    """Propiedad efectiva aproximada (0-1+): propiedad, y para el capitán más elegido del campo se suma
    la parte de sus dueños que lo capitanea. La API pública no da la capitanía del resto."""
    j = t["jugadores"].set_index("id")
    eo = (j["propiedad"].fillna(0) / 100.0).copy()
    ev = t["eventos"].set_index("id")
    cap = ev["most_captained"].get(gw_actual) if "most_captained" in ev.columns else None
    if cap is not None and not pd.isna(cap) and int(cap) in eo.index:
        eo[int(cap)] *= 1 + cfg["decision"].get("capitania_campo", 0.6)
    return eo


def auditar_capitania(fuente, team_id, reg, t):
    if len(reg) == 0:
        return []
    cerradas = set(t["eventos"].loc[t["eventos"]["finished"] == True, "id"])
    reales = t["jug_gw"].groupby(["gw", "element"])["puntos"].sum()
    filas = []
    for gw in sorted(set(reg["gw"]) & cerradas):
        try:
            p = fuente.get("entry/%d/event/%d/picks/" % (team_id, gw))
        except Exception:
            continue
        ids = [x["element"] for x in p["picks"]]
        cap = next((x["element"] for x in p["picks"] if x["is_captain"]), None)
        pr = reg[(reg["gw"] == gw) & reg["element"].isin(ids)].set_index("element")["xpts"]
        if cap is None or len(pr) == 0:
            continue
        sugerido = int(pr.idxmax())
        filas.append({"gw": int(gw), "capitan": int(cap), "xp_capitan": r2(pr.get(cap, 0)),
                      "real_capitan": int(reales.get((gw, cap), 0)),
                      "sugerido": sugerido, "xp_sugerido": r2(pr.max()),
                      "real_sugerido": int(reales.get((gw, sugerido), 0)),
                      "chip": p.get("active_chip")})
    return filas


def armar_salida(t, cfg, proy, diag, est, busq, planes, ft, gws, gw_actual, gw_sig, deadline, ahora,
                 valid, auditoria, pesos_gw, eo, rchips, boot):
    j = t["jugadores"].set_index("id")
    X = proy["matriz"]
    b = proy["base"]
    det = proy["detalle"]
    H = len(gws)
    xpt = (X * pesos_gw).sum(axis=1)
    sig = det[det["gw"] == gw_sig].groupby("id").agg(xmin=("xmin", "sum"), p60=("p60", "max"),
                                                     p_cs=("p_cs", "mean"))
    plantilla = est["plantilla"]
    gws_pas, u_pts, u_min, u_tot = diagnosticos.ultimas(t, 5)
    involucrados = set(plantilla)
    for pl in planes[:6]:
        for paso in pl["pasos"]:
            if paso["mov"]:
                involucrados |= set(np.atleast_1d(paso["mov"]["entra"])) | set(np.atleast_1d(paso["mov"]["sale"]))
    if rchips.get("detalle"):
        for dch in rchips["detalle"]:
            if dch.get("wildcard"):
                involucrados |= set(dch["wildcard"]["entran"])
    for m in busq["simples"][:40]:
        involucrados |= {m["entra"], m["sale"]}
    for m in busq["dobles"][:20]:
        involucrados |= set(m["entra"]) | set(m["sale"])

    jugadores = []
    for pid, r in j.iterrows():
        if r["status"] == "u" and pid not in involucrados:
            continue
        d = diag.loc[pid] if pid in diag.index else None
        if pid not in involucrados and (d is None or d["minutos"] == 0) and xpt.get(pid, 0) < 0.5:
            continue
        rec = {
            "id": int(pid), "n": r["web_name"], "e": r["abrev"], "t": int(r["team"]), "p": r["pos"],
            "$": r2(r["precio"], 1), "own": r2(r["propiedad"], 1), "st": r["status"],
            "news": r["news"] if isinstance(r["news"], str) else "", "pen": int(r["penalties_order"]) if r["penalties_order"] == 1 else 0,
            "ep": r2(r["ep_next"], 1),
            "neto": ni(r["transfers_in_event"]) - ni(r["transfers_out_event"]),
            "xp": [r2(v) for v in X.loc[pid].values], "xpt": r2(xpt[pid]),
            "xmin": r2(sig["xmin"].get(pid, 0), 0), "p60": r2(sig["p60"].get(pid, 0)),
            "ptit": r2(b.at[pid, "p_tit"]), "xg90": r2(b.at[pid, "xg90"]), "xa90": r2(b.at[pid, "xa90"]),
            "pdc": r2(b.at[pid, "p_dc"]) if r["pos"] != "GKP" else None,
            "eo": r2(eo.get(pid, 0), 3),
            "proy": r2(r["proy_precio"], 0),
        }
        if pid in u_tot.index:
            ut = u_tot.loc[pid]
            rec.update({
                "ult": [None if pd.isna(v) else int(v) for v in u_pts.loc[pid].values],
                "ultmin": [None if pd.isna(v) else int(v) for v in u_min.loc[pid].values],
                "ult_tit": ni(ut["tit"]), "ult_part": ni(ut["partidos"]), "ult_min": ni(ut["min"]),
                "ult_ga": ni(ut["goles"]) + ni(ut["asist"]), "ult_dc": ni(ut["dc"]),
                "ult_xgi90": r2(ut["xgi"] * 90 / ut["min"]) if ut["min"] > 0 else None,
            })
        if d is not None:
            rec.update({
                "pj": ni(d["pj"]), "min": ni(d["minutos"]),
                "pts": ni(d["pts"]), "med": r2(d["pts_mediana"], 1), "media": r2(d["pts_media"], 1),
                "conc": r2(d["concentracion"]), "sinmejor": r2(d["pts_sin_mejor"], 1),
                "g": ni(d["goles"]), "xg": r2(d["xg"]), "bz": r2(d["brecha_z"]),
                "reg": r2(d["regularidad"]), "xgi_rec": r2(d["xgi90_rec"]), "xgi_temp": r2(d["xgi90_temp"]),
                "tend": r2(d["tendencia"]),
                "gwm": ni(d["gw_mejor"]) if not pd.isna(d["gw_mejor"]) else None,
                "hace": ni(d["hace"]) if not pd.isna(d["hace"]) else None,
                "cxgi": r2(d["conc_xgi"]), "cpct": r2(d["conc_pct"]),
                "f": [k for k, c in (("haul", "alerta_haul"), ("suerte", "alerta_suerte"),
                                     ("forma", "en_forma")) if bool(d[c])],
            })
        jugadores.append(rec)

    # --- alineación recomendada para la siguiente GW ---
    pos = [j.at[i, "pos"] for i in plantilla]
    xi, cap, _ = transferencias.mejor_xi(X.loc[plantilla, gw_sig].values, pos)
    xs = X.loc[plantilla, gw_sig].values
    vice = max([i for i in xi if i != cap], key=lambda i: xs[i])
    banco = sorted([i for i in range(15) if i not in xi], key=lambda i: (pos[i] != "GKP", -xs[i]))

    def mov(m):
        return {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item()) for k, v in m.items()}

    por_salida = {}
    for m in busq["simples"]:
        lst = por_salida.setdefault(int(m["sale"]), [])
        if len(lst) < 3:
            lst.append(mov(m))

    entrada = est["entrada"]
    eqs = t["equipos"].set_index("team")
    fz = proy["fuerza"]
    fut = proy["futuros"]
    calendario = []
    for team, e in eqs.iterrows():
        f = fut[fut["team"] == team]
        calendario.append({
            "t": int(team), "e": e["abrev"], "nombre": e["equipo"],
            "ataque": r2(fz.at[team, "ataque"]) if team in fz.index else None,
            "defensa": r2(fz.at[team, "defensa"]) if team in fz.index else None,
            "partidos": [[int(r["gw"]), eqs.at[r["rival"], "abrev"], bool(r["local"]),
                          r2(r["lam_favor"]), r2(r["p_cs"])] for _, r in f.iterrows()],
        })

    hist = [{"gw": h["event"], "puntos": h["points"], "rank": h["overall_rank"],
             "promedio": int(t["eventos"].set_index("id")["average_entry_score"].get(h["event"], 0) or 0),
             "transf": h["event_transfers"], "golpe": h["event_transfers_cost"],
             "banco_pts": h["points_on_bench"]} for h in est["historial"]]

    meta = {
        "generado": ahora.isoformat(), "gw_actual": gw_actual, "gw_siguiente": gw_sig,
        "deadline": deadline.isoformat(), "gws": gws, "horizonte": H, "pesos": [r2(p, 3) for p in pesos_gw],
        "equipo": {"id": cfg["equipo"]["team_id"], "nombre": entrada.get("name"),
                   "manager": (str(entrada.get("player_first_name", "")) + " "
                               + str(entrada.get("player_last_name", ""))).strip(),
                   "puntos": entrada.get("summary_overall_points"), "rank": entrada.get("summary_overall_rank"),
                   "banco": est["banco"] / 10, "ft": ft, "ft_estimadas": est["ft_estimadas"],
                   "ft_configuradas": bool(cfg["equipo"]["transferencias_libres"]),
                   "pendientes": [{"entra": int(p["element_in"]), "sale": int(p["element_out"])}
                                  for p in est["pendientes"]]},
        "modelo": {"mu": r2(proy["mu"], 3), "ventaja_local": r2(proy["ventaja_local"], 3),
                   "vida_media": cfg["forma"]["vida_media_jugador"], "k_ataque": cfg["forma"]["k_ataque"],
                   "umbral_ft": cfg["decision"]["umbral_transferencia_libre"],
                   "margen_golpe": cfg["decision"]["margen_golpe"],
                   "aversion_riesgo": cfg["decision"].get("aversion_riesgo", 0.0),
                   "valor_dinero": cfg["decision"].get("valor_dinero", 0.0),
                   "permitir_golpes": bool(cfg["decision"].get("permitir_golpes", False))},
        "gws_pasados": gws_pas,
        "chips": [{"gw": int(c["event"]), "chip": c["name"]} for c in est["chips"]],
        "historial": hist,
    }
    def paso_json(p):
        return {"n": p["n"], "valor": r2(p["valor"]), "ft": p.get("ft"), "mov": mov(p["mov"]) if p["mov"] else None}

    def plan_json(pl):
        return {"total": r2(pl["total"]), "ft_final": pl["ft_final"], "golpes": pl["golpes"],
                "pasos": [paso_json(p) for p in pl["pasos"]]}

    mejor = planes[0]
    primero = mejor["pasos"][0]
    decision = {
        "tipo": {0: "guardar", 1: "simple", 2: "doble"}[primero["n"]],
        "mov": mov(primero["mov"]) if primero["mov"] else None,
        "momento": momento.evaluar(primero["mov"], t["jugadores"], est, deadline, ahora),
        "plan": plan_json(mejor), "planes": [plan_json(p) for p in planes[:6]],
    }
    # Riesgo frente al campo: jugadores muy poseídos que no tienes, ordenados por lo que el campo gana con ellos
    mios = set(plantilla)
    amen = []
    for pid in X.index:
        if pid in mios or eo.get(pid, 0) < 0.10:
            continue
        amen.append((float(eo[pid] * X.at[pid, gw_sig]), pid))
    amen.sort(reverse=True)
    riesgo = {"base": [r2(v) for v in busq["riesgo_base"]],
              "amenazas": [{"id": int(pid), "eo": r2(eo[pid], 3), "xp1": r2(X.at[pid, gw_sig]),
                            "campo": r2(v)} for v, pid in amen[:10]]}
    return {
        "meta": meta,
        "decision": decision,
        "transferencias": {"simples": [dict(mov(m), momento=momento.evaluar(m, t["jugadores"], est, deadline, ahora))
                                       for m in busq["simples"][:30]],
                           "dobles": [dict(mov(m), momento=momento.evaluar(m, t["jugadores"], est, deadline, ahora))
                                      for m in busq["dobles"][:20]],
                           "por_salida": {str(k): v for k, v in por_salida.items()}},
        "plantilla": {"ids": [int(i) for i in plantilla],
                      "venta": {str(k): v / 10 for k, v in est["venta"].items()},
                      "compra": {str(k): v / 10 for k, v in est["compra"].items() if k in plantilla},
                      "xi": [int(plantilla[i]) for i in xi], "banco": [int(plantilla[i]) for i in banco],
                      "capitan": int(plantilla[cap]), "vice": int(plantilla[vice]),
                      "valor_base": [r2(v) for v in busq["base"]]},
        "riesgo": riesgo,
        "chips": rchips,
        "calendario": calendario,
        "validacion": {"jornadas": valid, "capitania": auditoria},
        "jugadores": jugadores,
    }


def limpiar(o):
    """Convierte NaN/inf y tipos numpy a JSON válido."""
    if isinstance(o, dict):
        return {str(k): limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpiar(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def escribir(res, salida):
    res = limpiar(res)
    os.makedirs(os.path.join(salida, "db"), exist_ok=True)
    with open(os.path.join(salida, "resultados.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    docs = {
        "meta": {"meta": res["meta"], "decision": res["decision"]},
        "transferencias": res["transferencias"],
        "plantilla": res["plantilla"],
        "calendario": {"equipos": res["calendario"]},
        "validacion": res["validacion"],
        "chips": {"chips": res["chips"], "riesgo": res["riesgo"]},
    }
    jug = res["jugadores"]
    bloques = [jug[i:i + TAM_BLOQUE] for i in range(0, len(jug), TAM_BLOQUE)]
    docs["meta"]["meta"]["bloques_jugadores"] = len(bloques)
    for k, bl in enumerate(bloques):
        docs["jugadores-%d" % k] = {"lista": bl}
    for f in os.listdir(os.path.join(salida, "db")):
        os.remove(os.path.join(salida, "db", f))
    for k, d in docs.items():
        with open(os.path.join(salida, "db", k + ".json"), "w") as f:
            json.dump(d, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    tam = {k: os.path.getsize(os.path.join(salida, "db", k + ".json")) for k in docs}
    with open(os.path.join(salida, "db", "_indice.json"), "w") as f:
        json.dump({"generado": res["meta"]["generado"], "docs": tam}, f)
    print("Escrito:", {k: "%.0f KB" % (v / 1024) for k, v in tam.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local")
    ap.add_argument("--salida")
    a = ap.parse_args()
    r = ejecutar(local=a.local, salida=a.salida)
    d = r["decision"]
    print("GW siguiente:", r["meta"]["gw_siguiente"], "| decisión:", d["tipo"], "| plan:", d["plan"]["total"])
