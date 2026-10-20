"""Cuándo usar cada chip vigente: valor esperado en cada jornada del horizonte.

  Triple capitán -> puntos extra del mejor capitán de tu plantilla
  Bench Boost    -> puntos de tus 4 suplentes
  Free Hit       -> mejor equipo posible para esa jornada (presupuesto = valor de venta + banco) menos tu XI
  Wildcard       -> mejor plantilla para todo el horizonte menos la tuya actual
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .transferencias import CUPO, MAX_POS, MIN_POS, mejor_xi

NOMBRES = {"3xc": "Triple capitán", "bboost": "Bench Boost", "freehit": "Free Hit", "wildcard": "Wildcard"}


def vigentes(boot, usados, gw_sig):
    """Chips que puedes usar a partir de gw_sig, con la jornada límite de cada uno."""
    ventanas = []
    for c in boot.get("chips", []) or []:
        if c.get("name") in NOMBRES:
            ventanas.append((c["name"], int(c.get("start_event") or 1), int(c.get("stop_event") or 38)))
    if not ventanas:  # reglas 2025-26: cada chip una vez por mitad de temporada
        ventanas = [(n, a, b) for n in NOMBRES for a, b in ((1, 19), (20, 38))]
    out = []
    for nombre, a, b in ventanas:
        if not (a <= gw_sig <= b):
            continue
        if any(u["name"] == nombre and a <= u["event"] <= b for u in usados):
            continue
        out.append({"chip": nombre, "hasta": b})
    return out


def _mejor_equipo(X, jug, gws_idx, pesos, presupuesto):
    """Plantilla de 15 que maximiza Σ pesos·(XI + capitán) en las jornadas indicadas. Programación entera."""
    ids = list(X.index)
    n, K = len(ids), len(gws_idx)
    xp = X.values[:, gws_idx]
    pos = jug["pos"].reindex(ids).values
    club = jug["team"].reindex(ids).values
    precio = jug["now_cost"].reindex(ids).values.astype(float)
    # variables: x (n) | s (n·K) | c (n·K)
    nv = n + 2 * n * K
    obj = np.zeros(nv)
    for k in range(K):
        obj[n + k * n: n + (k + 1) * n] = -pesos[k] * xp[:, k]
        obj[n + n * K + k * n: n + n * K + (k + 1) * n] = -pesos[k] * xp[:, k]
    obj[:n] = -0.01 * xp.sum(axis=1)  # desempate: banca útil
    filas, lo, hi = [], [], []

    def fila(coefs):
        r = np.zeros(nv)
        for idx, v in coefs:
            r[idx] += v
        return r

    for p, q in CUPO.items():
        m = np.where(pos == p)[0]
        filas.append(fila([(i, 1) for i in m])); lo.append(q); hi.append(q)
    for t in np.unique(club):
        m = np.where(club == t)[0]
        filas.append(fila([(i, 1) for i in m])); lo.append(0); hi.append(3)
    filas.append(fila([(i, precio[i]) for i in range(n)])); lo.append(0); hi.append(presupuesto)
    for k in range(K):
        so, co = n + k * n, n + n * K + k * n
        filas.append(fila([(so + i, 1) for i in range(n)])); lo.append(11); hi.append(11)
        filas.append(fila([(co + i, 1) for i in range(n)])); lo.append(1); hi.append(1)
        for p in MIN_POS:
            m = np.where(pos == p)[0]
            filas.append(fila([(so + i, 1) for i in m])); lo.append(MIN_POS[p]); hi.append(MAX_POS[p])
    A = np.vstack(filas)
    # s <= x, c <= s  (en forma dispersa sería mejor; con ~600 jugadores y 5 GW sigue siendo manejable)
    extra = []
    for k in range(K):
        so, co = n + k * n, n + n * K + k * n
        for i in range(n):
            r = np.zeros(nv); r[so + i] = 1; r[i] = -1; extra.append(r)
            r = np.zeros(nv); r[co + i] = 1; r[so + i] = -1; extra.append(r)
    B = np.vstack(extra)
    res = milp(obj, integrality=np.ones(nv), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(A, lo, hi), LinearConstraint(B, -np.inf, 0)],
               options={"time_limit": 60})
    if res.x is None:
        return None, 0.0
    elegidos = [ids[i] for i in range(n) if res.x[i] > 0.5]
    return elegidos, float(-res.fun)


def evaluar(matriz, jugadores, estado, gws, gw_sig, boot, cfg, ganancia_plan=0.0):
    j = jugadores.set_index("id")
    plantilla = estado["plantilla"]
    pos = [j.at[i, "pos"] for i in plantilla]
    M = matriz.loc[plantilla].values
    H = len(gws)
    pesos = np.power(cfg["horizonte"]["descuento"], np.arange(H))
    c = cfg.get("chips", {})
    vig = vigentes(boot, estado["chips"], gw_sig)
    if not vig:
        return {"vigentes": [], "detalle": []}

    # Candidatos razonables para el optimizador (acelera mucho y no cambia el resultado)
    tot = (matriz * pesos).sum(axis=1)
    disp = j.index[j["status"] != "u"]
    cand = []
    for p, q in CUPO.items():
        ids = tot[[i for i in disp if j.at[i, "pos"] == p]].sort_values(ascending=False)
        cand += list(ids.head(45).index)
        cand += list(j.loc[[i for i in disp if j.at[i, "pos"] == p]].sort_values("now_cost").head(4).index)
    cand = list(dict.fromkeys(cand + list(plantilla)))
    Xc = matriz.loc[cand]
    presupuesto = estado["banco"] + sum(estado["venta"].values())

    por_gw = []
    for k, gw in enumerate(gws):
        xi, cap, v = mejor_xi(M[:, k], pos)
        banca = [i for i in range(15) if i not in xi]
        fila = {"gw": gw, "xi": round(v, 2), "3xc": round(float(M[cap, k]), 2),
                "capitan": int(plantilla[cap]), "bboost": round(float(M[banca, k].sum()), 2)}
        if any(v["chip"] == "freehit" for v in vig):
            _, mejor = _mejor_equipo(Xc, j, [k], [1.0], presupuesto)
            fila["freehit"] = round(mejor - v, 2)
        por_gw.append(fila)

    wc = None
    if any(v["chip"] == "wildcard" for v in vig):
        eq, val = _mejor_equipo(Xc, j, list(range(H)), pesos, presupuesto)
        actual = sum(pesos[k] * mejor_xi(M[:, k], pos)[2] for k in range(H))
        if eq:
            bruta = val - actual
            # Lo que el Wildcard aporta por encima de lo que ya ganarías con tus transferencias libres
            wc = {"ganancia": round(bruta - ganancia_plan, 2), "bruta": round(bruta, 2),
                  "plan_libres": round(ganancia_plan, 2), "entran": [int(i) for i in eq if i not in plantilla],
                  "salen": [int(i) for i in plantilla if i not in eq]}

    umbral = {"3xc": c.get("triple_capitan", 9.0), "bboost": c.get("bench_boost", 12.0),
              "freehit": c.get("free_hit", 15.0), "wildcard": c.get("wildcard", 20.0)}
    detalle = []
    for v in vig:
        ch = v["chip"]
        if ch == "wildcard":
            g = wc["ganancia"] if wc else 0.0
            usar = g >= umbral[ch]
            detalle.append({"chip": ch, "nombre": NOMBRES[ch], "hasta": v["hasta"], "mejor_gw": gw_sig,
                            "valor": g, "umbral": umbral[ch], "usar": bool(usar), "supera_umbral": bool(usar),
                            "wildcard": wc, "vence_en_horizonte": v["hasta"] <= gws[-1],
                            "restantes": v["hasta"] - gw_sig + 1})
            continue
        validos = [f for f in por_gw if f["gw"] <= v["hasta"] and ch in f]
        if not validos:
            continue
        mejor = max(validos, key=lambda f: f[ch])
        usar_ahora = mejor["gw"] == gw_sig and mejor[ch] >= umbral[ch]
        # Si vence dentro del horizonte, recomendarlo en su mejor jornada aunque no llegue al umbral
        vence = v["hasta"] <= gws[-1]
        detalle.append({"chip": ch, "nombre": NOMBRES[ch], "hasta": v["hasta"], "mejor_gw": mejor["gw"],
                        "valor": mejor[ch], "umbral": umbral[ch],
                        "usar": bool(usar_ahora or (vence and mejor["gw"] == gw_sig)),
                        "supera_umbral": bool(mejor[ch] >= umbral[ch]), "vence_en_horizonte": bool(vence),
                        "por_gw": [{"gw": f["gw"], "valor": f[ch]} for f in validos],
                        "restantes": v["hasta"] - gw_sig + 1,
                        "capitan": mejor.get("capitan") if ch == "3xc" else None})
    return {"vigentes": vig, "detalle": detalle, "por_gw": por_gw}
