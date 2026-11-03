"""Plan de chips hasta su vencimiento.

Valor proyectado de cada chip:
  Triple capitán -> puntos extra del mejor capitán de tu plantilla
  Bench Boost    -> puntos de tus 4 suplentes
  Free Hit       -> mejor equipo posible para esa jornada (presupuesto = valor de venta + banco) menos tu XI
  Wildcard       -> mejor plantilla para todo el horizonte menos la tuya actual, menos tu plan de transferencias

Lo que se midió en 2025-26 (41 jornadas normales, 9 Wildcards) y decide las reglas:
  - Valor real ≈ proyectado × realismo: Bench Boost 0.80 (IC90 0.69-0.90), Triple capitán 0.92,
    Free Hit ~1.0 (muy ruidoso), Wildcard ~0.5 (IC90 −0.2 a 1.2: dirección segura, magnitud no).
  - En jornadas normales, el valor proyectado NO distingue qué jornada es mejor para un chip
    (correlación proyectado-real ≈ 0). El momento lo deciden razones estructurales: jornadas dobles o
    en blanco, el Bench Boost tras el Wildcard, y bajas que justifiquen el Free Hit.
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


def _mejor_equipo(X, jug, gws_idx, pesos, presupuesto, bb=None):
    """Plantilla de 15 que maximiza Σ pesos·(XI + capitán) en las jornadas indicadas. Programación entera.

    bb: posición (dentro de gws_idx) de una jornada con Bench Boost, donde suman los 15."""
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
    if bb is not None:
        obj[n + bb * n: n + (bb + 1) * n] = 0.0
        obj[:n] += -pesos[bb] * xp[:, bb]
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


def evaluar(matriz, jugadores, estado, gws, gw_sig, boot, cfg, ganancia_plan=0.0, extra=None):
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

    # ¿Conviene encadenar Wildcard y luego Bench Boost? (no se pueden usar dos chips en la misma jornada)
    combo = None
    tiene = {v["chip"] for v in vig}
    if wc and {"wildcard", "bboost"} <= tiene:
        opciones = []
        for k in range(1, H):
            _, v_combo = _mejor_equipo(Xc, j, list(range(H)), pesos, presupuesto, bb=k)
            banca_wc = (v_combo - (wc["bruta"] + actual)) / pesos[k]   # banca obtenible, neta del sacrificio
            opciones.append({"gw": gws[k], "banca_tras_wc": round(banca_wc, 2),
                             "banca_actual": por_gw[k]["bboost"],
                             "ventaja": round(banca_wc - por_gw[k]["bboost"], 2)})
        if opciones:
            combo = {"wildcard_gw": gw_sig, "opciones": opciones,
                     "mejor": max(opciones, key=lambda o: o["banca_tras_wc"])}

    plan = planificar(vig, por_gw, wc, combo, gws, gw_sig, cfg, extra or {})
    return {"vigentes": vig, "detalle": plan["detalle"], "por_gw": por_gw, "combo_wc_bb": combo,
            "calendario": (extra or {}).get("calendario"), "realismo": plan["realismo"]}


REALISMO = {"3xc": 0.92, "bboost": 0.80, "freehit": 1.0, "wildcard": 0.5}


def calendario_hasta(fixtures, equipos, desde, hasta):
    """Jornadas dobles y en blanco entre `desde` y `hasta`, y partidos aún sin fecha."""
    fx = fixtures
    fut = fx[fx["event"].between(desde, hasta)]
    cuenta = pd.concat([fut[["event", "team_h"]].rename(columns={"team_h": "t"}),
                        fut[["event", "team_a"]].rename(columns={"team_a": "t"})]).groupby(["event", "t"]).size()
    abrev = equipos.set_index("team")["abrev"]
    especiales = []
    for gw in range(desde, hasta + 1):
        c = cuenta[cuenta.index.get_level_values(0) == gw] if len(cuenta) else cuenta
        por_eq = {int(t): int(v) for (_, t), v in c.items()}
        dobles = [abrev.get(t, t) for t, v in por_eq.items() if v >= 2]
        blancas = [abrev.get(t, t) for t in abrev.index if por_eq.get(t, 0) == 0]
        if dobles or blancas:
            especiales.append({"gw": gw, "dobles": dobles, "blancas": blancas})
    sin_fecha = int(fx["event"].isna().sum())
    return {"desde": desde, "hasta": hasta, "especiales": especiales, "sin_fecha": sin_fecha}


def planificar(vig, por_gw, wc, combo, gws, gw_sig, cfg, extra):
    c = cfg.get("chips", {})
    real = dict(REALISMO, **(c.get("realismo", {}) or {}))
    tiene = {v["chip"]: v["hasta"] for v in vig}
    cal = extra.get("calendario") or {"especiales": []}
    esp = {e["gw"]: e for e in cal.get("especiales", [])}
    bajas = extra.get("bajas", [])
    n_bajas = len(bajas)
    lim_bajas = int(c.get("free_hit_bajas", 3))
    umbral_wc = float(c.get("wildcard", 10.0))

    # Fechas límite escalonadas: un chip por jornada. Si vencen juntos, el Free Hit va al final
    # (es el seguro), antes el Bench Boost y antes el Wildcard (el Bench Boost va tras el Wildcard).
    limite = {}
    hastas = sorted(set(tiene.values()))
    for h in hastas:
        orden = [ch for ch in ("freehit", "bboost", "wildcard", "3xc") if tiene.get(ch) == h]
        for i, ch in enumerate(orden):
            limite[ch] = h - i
    detalle = []
    vgw = {f["gw"]: f for f in por_gw}

    def fila(ch, estado, gw_uso, razon, valor=None, por=True, **kw):
        d = {"chip": ch, "nombre": NOMBRES[ch], "hasta": tiene[ch], "limite": limite[ch],
             "restantes": tiene[ch] - gw_sig + 1, "estado": estado, "gw_uso": gw_uso, "razon": razon,
             "valor": None if valor is None else round(valor, 2),
             "realista": None if valor is None else round(valor * real[ch], 2),
             "realismo": real[ch], "usar": estado == "usar"}
        if por:
            d["por_gw"] = [{"gw": f["gw"], "valor": f.get(ch)} for f in por_gw if f.get(ch) is not None]
        d.update(kw)
        detalle.append(d)

    dobles = [g for g, e in esp.items() if e["dobles"]]
    blancas = [g for g, e in esp.items() if e["blancas"]]

    # --- Wildcard
    if "wildcard" in tiene:
        g = wc["ganancia"] if wc else 0.0
        realista = g * real["wildcard"]
        prox_esp = min([x for x in dobles + blancas if x <= tiene["wildcard"]], default=None)
        if gw_sig >= limite["wildcard"]:
            fila("wildcard", "usar", gw_sig, "Es tu última jornada para usarlo sin quitarle su semana al Bench "
                 "Boost o al Free Hit.", g, por=False, wildcard=wc)
        elif prox_esp and prox_esp - 1 == gw_sig and "bboost" in tiene and prox_esp in dobles:
            fila("wildcard", "usar", gw_sig, "La GW%d es doble: con el Wildcard ahora armas la plantilla para "
                 "usar el Bench Boost en ella." % prox_esp, g, por=False, wildcard=wc)
        elif prox_esp and prox_esp in dobles and "bboost" in tiene and prox_esp - 1 > gw_sig:
            fila("wildcard", "planificado", prox_esp - 1, "La GW%d es doble: usa el Wildcard en la GW%d para armar "
                 "la plantilla y el Bench Boost en la doble." % (prox_esp, prox_esp - 1), g, por=False, wildcard=wc)
        elif realista >= umbral_wc:
            fila("wildcard", "usar", gw_sig, "La mejora realista (%.1f pts por encima de tu plan de transferencias "
                 "libres) supera tu umbral de %.0f." % (realista, umbral_wc), g, por=False, wildcard=wc)
        else:
            fila("wildcard", "reservar", None, "La mejora realista (%.1f pts sobre tu plan de transferencias) no "
                 "llega a tu umbral de %.0f. Úsalo cuando quieras hacer 4 o más cambios; a más tardar en la GW%d."
                 % (realista, umbral_wc, limite["wildcard"]), g, por=False, wildcard=wc)

    # --- Bench Boost
    if "bboost" in tiene:
        en_doble = [x for x in dobles if x <= tiene["bboost"]]
        wc_ahora = any(d["chip"] == "wildcard" and d["estado"] == "usar" for d in detalle)
        if en_doble:
            gu = en_doble[0]
            fila("bboost", "usar" if gu == gw_sig else "planificado", gu,
                 "La GW%d es doble (%s): tus suplentes pueden jugar dos partidos."
                 % (gu, ", ".join(esp[gu]["dobles"])), vgw.get(gu, {}).get("bboost"))
        elif "wildcard" in tiene:
            gu = gw_sig + 1 if wc_ahora else None
            mejor = (combo or {}).get("mejor") or {}
            txt = ("Úsalo en la jornada siguiente al Wildcard: tu banca pasaría de unos %.1f a unos %.1f puntos "
                   "proyectados (%.1f reales)." % (mejor.get("banca_actual", 0), mejor.get("banca_tras_wc", 0),
                                                   mejor.get("banca_tras_wc", 0) * real["bboost"])) if mejor else \
                "Úsalo en la jornada siguiente al Wildcard, cuando tu banca sea más fuerte."
            fila("bboost", "planificado" if wc_ahora else "reservar", gu, txt + " A más tardar en la GW%d."
                 % limite["bboost"], mejor.get("banca_tras_wc"))
        elif gw_sig >= limite["bboost"]:
            fila("bboost", "usar", gw_sig, "Es tu última jornada para usarlo antes de que venza.",
                 vgw.get(gw_sig, {}).get("bboost"))
        else:
            fila("bboost", "reservar", None, "Sin jornadas dobles, todas las jornadas valen parecido: la proyección "
                 "no distingue cuál es mejor. Úsalo cuando tus 4 suplentes tengan partido y minutos; a más tardar "
                 "en la GW%d." % limite["bboost"], vgw.get(gw_sig, {}).get("bboost"))

    # --- Free Hit
    if "freehit" in tiene:
        en_blanca = [x for x in blancas if x <= tiene["freehit"]]
        if n_bajas >= lim_bajas:
            fila("freehit", "usar", gw_sig, "Tienes %d titulares que probablemente no jueguen la GW%d (%s): es el "
                 "caso para el que conviene guardarlo." % (n_bajas, gw_sig, ", ".join(b["n"] for b in bajas)),
                 vgw.get(gw_sig, {}).get("freehit"), bajas=bajas)
        elif en_blanca:
            gu = en_blanca[0]
            fila("freehit", "usar" if gu == gw_sig else "planificado", gu,
                 "La GW%d tiene equipos sin partido (%s)." % (gu, ", ".join(esp[gu]["blancas"])),
                 vgw.get(gu, {}).get("freehit"), bajas=bajas)
        elif gw_sig >= limite["freehit"]:
            fila("freehit", "usar", gw_sig, "Es tu última jornada para usarlo antes de que venza.",
                 vgw.get(gw_sig, {}).get("freehit"), bajas=bajas)
        else:
            fila("freehit", "reservar", None, "Guárdalo como seguro para una jornada con %d o más titulares sin "
                 "jugar (hoy: %d). Si no pasa, úsalo en la GW%d." % (lim_bajas, n_bajas, limite["freehit"]),
                 vgw.get(gw_sig, {}).get("freehit"), bajas=bajas)

    # --- Triple capitán
    if "3xc" in tiene:
        en_doble = [x for x in dobles if x <= tiene["3xc"]]
        if en_doble:
            gu = en_doble[0]
            fila("3xc", "usar" if gu == gw_sig else "planificado", gu, "La GW%d es doble: tu capitán puede jugar "
                 "dos partidos." % gu, vgw.get(gu, {}).get("3xc"))
        elif gw_sig >= limite["3xc"]:
            fila("3xc", "usar", gw_sig, "Es tu última jornada para usarlo antes de que venza.",
                 vgw.get(gw_sig, {}).get("3xc"))
        else:
            fila("3xc", "reservar", None, "Sin jornadas dobles, la proyección no distingue cuál jornada es mejor. "
                 "Úsalo cuando tu mejor capitán juegue de local contra un rival débil; a más tardar en la GW%d."
                 % limite["3xc"], vgw.get(gw_sig, {}).get("3xc"))
    # Un solo chip por jornada: si dos quedaron para esta semana, se queda el de fecha límite más cercana
    usar = [d for d in detalle if d["estado"] == "usar"]
    if len(usar) > 1:
        usar.sort(key=lambda d: (d["limite"], d["chip"] != "wildcard"))
        for d in usar[1:]:
            d["estado"], d["usar"] = "reservar", False
            d["razon"] += " (Esta semana va primero el %s: solo se puede usar un chip por jornada.)" % usar[0]["nombre"]
    return {"detalle": detalle, "realismo": real}
