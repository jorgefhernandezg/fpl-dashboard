"""Tu plantilla, precios de venta, transferencias libres y búsqueda de movimientos."""
import itertools

import numpy as np
import pandas as pd

MIN_POS = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_POS = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
CUPO = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


# ---------------------------------------------------------------- estado de tu equipo

def estado_equipo(fuente, team_id, t, gw_actual, gw_siguiente):
    entrada = fuente.get("entry/%d/" % team_id)
    hist = fuente.get("entry/%d/history/" % team_id)
    transfers = pd.DataFrame(fuente.get("entry/%d/transfers/" % team_id))
    chips = pd.DataFrame(hist.get("chips", []))
    fh_gws = set(chips.loc[chips["name"] == "freehit", "event"]) if len(chips) else set()

    picks = fuente.get("entry/%d/event/%d/picks/" % (team_id, gw_actual))
    plantilla = [p["element"] for p in picks["picks"]]
    banco = picks["entry_history"]["bank"]
    # Si juegas un Free Hit en la GW actual, la plantilla "real" es la de la GW anterior
    if gw_actual in fh_gws and gw_actual > 1:
        prev = fuente.get("entry/%d/event/%d/picks/" % (team_id, gw_actual - 1))
        plantilla = [p["element"] for p in prev["picks"]]
        banco = prev["entry_history"]["bank"]

    # Transferencias ya hechas para la siguiente jornada (antes del deadline)
    pendientes = transfers[transfers["event"] == gw_siguiente] if len(transfers) else transfers
    for _, tr in pendientes.sort_values("time").iterrows() if len(pendientes) else []:
        if tr["element_out"] in plantilla:
            plantilla[plantilla.index(tr["element_out"])] = tr["element_in"]
            banco += tr["element_out_cost"] - tr["element_in_cost"]

    # Precio de compra: última vez que entró (ignorando Free Hit); si no, precio al crear el equipo
    validas = transfers[~transfers["event"].isin(fh_gws)] if len(transfers) else transfers
    compra = {}
    if len(validas):
        for _, tr in validas.sort_values("time").iterrows():
            compra[tr["element_in"]] = tr["element_in_cost"]
    inicio = entrada.get("started_event") or 1
    ahora = t["jugadores"].set_index("id")["now_cost"]
    venta = {}
    for e in plantilla:
        if e not in compra:
            try:
                es = fuente.get("element-summary/%d/" % e)
                h = [x for x in es.get("history", []) if x.get("round", 0) >= inicio]
                compra[e] = h[0]["value"] if h else int(ahora.get(e, 0))
            except Exception:
                compra[e] = int(ahora.get(e, 0))
        p0, p1 = int(compra[e]), int(ahora.get(e, compra[e]))
        venta[e] = p0 + (p1 - p0) // 2 if p1 > p0 else p1

    # Transferencias libres (estimación con las reglas actuales: +1 por jornada, máximo 5)
    ft = 1
    filas_h = sorted(hist.get("current", []), key=lambda x: x["event"])
    chip_gw = {int(r["event"]): r["name"] for _, r in chips.iterrows()} if len(chips) else {}
    for h in filas_h:
        g = h["event"]
        if g <= inicio:
            ft = 1
            continue
        if g > gw_actual:
            break
        if chip_gw.get(g) in ("wildcard", "freehit"):
            ft = min(5, ft + 1)
        else:
            ft = min(5, max(ft - h["event_transfers"], 0) + 1)
    ft = max(0, ft - len(pendientes))

    return {
        "entrada": entrada, "historial": filas_h, "chips": chips.to_dict("records") if len(chips) else [],
        "plantilla": plantilla, "banco": int(banco), "venta": venta, "compra": compra,
        "ft_estimadas": int(ft), "pendientes": pendientes.to_dict("records") if len(pendientes) else [],
        "picks_actual": picks,
    }


# ---------------------------------------------------------------- alineación óptima

def mejor_xi(xp, pos):
    """xp: puntos de los 15 en una jornada; pos: posiciones. Devuelve (índices XI, capitán, valor)."""
    orden = np.argsort(-xp)
    elegidos, cuenta = [], {k: 0 for k in MIN_POS}
    for p in MIN_POS:  # mínimos por posición
        for i in orden:
            if pos[i] == p and cuenta[p] < MIN_POS[p]:
                elegidos.append(i)
                cuenta[p] += 1
    for i in orden:  # resto, respetando máximos
        if len(elegidos) == 11:
            break
        if i not in elegidos and cuenta[pos[i]] < MAX_POS[pos[i]]:
            elegidos.append(i)
            cuenta[pos[i]] += 1
    cap = max(elegidos, key=lambda i: xp[i])
    return elegidos, cap, float(xp[elegidos].sum() + xp[cap])


def valor(ids, X, pos_de, pesos_gw):
    """Valor descontado de una plantilla: suma de (mejor XI + capitán) por jornada.

    X puede ser un DataFrame (índice = id) o una tupla (array, {id: fila}) para ir más rápido."""
    if isinstance(X, tuple):
        arr, fila = X
        M = arr[[fila[i] for i in ids]]
    else:
        M = X.loc[ids].values
    pos = [pos_de[i] for i in ids]
    tot = np.zeros(M.shape[1])
    for k in range(M.shape[1]):
        tot[k] = mejor_xi(M[:, k], pos)[2]
    return tot * pesos_gw


# ---------------------------------------------------------------- búsqueda

def buscar(estado, matriz, jugadores, cfg, horizontes=(1, 3)):
    j = jugadores.set_index("id")
    H = matriz.shape[1]
    pesos_gw = np.power(cfg["horizonte"]["descuento"], np.arange(H))
    pos_de = j["pos"].to_dict()
    club = j["team"].to_dict()
    precio = j["now_cost"].to_dict()
    plantilla = list(estado["plantilla"])
    rapido = (matriz.values, {e: k for k, e in enumerate(matriz.index)})
    base = valor(plantilla, rapido, pos_de, pesos_gw)
    banco = estado["banco"]

    disponibles = j[(j["status"] != "u")].index.difference(plantilla)
    total_h = (matriz.loc[disponibles] * pesos_gw).sum(axis=1)

    def clubes(ids):
        c = {}
        for i in ids:
            c[club[i]] = c.get(club[i], 0) + 1
        return c

    def ganancia(nueva):
        v = valor(nueva, rapido, pos_de, pesos_gw) - base
        out = {"h%d" % h: float(v[:h].sum()) for h in horizontes if h < H}
        out["h%d" % H] = float(v.sum())
        return out

    # --- simples ---
    simples = []
    for sale in plantilla:
        p = pos_de[sale]
        presupuesto = banco + estado["venta"][sale]
        cand = [e for e in disponibles if pos_de[e] == p and precio[e] <= presupuesto]
        resto = [x for x in plantilla if x != sale]
        cr = clubes(resto)
        for entra in cand:
            if cr.get(club[entra], 0) >= 3:
                continue
            nueva = resto + [entra]
            g = ganancia(nueva)
            simples.append({"sale": sale, "entra": entra, "banco_final": presupuesto - precio[entra], **g})
    simples = sorted(simples, key=lambda d: -d["h%d" % H])

    # --- dobles (sobre los mejores candidatos de cada posición) ---
    n_cand = cfg["decision"]["candidatos_por_posicion"]
    top = {p: list(total_h[[pos_de[e] == p for e in total_h.index]].sort_values(ascending=False)
                   .head(n_cand).index) for p in CUPO}
    dobles = []
    for s1, s2 in itertools.combinations(plantilla, 2):
        presupuesto = banco + estado["venta"][s1] + estado["venta"][s2]
        resto = [x for x in plantilla if x not in (s1, s2)]
        cr = clubes(resto)
        for e1 in top[pos_de[s1]]:
            for e2 in top[pos_de[s2]]:
                if e1 == e2 or precio[e1] + precio[e2] > presupuesto:
                    continue
                if pos_de[s1] == pos_de[s2] and e1 > e2:
                    continue
                c = dict(cr)
                ok = True
                for e in (e1, e2):
                    c[club[e]] = c.get(club[e], 0) + 1
                    if c[club[e]] > 3:
                        ok = False
                if not ok:
                    continue
                g = ganancia(resto + [e1, e2])
                dobles.append({"sale": [s1, s2], "entra": [e1, e2],
                               "banco_final": presupuesto - precio[e1] - precio[e2], **g})
    dobles = sorted(dobles, key=lambda d: -d["h%d" % H])

    return {"simples": simples, "dobles": dobles, "valor_base": base.tolist(), "pesos": pesos_gw.tolist()}


def decidir(busq, ft, cfg, H):
    d = cfg["decision"]
    clave = "h%d" % H
    umbral_ft = d["umbral_transferencia_libre"] if ft < 5 else 0.0
    opciones = [{"tipo": "guardar", "neto": 0.0, "ganancia": 0.0, "coste": 0}]
    for n, lista in ((1, busq["simples"]), (2, busq["dobles"])):
        if not lista:
            continue
        mejor = lista[0]
        golpes = max(0, n - ft)
        libres = n - golpes
        coste = 4 * golpes
        neto = mejor[clave] - coste - libres * umbral_ft - golpes * d["margen_golpe"]
        opciones.append({"tipo": "simple" if n == 1 else "doble", "neto": float(neto),
                         "ganancia": float(mejor[clave]), "coste": coste, "mov": mejor})
    elegida = max(opciones, key=lambda o: o["neto"])
    return {"elegida": elegida, "opciones": opciones}
