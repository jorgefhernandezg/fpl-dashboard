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


# Varianza de los puntos de una jornada ≈ razón × puntos esperados (medido en 2025-26 con titulares)
RAZON_VARIANZA = {"GKP": 2.2, "DEF": 2.8, "MID": 2.4, "FWD": 3.1}


class Evaluador:
    """Valor esperado y riesgo frente al campo de una plantilla, desde la columna `desde` del horizonte.

    Riesgo: desviación típica de (tus puntos − puntos del campo). Cada jugador aporta
    (multiplicador_tuyo − propiedad_efectiva)² × varianza. No tener a alguien muy poseído suma riesgo;
    tenerlo lo neutraliza."""

    def __init__(self, matriz, jugadores, cfg, eo, desde=0):
        j = jugadores.set_index("id")
        H = matriz.shape[1]
        self.pesos = np.power(cfg["horizonte"]["descuento"], np.arange(H))[desde:]
        self.arr = matriz.values[:, desde:]
        self.fila = {e: k for k, e in enumerate(matriz.index)}
        self.pos_de = j["pos"].to_dict()
        self.club = j["team"].to_dict()
        self.precio = j["now_cost"].to_dict()
        self.status = j["status"].to_dict()
        self.eo = eo.reindex(matriz.index).fillna(0).values
        razon = np.array([RAZON_VARIANZA.get(self.pos_de.get(e), 2.5) for e in matriz.index])
        self.var = self.arr.clip(min=0) * razon[:, None]
        self.C = (self.eo[:, None] ** 2 * self.var).sum(axis=0)
        self.lam = cfg["decision"].get("aversion_riesgo", 0.0)
        self.index = matriz.index

    def evaluar(self, ids):
        filas = [self.fila[i] for i in ids]
        M = self.arr[filas]
        pos = [self.pos_de[i] for i in ids]
        eo = self.eo[filas]
        val = np.zeros(M.shape[1])
        rie = np.zeros(M.shape[1])
        for k in range(M.shape[1]):
            xi, cap, v = mejor_xi(M[:, k], pos)
            val[k] = v
            m = np.zeros(len(ids))
            m[xi] = 1
            m[cap] = 2
            r2 = self.C[k] + (((m - eo) ** 2 - eo ** 2) * self.var[filas, k]).sum()
            rie[k] = np.sqrt(max(r2, 0))
        return val * self.pesos, rie * self.pesos


def buscar(ev, plantilla, banco, venta, cfg, n_cand=None, dobles=True):
    """Todas las transferencias simples y las dobles entre los mejores candidatos por posición."""
    H = len(ev.pesos)
    base_v, base_r = ev.evaluar(plantilla)
    disponibles = [e for e in ev.index if ev.status.get(e) != "u" and e not in set(plantilla)]
    tot = {e: float((ev.arr[ev.fila[e]] * ev.pesos).sum()) for e in disponibles}

    def clubes(ids):
        c = {}
        for i in ids:
            c[ev.club[i]] = c.get(ev.club[i], 0) + 1
        return c

    def registro(nueva, sale, entra, bf):
        v, r = ev.evaluar(nueva)
        dv, dr = v - base_v, r - base_r
        out = {"sale": sale, "entra": entra, "banco_final": int(bf),
               "h1": float(dv[:1].sum()), "h3": float(dv[:3].sum()), "hT": float(dv.sum()),
               "riesgo": float(dr.sum()), "riesgo1": float(dr[:1].sum())}
        out["puntaje"] = out["hT"] - ev.lam * out["riesgo"]
        return out

    simples = []
    for sale in plantilla:
        p = ev.pos_de[sale]
        presupuesto = banco + venta[sale]
        resto = [x for x in plantilla if x != sale]
        cr = clubes(resto)
        for entra in disponibles:
            if ev.pos_de[entra] != p or ev.precio[entra] > presupuesto or cr.get(ev.club[entra], 0) >= 3:
                continue
            simples.append(registro(resto + [entra], sale, entra, presupuesto - ev.precio[entra]))
    simples.sort(key=lambda d: -d["puntaje"])

    lista_d = []
    if dobles:
        n_cand = n_cand or cfg["decision"]["candidatos_por_posicion"]
        top = {}
        for p in CUPO:
            cand = sorted([e for e in disponibles if ev.pos_de[e] == p], key=lambda e: -tot[e])
            top[p] = cand[:n_cand]
        for s1, s2 in itertools.combinations(plantilla, 2):
            presupuesto = banco + venta[s1] + venta[s2]
            resto = [x for x in plantilla if x not in (s1, s2)]
            cr = clubes(resto)
            for e1 in top[ev.pos_de[s1]]:
                for e2 in top[ev.pos_de[s2]]:
                    if e1 == e2 or ev.precio[e1] + ev.precio[e2] > presupuesto:
                        continue
                    if ev.pos_de[s1] == ev.pos_de[s2] and e1 > e2:
                        continue
                    c = dict(cr)
                    ok = True
                    for e in (e1, e2):
                        c[ev.club[e]] = c.get(ev.club[e], 0) + 1
                        ok = ok and c[ev.club[e]] <= 3
                    if ok:
                        lista_d.append(registro(resto + [e1, e2], [s1, s2], [e1, e2],
                                                presupuesto - ev.precio[e1] - ev.precio[e2]))
        lista_d.sort(key=lambda d: -d["puntaje"])
    return {"simples": simples, "dobles": lista_d, "base": base_v.tolist(), "riesgo_base": base_r.tolist()}


def aplicar(plantilla, banco, venta, mov, ev):
    sale =mov["sale"] if isinstance(mov["sale"], list) else [mov["sale"]]
    entra = mov["entra"] if isinstance(mov["entra"], list) else [mov["entra"]]
    nueva = list(plantilla)
    v = dict(venta)
    for s, e in zip(sale, entra):
        nueva[nueva.index(s)] = e
        banco += venta[s] - ev.precio[e]
        v[e] = ev.precio[e]
    return nueva, banco, v


def planificar(matriz, jugadores, cfg, eo, estado, ft, busq_ahora):
    """Compara planes de 2 semanas usando solo transferencias libres (o con golpes, si se permiten).

    Valor de un plan = ganancia esta semana + ganancia la próxima (sobre la plantilla resultante)
    + valor de las transferencias libres que quedan guardadas al final."""
    d = cfg["decision"]
    golpes_ok = d.get("permitir_golpes", False)
    v_ft = d["umbral_transferencia_libre"]
    ev2 = Evaluador(matriz, jugadores, cfg, eo, desde=1)
    ev1 = Evaluador(matriz, jugadores, cfg, eo, desde=0)
    plantilla, banco, venta = list(estado["plantilla"]), estado["banco"], dict(estado["venta"])

    def coste(n, libres):
        golpes = max(0, n - libres)
        return 4 * golpes + golpes * d["margen_golpe"], golpes

    def max_mov(libres):
        return 2 if golpes_ok else min(2, libres)

    primeras = [(0, None)]
    if max_mov(ft) >= 1:
        primeras += [(1, m) for m in busq_ahora["simples"][:6]]
    if max_mov(ft) >= 2:
        primeras += [(2, m) for m in busq_ahora["dobles"][:4]]

    planes = []
    for n1, m1 in primeras:
        c1, g1 = coste(n1, ft)
        if m1 is None:
            p1, b1, v1, val1 = plantilla, banco, venta, 0.0
        else:
            p1, b1, v1 = aplicar(plantilla, banco, venta, m1, ev1)
            val1 = m1["puntaje"]
        ft2 = min(5, max(ft - n1, 0) + 1)
        b2 = buscar(ev2, p1, b1, v1, cfg, n_cand=10, dobles=max_mov(ft2) >= 2)
        segundas = [(0, None)]
        if max_mov(ft2) >= 1 and b2["simples"]:
            segundas.append((1, b2["simples"][0]))
        if max_mov(ft2) >= 2 and b2["dobles"]:
            segundas.append((2, b2["dobles"][0]))
        for n2, m2 in segundas:
            c2, g2 = coste(n2, ft2)
            ft3 = min(5, max(ft2 - n2, 0) + 1)
            val2 = m2["puntaje"] if m2 else 0.0
            total = val1 - c1 + val2 - c2 + v_ft * (ft3 - 1)
            planes.append({"total": float(total), "ft_final": ft3, "golpes": g1 + g2,
                           "pasos": [{"n": n1, "mov": m1, "valor": float(val1 - c1)},
                                     {"n": n2, "mov": m2, "valor": float(val2 - c2), "ft": ft2}]})
    planes.sort(key=lambda p: -p["total"])
    # quitar planes repetidos en su primer paso para mostrar alternativas distintas
    vistos, unicos = set(), []
    for p in planes:
        clave = (p["pasos"][0]["n"], str(p["pasos"][0]["mov"] and p["pasos"][0]["mov"]["entra"]),
                 p["pasos"][1]["n"], str(p["pasos"][1]["mov"] and p["pasos"][1]["mov"]["entra"]))
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(p)
    return unicos
