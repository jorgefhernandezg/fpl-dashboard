"""Proyección de puntos esperados (xPts) por jugador y por jornada futura.

Piezas:
  minutos   -> probabilidad de ser titular, de jugar 60+, minutos esperados
  ataque    -> xG y xA por 90 recientes (pesos que decaen), anclados al promedio de la posición
  defensa   -> probabilidad de portería a cero, goles en contra, contribución defensiva, atajadas
  rivales   -> fuerza de ataque/defensa de cada equipo y ventaja de local, partido a partido
"""
import re

import numpy as np
import pandas as pd

MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


# ---------------------------------------------------------------- utilidades

def pesos(gw, gw_ref, vida_media):
    return np.power(0.5, (gw_ref - gw) / float(vida_media))


def _poisson_pmf(lam, nmax=25):
    lam = np.asarray(lam, dtype=float)[..., None]
    k = np.arange(nmax + 1)
    logf = np.cumsum(np.log(np.maximum(k, 1)))
    return np.exp(k * np.log(np.maximum(lam, 1e-12)) - lam - logf)


def esperanza_piso(lam, div):
    """E[floor(X/div)] con X ~ Poisson(lam). Sirve para atajadas (div 3) y goles en contra (div 2)."""
    pmf = _poisson_pmf(lam)
    k = np.arange(pmf.shape[-1])
    return (pmf * np.floor(k / div)).sum(axis=-1)


def encoger(suma_x, suma_min, prior_90, k):
    """Tasa por 90 encogida: (Σx + k·prior/90) / (Σmin + k) · 90."""
    return 90.0 * (suma_x + k * prior_90 / 90.0) / (suma_min + k)


# ---------------------------------------------------------------- minutos

def modelo_minutos(t, cfg):
    g = t["jug_gw"].copy()
    gw_ref = g["gw"].max()
    g["w"] = pesos(g["gw"], gw_ref, cfg["forma"]["vida_media_jugador"])
    n = g["n_part"].clip(lower=1)
    g["tit_pp"] = (g["titular"] / n).clip(upper=1)
    g["min_pp"] = g["minutos"] / n
    g["entro_suplente"] = ((g["minutos"] > 0) & (g["titular"] == 0)).astype(float)
    g["es_tit"] = (g["titular"] > 0).astype(float)
    g["tit60"] = ((g["titular"] > 0) & (g["min_pp"] >= 60)).astype(float)

    def agg(d):
        w = d["w"]
        sw = w.sum()
        wt = (w * d["es_tit"]).sum()
        wn = (w * (1 - d["es_tit"])).sum()
        return pd.Series({
            "p_tit": ((w * d["tit_pp"]).sum() + 0.5 * 0.15) / (sw + 0.5),
            "min_tit": ((w * d["es_tit"] * d["min_pp"]).sum() + 2 * 78) / (wt + 2),
            "p60_tit": ((w * d["tit60"]).sum() + 2 * 0.8) / (wt + 2),
            "p_sup": ((w * d["entro_suplente"]).sum() + 1 * 0.25) / (wn + 1),
            "min_sup": (d.loc[d["entro_suplente"] > 0, "min_pp"].mean()
                        if d["entro_suplente"].sum() else 18.0),
            "min_total": d["minutos"].sum(),
            "gw_jugadas": (d["minutos"] > 0).sum(),
        })

    m = g.groupby("element").apply(agg, include_groups=False)
    m["min_sup"] = m["min_sup"].clip(5, 35).fillna(18.0)
    m["min_tit"] = m["min_tit"].clip(30, 90)
    return m


def disponibilidad(j, eventos, gws):
    """Factor 0-1 de disponibilidad por jugador y jornada futura, a partir de status, chance y news."""
    dl = eventos.set_index("id")["deadline"]
    ahora = dl[gws[0]] if gws[0] in dl.index else pd.Timestamp.now(tz="UTC")
    filas = {}
    for _, r in j.iterrows():
        st = r["status"]
        ch = r["chance_of_playing_next_round"]
        f1 = 1.0 if st == "a" else (ch / 100.0 if not pd.isna(ch) else (0.5 if st == "d" else 0.0))
        vuelta = None
        mt = re.search(r"(?:back|until) (\d{1,2}) ([A-Z][a-z]{2})", str(r["news"] or ""))
        if mt and mt.group(2) in MESES:
            mes = MESES[mt.group(2)]
            anio = ahora.year + (1 if mes < ahora.month - 6 else 0)
            try:
                vuelta = pd.Timestamp(year=anio, month=mes, day=int(mt.group(1)), tz="UTC")
            except ValueError:
                vuelta = None
        fila = []
        for i, gw in enumerate(gws):
            if st == "a":
                f = 1.0
            elif st == "u":
                f = 0.0
            elif vuelta is not None:
                f = 1.0 if (gw in dl.index and dl[gw] >= vuelta) else (f1 if i == 0 else 0.0)
            elif st == "d":
                f = f1 if i == 0 else min(1.0, f1 + 0.5 * i)
            elif st == "s":
                f = f1 if i == 0 else 1.0
            else:  # lesionado sin fecha
                f = f1 if i == 0 else min(1.0, f1 + 0.2 * i)
            fila.append(f)
        filas[r["id"]] = fila
    return pd.DataFrame.from_dict(filas, orient="index", columns=gws)


# ---------------------------------------------------------------- tasas por 90

def tasas(t, cfg):
    g = t["jug_gw"].copy()
    # Un partido extraordinario (dos penales, un 4-0) no debe definir la tasa de todo el horizonte:
    # se recorta lo que un solo partido puede aportar de xG y de xA.
    tg, ta = cfg["forma"].get("tope_xg_partido"), cfg["forma"].get("tope_xa_partido")
    n = g["n_part"].clip(lower=1)
    if tg:
        g["xg"] = g["xg"].clip(upper=tg * n)
    if ta:
        g["xa"] = g["xa"].clip(upper=ta * n)
    gw_ref = g["gw"].max()
    g["w"] = pesos(g["gw"], gw_ref, cfg["forma"]["vida_media_jugador"])
    f = cfg["forma"]

    # Prior por posición: tasa agregada de jugadores con 270+ minutos (sin pesos)
    tot = g.groupby("element").agg(min=("minutos", "sum"), pos=("pos", "first"))
    reg = tot[tot["min"] >= 270].index
    gp = g[g["element"].isin(reg)].groupby("pos")[["xg", "xa", "bonus", "atajadas", "minutos"]].sum()
    prior = gp[["xg", "xa", "bonus", "atajadas"]].div(gp["minutos"], axis=0) * 90

    # Prior de contribución defensiva: % de partidos 60+ en que cobró
    g60 = g[(g["minutos"] / g["n_part"].clip(lower=1)) >= 60]
    prior_dc = (g60.groupby("pos")["dc_hits"].sum() / g60.groupby("pos")["n_part"].sum()).fillna(0)

    for c in ["xg", "xa", "bonus", "atajadas", "minutos"]:
        g["w_" + c] = g["w"] * g[c]
    a = g.groupby("element").agg(
        pos=("pos", "first"), sx=("w_xg", "sum"), sa=("w_xa", "sum"), sb=("w_bonus", "sum"),
        ss=("w_atajadas", "sum"), sm=("w_minutos", "sum"))
    pr = prior.reindex(a["pos"]).set_index(a.index)
    out = pd.DataFrame(index=a.index)
    out["xg90"] = encoger(a["sx"], a["sm"], pr["xg"].fillna(0), f["k_ataque"])
    out["xa90"] = encoger(a["sa"], a["sm"], pr["xa"].fillna(0), f["k_ataque"])
    out["bonus90"] = encoger(a["sb"], a["sm"], pr["bonus"].fillna(0), f["k_bonus"])
    out["atajadas90"] = encoger(a["ss"], a["sm"], pr["atajadas"].fillna(0), f["k_atajadas"])

    g60 = g60.assign(w=pesos(g60["gw"], gw_ref, f["vida_media_jugador"]))
    d = g60.assign(wh=g60["w"] * g60["dc_hits"], wn=g60["w"] * g60["n_part"]).groupby("element")[["wh", "wn"]].sum()
    d = d.reindex(out.index).fillna(0)
    pdc = prior_dc.reindex(a["pos"]).fillna(0).values
    out["p_dc"] = (d["wh"] + 2 * pdc) / (d["wn"] + 2)
    return out


# ---------------------------------------------------------------- rivales

def fuerza_equipos(partidos, cfg):
    """Ataque y defensa relativos al promedio de la liga (1 = promedio; defensa >1 = concede más)."""
    p = partidos.copy()
    gw_ref = p["gw"].max()
    p["w"] = pesos(p["gw"], gw_ref, cfg["forma"]["vida_media_equipo"])
    mu = p["xgf"].mean()
    c = 3.0
    a = p.assign(wf=p["w"] * p["xgf"], wa=p["w"] * p["xga"]).groupby("team")[["wf", "wa", "w"]].sum()
    fuerza = pd.DataFrame({
        "ataque": (a["wf"] + c * mu) / (a["w"] + c) / mu,
        "defensa": (a["wa"] + c * mu) / (a["w"] + c) / mu,
    })
    loc = p[p["local"]]["xgf"].mean()
    vis = p[~p["local"]]["xgf"].mean()
    n = len(p) / 2
    r = (loc / vis * n + 1.15 * 40) / (n + 40)  # ventaja de local encogida hacia 1.15
    r = float(np.clip(r, 1.0, 1.4))
    return fuerza, mu, r


def partidos_futuros(t, gws):
    fx = t["fixtures"]
    fut = fx[fx["event"].isin(gws)]
    filas = []
    for _, r in fut.iterrows():
        filas.append({"gw": int(r["event"]), "team": r["team_h"], "rival": r["team_a"], "local": True})
        filas.append({"gw": int(r["event"]), "team": r["team_a"], "rival": r["team_h"], "local": False})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- puntos esperados

def proyectar(t, cfg, gws, ajuste_min=None):
    j = t["jugadores"]
    pts = cfg["puntuacion"]
    mins = modelo_minutos(t, cfg)
    tas = tasas(t, cfg)
    from .datos import xg_equipos
    fuerza, mu, r = fuerza_equipos(xg_equipos(t), cfg)
    disp = disponibilidad(j, t["eventos"], gws)
    fut = partidos_futuros(t, gws)

    hf = {True: np.sqrt(r), False: 1 / np.sqrt(r)}
    fut["hf"] = fut["local"].map(hf)
    fut["hf_rival"] = (~fut["local"]).map(hf)
    fz = fuerza.reindex(fut["team"]).values
    fr = fuerza.reindex(fut["rival"]).values
    fut["lam_favor"] = mu * fz[:, 0] * fr[:, 1] * fut["hf"]
    fut["lam_contra"] = mu * fr[:, 0] * fz[:, 1] * fut["hf_rival"]
    fut["p_cs"] = np.exp(-fut["lam_contra"])
    fut["mult_ataque"] = fr[:, 1] * fut["hf"]          # cuánto facilita el rival marcar
    fut["mult_atajadas"] = fr[:, 0] * fut["hf_rival"]  # cuánto ataca el rival

    base = j[["id", "team", "pos"]].set_index("id").join(mins).join(tas)
    base["p_tit"] = base["p_tit"].fillna(0.05)
    base["min_tit"] = base["min_tit"].fillna(75.0)
    base["p60_tit"] = base["p60_tit"].fillna(0.7)
    base["p_sup"] = base["p_sup"].fillna(0.15)
    base["min_sup"] = base["min_sup"].fillna(18.0)
    for c in ["xg90", "xa90", "bonus90", "atajadas90", "p_dc"]:
        base[c] = base[c].fillna(base.groupby("pos")[c].transform("median")).fillna(0)
    if ajuste_min:
        from .ajustes import aplicar_minutos
        base = aplicar_minutos(base, ajuste_min)
    # Para el desglose: el mismo partido contra un rival promedio y en cancha neutral
    fut["mult_ataque_n"] = 1.0
    fut["lam_contra_n"] = mu * fz[:, 1]
    fut["p_cs_n"] = np.exp(-fut["lam_contra_n"])

    filas = []
    for pid, b in base.iterrows():
        eq = fut[fut["team"] == b["team"]]
        for _, p in eq.iterrows():
            f = disp.at[pid, p["gw"]] if pid in disp.index else 1.0
            p_juega = f * (b["p_tit"] + (1 - b["p_tit"]) * b["p_sup"])
            p60 = f * b["p_tit"] * b["p60_tit"]
            m = f * (b["p_tit"] * b["min_tit"] + (1 - b["p_tit"]) * b["p_sup"] * b["min_sup"])
            pos = b["pos"]
            xg = b["xg90"] * m / 90 * p["mult_ataque"]
            xa = b["xa90"] * m / 90 * p["mult_ataque"]
            c = {
                "aparicion": p_juega + p60,
                "goles": xg * pts["gol"][pos],
                "asist": xa * pts["asistencia"],
                "cs": p60 * p["p_cs"] * pts["porteria_cero"][pos],
                "dc": (p60 * b["p_dc"] * pts["contribucion_defensiva"]) if pos != "GKP" else 0.0,
                "bonus": b["bonus90"] * m / 90,
                "conc": 0.0, "atajadas": 0.0,
            }
            if pos in ("GKP", "DEF"):
                c["conc"] = -p60 * float(esperanza_piso(p["lam_contra"], pts["goles_en_contra_por_punto"]))
            if pos == "GKP":
                lam_s = b["atajadas90"] * p["mult_atajadas"]
                c["atajadas"] = p60 * float(esperanza_piso(lam_s, pts["atajadas_por_punto"]))
            # Versión neutral (rival promedio, sin localía): separa el efecto del calendario
            n = {
                "aparicion": c["aparicion"],
                "ataque": (b["xg90"] * pts["gol"][pos] + b["xa90"] * pts["asistencia"]) * m / 90,
                "defensa": p60 * p["p_cs_n"] * pts["porteria_cero"][pos] + c["dc"],
                "bonus": c["bonus"],
            }
            if pos in ("GKP", "DEF"):
                n["defensa"] -= p60 * float(esperanza_piso(p["lam_contra_n"], pts["goles_en_contra_por_punto"]))
            if pos == "GKP":
                n["defensa"] += p60 * float(esperanza_piso(b["atajadas90"], pts["atajadas_por_punto"]))
            total = sum(c.values())
            filas.append({"id": pid, "gw": p["gw"], "rival": p["rival"], "local": p["local"],
                          "xmin": m, "p60": p60, "xg": xg, "xa": xa, "p_cs": p["p_cs"],
                          **c, "xpts": total,
                          "n_aparicion": n["aparicion"], "n_ataque": n["ataque"], "n_defensa": n["defensa"],
                          "n_bonus": n["bonus"], "calendario": total - sum(n.values())})
    det = pd.DataFrame(filas)
    matriz = det.pivot_table(index="id", columns="gw", values="xpts", aggfunc="sum").reindex(
        columns=gws).fillna(0.0)
    matriz = matriz.reindex(j["id"]).fillna(0.0)
    return {"matriz": matriz, "detalle": det, "base": base, "fuerza": fuerza, "mu": mu,
            "ventaja_local": r, "futuros": fut, "disponibilidad": disp}
