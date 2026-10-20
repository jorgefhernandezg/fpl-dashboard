"""Descarga la temporada actual y la ordena en tablas. No guarda historial: todo se recalcula."""
import numpy as np
import pandas as pd

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def descargar(fuente):
    boot = fuente.get("bootstrap-static/")
    fixtures = fuente.get("fixtures/")
    fx = pd.DataFrame(fixtures)
    fx["event"] = _num(fx["event"])
    jugadas = fx[(fx["finished_provisional"] == True) | (fx["finished"] == True)]
    gws = sorted(int(g) for g in jugadas["event"].dropna().unique())
    live = {g: fuente.get("event/%d/live/" % g) for g in gws}
    return {"boot": boot, "fixtures": fixtures, "live": live}


def tablas(raw):
    boot = raw["boot"]
    equipos = pd.DataFrame(boot["teams"])[["id", "name", "short_name"]]
    equipos.columns = ["team", "equipo", "abrev"]

    j = pd.DataFrame(boot["elements"])
    for c in ["now_cost", "selected_by_percent", "chance_of_playing_next_round", "ep_next",
              "penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order",
              "transfers_in_event", "transfers_out_event"]:
        if c not in j.columns:
            j[c] = np.nan
        j[c] = _num(j[c])
    j["pos"] = j["element_type"].map(POS)
    # Proyección de cambio de precio para esta noche (offset 0): ±100% o más = cambio previsto
    def _proy(v):
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict) and it.get("offset") == 0:
                    return it.get("projected_percent")
        return None
    j["proy_precio"] = _num(j["price_change_projections"].apply(_proy)) if "price_change_projections" in j.columns else np.nan
    j["precio"] = j["now_cost"] / 10.0
    j["propiedad"] = j["selected_by_percent"]
    j = j.merge(equipos, on="team", how="left")
    j = j[["id", "web_name", "team", "abrev", "pos", "precio", "now_cost", "propiedad", "status",
           "chance_of_playing_next_round", "news", "ep_next", "penalties_order",
           "direct_freekicks_order", "corners_and_indirect_freekicks_order",
           "transfers_in_event", "transfers_out_event", "proy_precio"]].copy()

    ev = pd.DataFrame(boot["events"])
    ev["deadline"] = pd.to_datetime(ev["deadline_time"], utc=True)

    fx = pd.DataFrame(raw["fixtures"])
    fx["event"] = _num(fx["event"])
    fx["jugado"] = (fx["finished_provisional"] == True) | (fx["finished"] == True)

    # --- equipo-jornada: cuántos partidos jugó cada equipo en cada GW ---
    jug = fx[fx["jugado"]]
    eq_gw = pd.concat([
        jug[["event", "team_h"]].rename(columns={"team_h": "team"}),
        jug[["event", "team_a"]].rename(columns={"team_a": "team"}),
    ]).groupby(["event", "team"]).size().rename("n_part").reset_index().rename(columns={"event": "gw"})
    eq_gw["gw"] = eq_gw["gw"].astype(int)

    # --- estadísticas por jugador y jornada ---
    filas = []
    for gw, data in raw["live"].items():
        for el in data.get("elements", []):
            s = el.get("stats", {}) or {}
            pts_dc = 0
            fixt = []
            for b in el.get("explain", []) or []:
                fixt.append(b.get("fixture"))
                for st in b.get("stats", []) or []:
                    if st.get("identifier") == "defensive_contribution":
                        pts_dc += st.get("points") or 0
            filas.append({
                "gw": int(gw), "element": el["id"],
                "minutos": s.get("minutes", 0) or 0,
                "titular": s.get("starts", 0) or 0,
                "goles": s.get("goals_scored", 0) or 0,
                "asist": s.get("assists", 0) or 0,
                "xg": float(s.get("expected_goals") or 0),
                "xa": float(s.get("expected_assists") or 0),
                "xgc": float(s.get("expected_goals_conceded") or 0),
                "bonus": s.get("bonus", 0) or 0,
                "bps": s.get("bps", 0) or 0,
                "atajadas": s.get("saves", 0) or 0,
                "puntos": s.get("total_points", 0) or 0,
                "dc_hits": pts_dc / 2.0,
            })
    st = pd.DataFrame(filas)

    # Malla completa: cada jugador en cada jornada en que su equipo jugó (ceros si no entró).
    malla = j[["id", "team", "pos"]].rename(columns={"id": "element"}).merge(eq_gw, on="team")
    g = malla.merge(st, on=["gw", "element"], how="left")
    cols = ["minutos", "titular", "goles", "asist", "xg", "xa", "xgc", "bonus", "bps",
            "atajadas", "puntos", "dc_hits"]
    g[cols] = g[cols].fillna(0)
    return {"jugadores": j, "equipos": equipos, "eventos": ev, "fixtures": fx,
            "jug_gw": g, "eq_gw": eq_gw}


def xg_equipos(t):
    """xG a favor y en contra de cada equipo en cada partido jugado."""
    g = t["jug_gw"]
    eq = g.groupby(["team", "gw"]).agg(xg=("xg", "sum"), n=("n_part", "first")).reset_index()
    eq["xg_pp"] = eq["xg"] / eq["n"]
    fx = t["fixtures"]
    jug = fx[fx["jugado"]][["id", "event", "team_h", "team_a"]].copy()
    jug["event"] = jug["event"].astype(int)
    m = eq.set_index(["team", "gw"])["xg_pp"]
    filas = []
    for _, r in jug.iterrows():
        xh = m.get((r["team_h"], r["event"]), np.nan)
        xa = m.get((r["team_a"], r["event"]), np.nan)
        filas.append({"gw": r["event"], "team": r["team_h"], "rival": r["team_a"], "local": True,
                      "xgf": xh, "xga": xa})
        filas.append({"gw": r["event"], "team": r["team_a"], "rival": r["team_h"], "local": False,
                      "xgf": xa, "xga": xh})
    return pd.DataFrame(filas).dropna()
