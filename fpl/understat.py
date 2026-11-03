"""Descarga los remates de la Premier League desde Understat y guarda los penales por jugador y partido.

Understat no tiene API oficial: los datos vienen embebidos en sus páginas (JSON.parse('...')) o en sus
endpoints internos (getLeagueData / getMatchData). Se prueban ambos formatos y se deja un diagnóstico.

Uso:
    python -m fpl.understat 2025 2026     # temporadas que empiezan en esos años (2025 = 2025-26)
Salida:
    data/understat/penales_<año>.json     # partidos procesados + penales (jugador, equipo, fecha, xG)
    data/understat/_diagnostico.json      # qué formato funcionó, conteos, errores
"""
import codecs
import json
import os
import re
import sys
import time

import requests

BASE = "https://understat.com"
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "data", "understat")
CABECERAS = {"User-Agent": "Mozilla/5.0 (fpl-dashboard personal)", "Accept-Language": "en"}
XHR = dict(CABECERAS, **{"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"})


def _embebido(html, var):
    m = re.search(r"var\s+%s\s*=\s*JSON\.parse\('(.+?)'\)" % var, html, re.S)
    if not m:
        return None
    return json.loads(codecs.decode(m.group(1), "unicode_escape"))


def _get(s, url, xhr=False, intentos=3):
    ultimo = None
    for k in range(intentos):
        try:
            r = s.get(url, headers=XHR if xhr else CABECERAS, timeout=(20, 60))
            r.raise_for_status()
            time.sleep(0.8)
            return r
        except requests.RequestException as e:
            ultimo = e
            time.sleep(5 * (k + 1))
    raise ultimo


def sonda():
    """Prueba de conectividad para distinguir un corte temporal de un bloqueo."""
    import socket
    out = {}
    try:
        out["dns"] = socket.gethostbyname("understat.com")
    except Exception as e:
        out["dns"] = repr(e)[:200]
    for url in ("https://understat.com/", "http://understat.com/", "https://www.understat.com/",
                "https://fantasy.premierleague.com/api/event-status/"):
        try:
            r = requests.get(url, headers=CABECERAS, timeout=(15, 30), allow_redirects=False)
            out[url] = r.status_code
        except Exception as e:
            out[url] = type(e).__name__
    return out


def liga(s, anio, diag):
    """Lista de partidos de la temporada: id, fecha, equipos, si terminó."""
    partidos = None
    try:
        html = _get(s, "%s/league/EPL/%d" % (BASE, anio)).text
        partidos = _embebido(html, "datesData")
        diag["liga_formato"] = "embebido" if partidos else "sin datesData en la página"
    except Exception as e:
        diag["liga_error_pagina"] = repr(e)[:300]
    if not partidos:
        try:
            j = _get(s, "%s/getLeagueData/EPL/%d" % (BASE, anio), xhr=True).json()
            partidos = j.get("dates") or j.get("datesData")
            diag["liga_formato"] = "getLeagueData"
        except Exception as e:
            diag["liga_error_api"] = repr(e)[:300]
    return partidos or []


def remates(s, mid, diag):
    try:
        html = _get(s, "%s/match/%s" % (BASE, mid)).text
        r = _embebido(html, "shotsData")
        if r:
            diag["partido_formato"] = "embebido"
            return r
    except Exception as e:
        diag["partido_error_pagina"] = repr(e)[:300]
    try:
        j = _get(s, "%s/getMatchData/%s" % (BASE, mid), xhr=True).json()
        diag["partido_formato"] = "getMatchData"
        return j.get("shots") or j.get("shotsData") or j
    except Exception as e:
        diag["partido_error_api"] = repr(e)[:300]
    return None


def actualizar(anio, s, diag_global):
    ruta = os.path.join(SALIDA, "penales_%d.json" % anio)
    previo = json.load(open(ruta)) if os.path.exists(ruta) else {"partidos": {}, "penales": []}
    diag = {"temporada": anio}
    partidos = liga(s, anio, diag)
    terminados = [p for p in partidos if str(p.get("isResult")).lower() in ("true", "1")]
    diag["partidos_temporada"] = len(partidos)
    diag["terminados"] = len(terminados)
    nuevos = [p for p in terminados if str(p["id"]) not in previo["partidos"]]
    situaciones = {}
    fallos = seguidos = 0
    for p in nuevos:
        sh = remates(s, p["id"], diag)
        if not sh:
            fallos += 1
            seguidos += 1
            if seguidos >= 5:
                diag["abortado"] = "5 partidos seguidos sin datos"
                break
            continue
        seguidos = 0
        lista = (sh.get("h", []) + sh.get("a", [])) if isinstance(sh, dict) else list(sh)
        n_pen = 0
        for x in lista:
            sit = x.get("situation")
            situaciones[sit] = situaciones.get(sit, 0) + 1
            if sit == "Penalty":
                n_pen += 1
                previo["penales"].append({
                    "partido": str(p["id"]), "fecha": p.get("datetime"),
                    "local": p.get("h", {}).get("title"), "visita": p.get("a", {}).get("title"),
                    "lado": x.get("h_a"), "jugador": x.get("player"), "jugador_id": x.get("player_id"),
                    "xg": float(x.get("xG") or 0), "resultado": x.get("result"), "minuto": x.get("minute"),
                })
        previo["partidos"][str(p["id"])] = {"fecha": p.get("datetime"), "local": p.get("h", {}).get("title"),
                                            "visita": p.get("a", {}).get("title"), "remates": len(lista),
                                            "penales": n_pen}
    diag["procesados_ahora"] = len(nuevos) - fallos
    diag["fallos"] = fallos
    diag["situaciones_vistas"] = situaciones
    diag["penales_total"] = len(previo["penales"])
    os.makedirs(SALIDA, exist_ok=True)
    json.dump(previo, open(ruta, "w"), ensure_ascii=False, indent=0)
    diag_global.append(diag)


if __name__ == "__main__":
    anios = [int(a) for a in sys.argv[1:]] or [2026]
    s = requests.Session()
    diag = [{"sonda": sonda()}]
    for a in anios:
        try:
            actualizar(a, s, diag)
        except Exception as e:
            diag.append({"temporada": a, "error": repr(e)[:500]})
    os.makedirs(SALIDA, exist_ok=True)
    json.dump({"diagnostico": diag, "hora": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(os.path.join(SALIDA, "_diagnostico.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(diag, ensure_ascii=False)[:3000])
