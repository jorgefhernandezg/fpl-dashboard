"""Ajustes manuales de config.toml: minutos esperados y listas de compra/venta.

Los jugadores se nombran como aparecen en FPL ("Cherki", "B.Fernandes"); sin distinguir mayúsculas
ni tildes. También se acepta el id numérico. Si un nombre coincide con varios jugadores, se avisa y
se ignora: usa el id o añade el equipo, por ejemplo "Gabriel ARS".
"""
import unicodedata


def _norm(s):
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", ". ").split())


def resolver(nombre, jugadores, avisos):
    j = jugadores
    if isinstance(nombre, int) or str(nombre).isdigit():
        pid = int(nombre)
        if pid in set(j["id"]):
            return pid
        avisos.append("No existe un jugador con id %s." % nombre)
        return None
    q = _norm(nombre)
    partes = q.split()
    equipo = None
    if len(partes) > 1 and partes[-1].upper() in set(j["abrev"].str.upper()):
        equipo, q = partes[-1].upper(), " ".join(partes[:-1])
    cand = j[j["web_name"].map(_norm) == q]
    if not len(cand):
        cand = j[j["web_name"].map(_norm).str.contains(q, regex=False)]
    if equipo:
        cand = cand[cand["abrev"].str.upper() == equipo]
    if len(cand) == 1:
        return int(cand["id"].iloc[0])
    if not len(cand):
        avisos.append('No encontré a "%s". Escríbelo como aparece en FPL o usa su id.' % nombre)
    else:
        ops = ", ".join("%s (%s, id %d)" % (r.web_name, r.abrev, r.id) for r in cand.head(5).itertuples())
        avisos.append('"%s" coincide con varios jugadores: %s. Usa el id o añade el equipo.' % (nombre, ops))
    return None


def cargar(cfg, jugadores):
    a = cfg.get("ajustes", {}) or {}
    avisos = []
    minutos = {}
    for nombre, v in (a.get("minutos", {}) or {}).items():
        pid = resolver(nombre, jugadores, avisos)
        if pid is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            avisos.append('Minutos de "%s" no es un número.' % nombre)
            continue
        if not 0 <= v <= 90:
            avisos.append('Minutos de "%s" debe estar entre 0 y 90.' % nombre)
            continue
        minutos[pid] = v

    def lista(clave):
        out = []
        for nombre in a.get(clave, []) or []:
            pid = resolver(nombre, jugadores, avisos)
            if pid is not None:
                out.append(pid)
        return out

    out = {"minutos": minutos, "no_vender": lista("no_vender"), "no_comprar": lista("no_comprar"),
           "vender": lista("vender"), "comprar": lista("comprar"), "avisos": avisos}
    nombre = jugadores.set_index("id")["web_name"]
    for a_, b_ in (("comprar", "no_comprar"), ("vender", "no_vender")):
        choque = set(out[a_]) & set(out[b_])
        for pid in choque:
            avisos.append("%s está en %s y en %s: se ignora en %s." % (nombre.get(pid, pid), a_, b_, a_))
        out[a_] = [x for x in out[a_] if x not in choque]
    return out


def aplicar_minutos(base, minutos):
    """Reemplaza el modelo de minutos por los minutos esperados por partido que indicaste.

    Se aplica cuando el jugador está disponible: si FPL lo marca lesionado o sancionado, manda FPL."""
    for pid, v in minutos.items():
        if pid not in base.index:
            continue
        if v >= 60:
            base.at[pid, "p_tit"] = 0.95
            base.at[pid, "min_tit"] = min(90.0, v / 0.95)
            base.at[pid, "p60_tit"] = 0.9 if v >= 70 else 0.75
            base.at[pid, "p_sup"] = 0.5
        else:
            base.at[pid, "min_tit"] = 80.0
            base.at[pid, "p60_tit"] = 0.8
            base.at[pid, "p_sup"] = 0.6
            min_sup = base.at[pid, "min_sup"] if base.at[pid, "min_sup"] == base.at[pid, "min_sup"] else 18.0
            # p·80 + (1−p)·0.6·min_sup = v
            p = (v - 0.6 * min_sup) / (80.0 - 0.6 * min_sup)
            base.at[pid, "p_tit"] = max(0.0, min(1.0, p))
    return base
