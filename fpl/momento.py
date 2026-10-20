"""¿Hacer la transferencia hoy o esperar al deadline?

El precio no decide QUÉ cambio hacer (son £0.1, unas centésimas de punto), pero sí CUÁNDO:
  - si quien entra va a subir esta noche, o quien sale va a bajar, conviene hacerlo hoy;
  - si hay dudas de lesión, conviene esperar a las noticias de antes del deadline.
"""
import pandas as pd

SUBE, CERCA = 100.0, 80.0


def _venta(compra, precio):
    return compra + (precio - compra) // 2 if precio > compra else precio


def evaluar(mov, jugadores, estado, deadline, ahora):
    if not mov:
        return None
    j = jugadores.set_index("id")
    sale = mov["sale"] if isinstance(mov["sale"], list) else [mov["sale"]]
    entra = mov["entra"] if isinstance(mov["entra"], list) else [mov["entra"]]
    horas = (deadline - ahora).total_seconds() / 3600
    motivos, esperar, hoy = [], [], []

    for e in entra:
        r = j.loc[e]
        if r["status"] in ("d", "i", "s"):
            esperar.append("%s tiene una duda (%s)" % (r["web_name"], r["news"] if isinstance(r["news"], str) else r["status"]))
        pp = r["proy_precio"]
        if not pd.isna(pp) and pp >= SUBE:
            hoy.append("%s sube esta noche (£0.1 más caro mañana)" % r["web_name"])
        elif not pd.isna(pp) and pp >= CERCA:
            motivos.append("%s está cerca de subir (%d%%)" % (r["web_name"], pp))
    for s_ in sale:
        r = j.loc[s_]
        if r["status"] == "d":
            esperar.append("%s está en duda: si se recupera, quizá no convenga venderlo" % r["web_name"])
        pp = r["proy_precio"]
        if not pd.isna(pp) and pp <= -SUBE:
            compra = estado["compra"].get(s_, r["now_cost"])
            perdida = _venta(compra, r["now_cost"]) - _venta(compra, r["now_cost"] - 1)
            if perdida > 0:
                hoy.append("%s baja esta noche y perderías £0.1 de su precio de venta" % r["web_name"])
            else:
                motivos.append("%s baja esta noche, pero por la regla de las mitades no pierdes nada" % r["web_name"])
        elif not pd.isna(pp) and pp <= -CERCA:
            motivos.append("%s está cerca de bajar (%d%%)" % (r["web_name"], pp))

    if horas <= 24:
        estado_m, texto = "deadline", "El deadline es en menos de 24 horas: decide con las noticias de hoy."
    elif esperar:
        estado_m, texto = "esperar", "Espera a las noticias previas al deadline: " + "; ".join(esperar) + "."
        if hoy:
            texto += " Aunque " + "; ".join(hoy) + ", la duda pesa más que £0.1."
    elif hoy:
        estado_m, texto = "hoy", "Hazlo hoy: " + "; ".join(hoy) + "."
    else:
        estado_m, texto = "puede_esperar", "Sin presión de precio: puedes esperar a las noticias previas al deadline."
    return {"estado": estado_m, "texto": texto, "otros": motivos, "horas_deadline": round(horas, 1)}
