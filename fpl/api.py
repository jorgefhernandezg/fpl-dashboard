"""Acceso a la API pública de FPL (o a una carpeta local con respuestas guardadas, para pruebas)."""
import json
import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://fantasy.premierleague.com/api"


class Fuente:
    def __init__(self, local=None, pausa=0.25):
        self.local = local
        self.pausa = pausa
        self.llamadas = 0
        if not local:
            s = requests.Session()
            reintentos = Retry(total=4, backoff_factor=1.5,
                               status_forcelist=(429, 500, 502, 503, 504),
                               allowed_methods=("GET",))
            s.mount("https://", HTTPAdapter(max_retries=reintentos))
            s.headers.update({"User-Agent": "Mozilla/5.0 (fpl-dashboard personal)"})
            self.sesion = s

    def get(self, ruta):
        ruta = ruta.strip("/") + "/"
        self.llamadas += 1
        if self.local:
            p = os.path.join(self.local, ruta.strip("/").replace("/", "__") + ".json")
            with open(p) as f:
                return json.load(f)
        r = self.sesion.get(BASE + "/" + ruta, timeout=30)
        r.raise_for_status()
        time.sleep(self.pausa)
        return r.json()
