# Mesa de fichajes FPL

Proyección de puntos esperados y motor de transferencias para Fantasy Premier League.
Corre solo en GitHub Actions dos veces al día y alimenta un dashboard en Claude.

## Cómo funciona

1. **GitHub Actions** (`.github/workflows/actualizar.yml`) se ejecuta a las 05:17 y 17:17 (hora de Lima),
   descarga la temporada completa desde la API oficial de FPL y corre `python -m fpl.main`.
2. El modelo calcula los puntos esperados de cada jugador en las próximas jornadas y evalúa
   todas las transferencias simples y dobles posibles para tu plantilla.
3. Los resultados quedan en `data/db/*.json` (lo que lee el dashboard) y `data/resultados.json`.
4. Una tarea programada en Claude copia esos archivos al dashboard cada mañana.

No se guarda historial de partidos: cada corrida recalcula todo desde la API. Lo único que se acumula es
`data/predicciones.csv`, la proyección congelada antes de cada deadline, que sirve para medir si el modelo acierta.

## El modelo

| Pieza | Qué estima |
|---|---|
| Minutos | Probabilidad de ser titular, de jugar 60+ y minutos esperados, con más peso en las últimas jornadas. Lesiones y sanciones vía `status`, `chance_of_playing` y la fecha de regreso del campo `news`. |
| Ataque | xG y xA por 90 con pesos que decaen (vida media configurable) y un ancla hacia el promedio de la posición. Usa procesos (xG), no goles, para que un haul aislado no infle la proyección. |
| Defensa | Portería a cero `exp(−goles esperados en contra)`, goles en contra, probabilidad de cobrar contribución defensiva y atajadas (portero). |
| Rivales | Fuerza de ataque y defensa de cada equipo según su xG a favor y en contra, más la ventaja de local. Se aplica partido a partido, así que las jornadas dobles y en blanco salen solas. |
| Transferencias | Ganancia = (mejor XI con capitán de la plantilla nueva − la actual), sumada en el horizonte con descuento, menos golpes. Usa tu precio de venta real (mitad de cada subida). Recomienda guardar la transferencia si nada supera el umbral. |

Los valores por defecto se calibraron con un backtest sobre 2025-26: en cinco ventanas de 5 jornadas, el
modelo ordena a los titulares mejor que los puntos por partido (correlación de rangos 0.28 frente a 0.23).

## Ajustes

Todo está en `config.toml`: tu `team_id`, horizonte, vidas medias, umbrales de decisión y reglas de puntuación.
Si la estimación de transferencias libres no coincide con la web de FPL, fija `transferencias_libres`.
Al hacer commit de un cambio en `config.toml` el workflow se ejecuta de inmediato.

Para correrlo a mano: pestaña **Actions → Actualizar FPL → Run workflow**.

## Local

```bash
pip install -r requirements.txt
python -m fpl.main
```
