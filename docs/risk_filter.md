# Filtro causal de inacción y recompensa de riesgo (v3)

Actualización de umbral: [rango operativo 58–65%](drift_threshold_sensitivity.md), con default de entrenamiento 60%. Los ejemplos de 70% de esta referencia corresponden a PR #5.

Actualización de drift: [momentum causal](dynamic_drift.md). Los resultados y el ejemplo de drift cero de esta página corresponden a la referencia de PR #5; el filtro y sus límites se mantienen.

El resultado anterior (PF 0,88) no demuestra por sí solo sobreoperativa: ya hubo
87,7% de decisiones NEUTRO. Una frecuencia menor tampoco demuestra una ventaja.
Este cambio impone una política explícita de abstención y permite auditarla.
Los valores por defecto se fijan antes del nuevo backtest; no se optimizan
contra los resultados del último 10%.

## Monte Carlo: horizonte correcto y significado

Se conservan las 10.000 rutas GBM y los percentiles terminales P05/P95 a 32 pasos.
Se añaden mc_prob_up, mc_prob_down y mc_prob_tie: fracciones empíricas de esas
mismas rutas en su PRIMER paso, que es la expiración binaria real. La comparación
es estricta (> o < el precio inicial); un escenario de sigma=0 y drift=0 es empate.
La implementación usa innovaciones ordenadas/searchsorted, evitando una matriz
de n_velas x 10.000. GARCH y el drift configurado usan solo información disponible
al cierre de la fila. No se usa close[t+1] para decidir ni para generar features.

No se usa la softmax del actor como probabilidad de acierto. Tampoco se interpretan
P05/P95 de ocho horas como probabilidad direccional a quince minutos. El porcentaje
MC es una probabilidad **bajo el GBM supuesto**, no una certeza calibrada. El intervalo
Wilson trata solo el ruido de muestreo de rutas; no corrige error del modelo.

Con drift=0 y volatilidad habitual, P_up y P_down quedan cerca de 50%. Por tanto
el filtro de 70% normalmente produce NEUTRO en todo este snapshot. Ese resultado
es abstención, no rentabilidad demostrada; PF, Win Rate y Sharpe sin operaciones
son null/estado explícito. No se introduce drift artificial para fabricar confianza.

## Decisión compartida

src/decision.py contiene RiskConfig y resolve_action, utilizados en entorno,
backtest e inferencia Telegram. PUT/CALL solo se ejecutan cuando:

1. La probabilidad empírica en la dirección solicitada es estrictamente mayor
   que min_confidence (0,70 por defecto).
2. La ventaja conservadora payout*p_inferior - (1-p_inferior) supera
   min_expected_edge (0,02 por defecto). p_inferior es el límite Wilson con z=1,96
   y el número real de rutas. Esta cota trata empates como pérdidas, conservadoramente.

Un intento que no pasa se convierte en NEUTRO. requested_action conserva la
intención; action es la ejecución. El PnL y el contador de contratos usan action.
El agente recibe las tres probabilidades MC en la observación; trade_activity
(EWMA) se añade al estado de cuenta. No puede observar el siguiente cierre.

## Recompensa separada del dinero

La liquidación monetaria sigue siendo +payout*stake al acertar, -stake al fallar,
y empate conforme a tie_policy. Las penalizaciones de aprendizaje no se presentan
como comisiones reales ni se restan del saldo.

Para un contrato ejecutado:

reward = (PnL/stake)/max(std_payoff_MC, 0.10) - 0,05
         - 2*max(actividad_EWMA-0.10, 0)
         - 10*incremento_positivo_del_drawdown

La desviación del payoff se calcula con probabilidades disponibles antes de la
acción, payout y regla de empate. Se usa una rentabilidad ajustada al riesgo,
no un Sharpe retrospectivo inestable como reward. La EWMA usa decaimiento 0,95.
Esperar voluntariamente tiene reward=0. Intentar operar sin pasar el filtro tiene
reward=-0,25 y PnL=0. El drawdown permitido es el mínimo de account.max_drawdown y
risk.drawdown_limit (0,20); al alcanzarlo termina el episodio y se añade -5 al reward.
Saldo, pico y actividad se transportan entre segmentos en evaluación/backtest;
no hay contratos que crucen huecos. La cuenta de inferencia sigue siendo simulada,
inicial y flat; no se afirma conocer operaciones/saldo reales del destinatario.

La selección de checkpoint usa solo validación:
score = retorno_neto - drawdown_penalty*maxDD
        - trade_penalty*fracción_de_contratos
        - blocked_penalty*fracción_de_intentos_bloqueados.

## Reportes

reports/metrics.json, signals.csv, trades.csv y summary.md contienen:

- Conteo y porcentaje de PUT, NEUTRO y CALL realmente ejecutados.
- Conteo de decisiones propuestas e intentos bloqueados; JSON separa ambos.
- Media de probabilidad MC por tipo. PUT usa P_down; CALL usa P_up. Para NEUTRO
  se informa max(P_up,P_down): no existe una probabilidad de ganar NEUTRO.
- Profit Factor, PnL y cantidad de contratos ejecutados con confianza estrictamente
  superior a 60%, 70% y 80%. Son subconjuntos descriptivos posteriores de la
  política registrada; no son una reejecución contrafactual de todo el agente ni
  una búsqueda del mejor umbral con holdout. Sin operaciones, PF=null/no_trades.

La consola imprime el mismo desglose JSON. Actions incluye ahora summary.md
entre los artefactos de reportes.

## Migración y ejecución

Este contrato requiere bundle v3 y reward_version=risk_adjusted_v1. La observación
cambia de tamaño y el normalizador se vuelve a ajustar únicamente en train.
No se reutilizan pesos v2 cambiando el filtro por fuera: backtest/Telegram los
rechazan con un mensaje de regeneración. El manifiesto congela RiskConfig,
features, configuración cuantitativa, hashes y límites cronológicos.

```bash
python -m src.data_loader
python -m src.quant_engine --drop-warmup --seed 42
python -m src.train --total-timesteps 100000 --min-confidence 0.70 --min-expected-edge 0.02 --risk-drawdown-limit 0.20
python -m src.backtest
```

El workflow mode=train ya ejecuta estos módulos con los nuevos defaults. Un
bundle nuevo solo se publica después de entrenamiento y backtest exitosos.
La falta de cotizaciones actuales sigue impidiendo alertas en tiempo real.

## Holdout y límites de la conclusión

El split cronológico 70/20/10 y la purga siguen separados. Ni estadísticas de
normalización, ni pesos PPO, ni selección de checkpoint usan el 10% final.
Las nuevas probabilidades son causales; sus pruebas cambian/eliminan precios
futuros y comparan los prefijos. La política/umbrales quedan registrados antes
de calcular los resultados de backtest.

Sin embargo, ese holdout histórico YA se examinó y motivó esta tarea. Su próxima
comparación es investigación retrospectiva, no una nueva prueba independiente.
Para confirmar una mejora se necesita un periodo futuro sin usar, o un protocolo
walk-forward anidado fijado previamente. No se afirma que el filtro repare la
capacidad predictiva de un GBM con drift cero ni que NEUTRO al 100% sea una ventaja.

RiskConfig(enabled=False) queda disponible explícitamente para pruebas de
liquidación base/comparaciones técnicas; no es el default del entrenamiento.

## Comprobación técnica de esta entrega

156 pruebas locales superadas, incluidas regresiones v2 explícitas de caja,
probabilidades empíricas frente a las 10.000 rutas, prefijos causales, filtrado
compartido de Telegram, límites de drawdown y exportación estricta JSON/Markdown.
Se recalcularon 135.194 filas del snapshot real y se ejecutó un smoke test PPO
de 2.048 pasos (no un entrenamiento nuevo completo de 100.000). En su backtest
retrospectivo hubo 13.104 decisiones, 1.156 intentos bloqueados, 100% NEUTRO,
cero contratos y PnL cero. PF=null/no_trades. Es validación de integración y
abstención; no evidencia de mejora predictiva ni una nueva prueba fuera de muestra.
