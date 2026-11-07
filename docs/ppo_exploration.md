> Actualización PR #10: [contrato de mercado v4](market_contract_v4.md). `train` ahora publica un bundle de investigación aislado; `diagnostics` conserva el protocolo PR8.

# Exploración PPO e incentivos causales (bundle v3, reward v2)

El coeficiente anterior ya era 0.01. Se hace configurable y el default pasa a
**0.02**, con opciones 0.01/0.02/0.03 en workflow_dispatch. Stable-Baselines3 lo
usa como peso de la pérdida de entropía; favorece una distribución menos
concentrada al entrenar. No obliga a ejecutar PUT/CALL ni altera predict con
deterministic=True, que conserva la acción de mayor probabilidad.

Referencia de la implementación de SB3 2.7.1:
https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/ppo/ppo.html

## Incentivos ajustados

| Parámetro | Antes | Default nuevo |
| --- | ---: | ---: |
| ent_coef | 0.01 fijo | 0.02 configurable |
| trade_penalty | 0.05 | 0.025 |
| blocked_penalty | 0.25 | 0.05 |
| neutral_opportunity_penalty | 0 | 0.005, desactivable |

La nueva penalización de NEUTRO se aplica SOLO si una dirección supera el
umbral, Wilson y edge conservador por payout. Se calcula al cierre actual,
antes de observar close[t+1], y es igual tanto si esa oportunidad luego gana
como si pierde. No es una etiqueta de ganador ni prueba de una ventaja real.
Sin setup elegible, esperar sigue teniendo recompensa cero. El coste no es
reiterativo ni depende de una racha oculta; no cambia la forma de la observación.

Un intento bloqueado paga solo blocked_penalty; no acumula también el coste de
omisión. La liquidación monetaria sigue siendo payout*stake o -stake; ninguna
penalización de entrenamiento se resta del saldo ni se presenta como comisión.

No cambian el stop de drawdown del 20%, las penalizaciones por incremento del
drawdown/stop, la ventaja mínima por payout ni el filtro de confianza de PR #7.
No se fuerza una muestra de contratos castigando indiscriminadamente la espera.
El score de selección de validación incluye también el coste configurado por
setups elegibles omitidos, además de sus componentes anteriores. El holdout no
interviene en el aprendizaje ni en la selección.

## Parámetros desde Android / Actions

En Run workflow elegir mode=train, por ejemplo:

- confidence_threshold: 0.58
- drift_scale: 1.5
- ent_coef: 0.02 (o 0.03 para otro experimento predefinido)
- blocked_penalty: 0.05
- neutral_opportunity_penalty: 0.005 (o la opción string 0 para desactivarla)

Para push/cron se admiten FOREX_ENT_COEF, FOREX_BLOCKED_PENALTY,
FOREX_NEUTRAL_OPPORTUNITY_PENALTY y FOREX_TRADE_PENALTY como variables del repo.
Los inputs seleccionados tienen precedencia. Los coeficientes quedan registrados
en training_manifest.json y en el PPO serializado; alerts mantiene los de su
bundle y no toma estas opciones como overrides de un modelo ya entrenado.

```bash
python -m src.quant_engine --drop-warmup --drift-scale 1.5
python -m src.train --min-confidence .58 --ent-coef .02 --blocked-penalty .05 --trade-penalty .025 --neutral-opportunity-penalty .005 --total-timesteps 100000
```

El coeficiente de entropía es fijo durante cada corrida (sin annealing). La API
acepta valores finitos entre 0 y 0.1; la UI limita la comparación a 0.01–0.03.
Los umbrales 58/60/62/65% y las escalas 1/1.5/2 siguen disponibles.

## Reportes de exploración y evaluación

El callback guarda models/exploration_history.json por rollout y
models/exploration_summary.json al finalizar. Actions ya publica models/*.json
como artefactos de reportes. La auditoría contiene:

- Propuestas PUT/NEUTRO/CALL estocásticas durante entrenamiento.
- Acciones realmente ejecutadas, bloqueos y setups omitidos.
- Entropía media en nats y probabilidades medias de la política sobre hasta
  256 observaciones del rollout. La máxima entropía de tres acciones es ln(3).

La inspección usa no_grad y ninguna muestra aleatoria adicional: no cambia el
RNG del entrenamiento. Las filas por rollout describen la política antes de
su actualización PPO; el resumen final inspecciona el último modelo actualizado,
que puede diferir del checkpoint seleccionado por validación.

validation_history.json y best_validation.json también guardan un snapshot de
probabilidades/entropía en una muestra predefinida de observaciones de validación.
No se usa esa entropía para seleccionar al ganador. Las acciones de validación
y backtest siguen siendo deterministas; no se los vuelve estocásticos solo para
inflar la cantidad de operaciones. Los CSV/JSON/Markdown de backtest incluyen
missed_opportunity y el conteo agregado de setups elegibles omitidos.

## Compatibilidad

La forma sigue siendo de 32 observaciones y manifest_version=3. Los entrenamientos
nuevos escriben reward_version=risk_adjusted_v2. Backtest/Telegram admiten v1 y v2;
RiskConfig.from_metadata asigna coste de omisión cero cuando no estaba archivado.
Los valores anteriores de trade_penalty y blocked_penalty se conservan en sus
manifiestos, y PPO.load mantiene su ent_coef original. No se reinterpreta la
recompensa/estado de pesos antiguos con los defaults nuevos. Para usar estos
incentivos se requiere un entrenamiento nuevo; no se modifican los pesos guardados.

## Límites de la evaluación

Más acciones de entrenamiento no son contratos independientes fuera de muestra:
los resets pueden repetir barras. Más entropía no garantiza que el argmax deje
NEUTRO ni crea una ventaja predictiva. Una muestra representativa no se obtiene
forzando operaciones en el holdout ni eligiendo el coeficiente que más opere allí.
La comparación se limita a entrenamiento y validación. El holdout histórico ya
se examinó: la confirmación requiere nuevos datos o un protocolo walk-forward
predefinido, tamaños muestrales suficientes y comparación de payout/PF/drawdown.
No se cambia el requisito de cotizaciones recientes para alertas Telegram.

## Comparación reproducida

[Resultados de control y nuevos defaults](research/ppo_exploration_comparison.md), con JSON de auditoría: más propuestas estocásticas, pero persistencia de NEUTRO en validación. No se promueve ningún peso de estas corridas.
