# Comparación exploratoria PPO: semilla 42

Ambas corridas usan las mismas features estáticas de PR #7, umbral 58%, escala
1.5×, 100.352 pasos efectivos y validación determinista. No se ejecutó el holdout.
La selección de checkpoint usa el score de cada configuración; esos scores no
son directamente comparables entre recompensas distintas. No se publicó ningún
peso ni se sustituyó el bundle activo.

| Medida | Control (incentivos anteriores) | Nuevos defaults |
| --- | ---: | ---: |
| ent_coef | 0.01 | 0.02 |
| blocked / trade / neutral opportunity | 0.25 / 0.05 / 0 | 0.05 / 0.025 / 0.005 |
| Propuestas estocásticas PUT + CALL (train) | 7.474 | 10.170 |
| Contratos ejecutados (train) | 25 | 28 |
| Propuestas bloqueadas (train) | 7.449 | 10.142 |
| Entropía final en muestra del rollout, nats | 0.004234 | 0.014125 |
| Mejor checkpoint por validación (pasos) | 25.000 | 100.352 |
| Contratos del mejor checkpoint (validación) | 1 | 0 |
| PnL del mejor checkpoint (validación) | US$8 | US$0 |
| NEUTRO del mejor checkpoint (validación) | 26.394 / 26.395 | 26.395 / 26.395 |

Se incrementó la exploración en entrenamiento, pero el colapso determinista
persiste en esta semilla con los nuevos defaults. La entropía medida en el último
rollout no es la del checkpoint seleccionado ni se mide sobre el mismo conjunto
de estados en ambas corridas. Los resets de entrenamiento repiten barras; sus
contratos no son una muestra independiente. Se modifican varios incentivos a la
vez: esta comparación no aísla el efecto causal de ent_coef.

No hay evidencia de mejora económica ni de muestra representativa; no se
recomienda promover estos pesos. Para estudiar 0.03 u otras penalizaciones,
predefinir el experimento y comparar múltiples semillas en train/validación,
sin elegir por el histórico holdout ya examinado. El JSON adjunto contiene los
hashes del input, versiones, parámetros, checkpoints y auditorías exactos.
