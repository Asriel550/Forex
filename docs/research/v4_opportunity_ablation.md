# Ablación corta del contrato v4: reloj de oportunidades y reloj de barras

Dos corridas locales con el MISMO input causal, semilla PPO42, ent_coef=0.02,
16.384 pasos y evaluaciones cada 8.192 pasos. Ambas usan M15/binary/expiry1,
neutral_opportunity_penalty=0.05 y el selector v4 alineado con la recompensa.
Se comparan entrenamiento por oportunidades y entrenamiento por barras. El
reloj de descuento también cambia explícitamente; no se presupone equivalencia
en tiempo transcurrido o cantidad de filas de mercado.

**No se evaluó el holdout, no se eligieron parámetros por él y no se promovieron
pesos.** La selección es por score de validación, no por máximo PnL o máximo PF.
El JSON adjunto registra hashes, versiones, costes, auditoría e historial.

| Validación completa | Por oportunidades | Por barras |
| --- | ---: | ---: |
| Setups MC elegibles | 48 | 48 |
| Contratos del checkpoint seleccionado | 27 | 7 |
| Captura de setups | 56,25% | 14,58% |
| PnL simulado | +US$72 | +US$20 |
| Profit Factor | 1,90 | 2,00 |
| Acierto sin reembolsos | 70,37% | 71,43% |
| Max drawdown | 1,883% | 0,984% |
| Score de selección | 0,11203 | −0,00418 |
| Pasos del checkpoint seleccionado | 16.384 | 16.384 |

Ambos checkpoints seleccionados tienen entrenamiento efectivo y superaron el
score de su política inicial (−0,05). Con el reloj de oportunidades se ejecutaron
5.712 contratos durante train frente a 5 en el control por barras. **Esos conteos
repiten los pocos setups de train y no representan miles de contratos nuevos
independientes.** La validación usa todas las velas sin el wrapper, las mismas
probabilidades causales y los filtros originales de confianza/edge.

El resultado muestra que esta combinación evita una selección sin operaciones
en esta prueba corta y aumenta captura; no demuestra una solución definitiva,
una ventaja calibrada ni rentabilidad futura. Una sola semilla, 27 operaciones
y un conjunto de validación ya inspeccionado no bastan para esa conclusión.
Los resultados no se equiparan con los barridos PR #9 de 100.352 pasos en otro
entorno/versiones; las versiones e input_sha256 se conservan en el JSON.

La limitación estructural sigue siendo que el filtro habilita 48 de 26.395
transiciones. Incluso capturar todos los setups mantiene un porcentaje elevado
de NEUTRO ejecutado; eso es distinto de una red que ignora setups elegibles.
Confirmar con múltiples semillas y un protocolo predefinido en train/validación,
además de nuevos datos futuros independientes. No reabrir el holdout histórico.
