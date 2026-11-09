# PR41 — GARCH independiente por divisa y 20 generaciones evolutivas para cada mercado

**Estado:** protocolo de neuroevolución independiente. Q1 y Q2 de H1 2026 como IS, Q3 julio–septiembre totalmente cerrado. No se realizan afirmaciones de rentabilidad sin ejecutar el CI.

## 1. Por qué cambia la arquitectura de PR40

La [PR40](https://github.com/Asriel550/Forex/pull/40) confirmó que aplicar idénticos pesos entrenados en EURUSD a GBPUSD, AUDUSD y USDCAD no conserva alfa: las tres semillas terminaron con PF combinado <1 y stop de cuenta superior al 20%. Los universos GARCH de cada par **ya se calcularon de manera independiente en PR40**; el error residual era **transferir los genes EUR** en la inferencia.

En PR41 se reemplazan los **pesos de dirección, entrada y cotas GARCH** por 22 genes entrenables **independientes** para cada activo. No se continúan los genomas PR38 ni se reproduce un fundador de EURUSD en otras divisas. Todos comienzan desde un genoma neutro propio y se reproducen en **semillas 42,43,44**, con streams RNG propios por símbolo.

## 2. Fuentes H1 inmutables (sin señales M1)

| Símbolo | Release H1 M5/M15 + GARCH nativo | SHA256 `universe.npz` |
| --- | --- | --- |
| EURUSD | `data-m5-m15-is-2026h1-full-universe-v2` | `05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9` |
| GBPUSD | `data-pr40-gbpusd-m5-m15-garch-universe-2026h1-v1` | `2cc9ab95266cb3a87c2988ffd8006c79e9db68765d5ab54ded129c161fff4a0b` |
| AUDUSD | `data-pr40-audusd-m5-m15-garch-universe-2026h1-v1` | `8d1f34d5bf38ce1c4beeb5079d374a7af1a9d07a8161d07fc010d56b251b0eee` |
| USDCAD | `data-pr40-usdcad-m5-m15-garch-universe-2026h1-v1` | `bf866689bb549bf2d647d8600bd343148dfb44040fb51c9cbb2baa7738f4f8b9` |

Las OHLCV de GBPUSD/AUDUSD/USDCAD provienen de HistData (M1 usado **únicamente como materia prima de agregación a OHLCV M5/M15 completas**, no como feature). GARCH/Monte Carlo M5 fue recalculado causalmente en cada activo con el mismo procedimiento, sin copiar la serie de EURUSD. Se excluyeron los registros UTC que caían en julio de 2026 **antes de la generación de los indicadores**. El esquema operativo utiliza solo las 8 features de precio de velas M5 y **última M15 cerrada**. Los históricos Q3 no son insumo del pipeline.

## 3. Espacio genético por activo: 22 genes independientes

La cabeza local se redefine deliberadamente para **no heredar** la red direccional de EURUSD. No es un fine-tune de sus pesos: es una cabeza `tanh` de ocho variables de temporalidad. Este cambio de arquitectura permite una prueba rápida de 20 generaciones; **no equivale a volver a entrenar toda la red multicapa antigua**.

- **Genes 0–7:** ponderación de entrada sobre ocho características M5/M15 locales.
- **Gen 8:** sesgo de entrada local; la admisión exige entrada positiva.
- **Genes 9–16:** pesos de dirección local sobre ocho características M5/M15.
- **Gen 17:** sesgo de dirección.
- **Gen 18:** ganancia del score direccional `(4.5 + 2 tanh(g18))`.
- **Genes 19–20:** cotas GARCH independientes `low=.35+sigmoid(g19)`, `high=1.2+2 sigmoid(g20)`, definidas sobre `sigma_M5_actual / mediana_desplazada_48`.
- **Gen 21:** umbral propio de margen direccional `0.3+2 sigmoid(g21)`.

**Nada** de los genes de cabeza de entrada/dirección EURUSD PR38 alimenta al modelo local. Se reutilizan **solo** las reglas económicas y operativas comunes del contrato binario M5 (+80% acierto, −100% error), el tamaño de apuesta, el stop y cupo de 2/h. No se transforma cada activo en una cuenta ficticia con mayor capital. El **Gate p85 es causal y separado por símbolo**, con 500 observaciones de calentamiento y liquidaciones maduradas.

Cada genoma se evalúa con `scripts.run_m5_dynamic_upstream.one_candidate` **solo como motor de replay**, forzando un selector neutro que reproduce exactamente la máscara ya calculada por la cabeza local: **no ejecuta ni optimiza genes EUR de entry**.

## 4. Presupuesto experimental y selección

Por cada símbolo, **20 generaciones × 128 candidatos × 3 semillas = 7.680 evaluaciones**; total cuatro símbolos **30.720**. Las generaciones se numeran G1–G20 **locales**, no se continúan las Gen376–497. Cada símbolo tiene su propia función `evaluate_local`, datos GARCH y streams aleatorios (`seed+1000×indice_del_par`).

El fitness exploratorio mezcla:
- aptitud del replay temporal de los trimestres,
- peor PF entre Q1/Q2, media geométrica y densidad ejecutada,
- penalizaciones por DD superior a 15% y por stops operativos.

El criterio **duro** individual NO puede ser sustituido por score:
**PF >1,20 en Q1 y Q2 por separado, PnL positivo ambos trimestres, DD trimestral y semestral <15%, DD sin stop y sin parada de cuenta**. Cada símbolo debe cumplir **en las 3 semillas originales** para declararse `eligible_assets_3seeds`. También se informa frecuencia real/h; un PF alto con pocas entradas no demuestra capacidad operativa.

El motor de consolidación agrupa obligatoriamente las doce combinaciones símbolo/semilla y **no permite elegir la mejor semilla diferente por activo** para fabricar una cartera apta.

## 5. Cartera filtrada: condicional a la calidad individual

**Solo si al menos dos pares** satisfacen el criterio completo en todas las semillas, se permite por workflow otro replay independiente desde los campeones locales congelados. Se calcula con:

- la **misma cesta aprobada previamente** para seeds42/43/44,
- una **cuenta continua compartida**, 2 operaciones por hora **globales**, stop de DD y liquidaciones simultáneas,
- PF Q1/Q2 de replays trimestrales separados y DD/frecuencia del semestre con una única cuenta, sin reset tras Q1.

El criterio operativo de investigación de cartera sigue siendo PF>1,20 en Q1 y Q2, **≥1 contrato/h ejecutado** en H1 y DD H1<15% sin stops en las tres semillas. Si pasan menos de dos pares, los trabajos multiasset quedan **SKIPPED**: no se finge que hay diversificación ni se usan pares con PF malo.

**Limitación científica importante:** el filtro individual y la cartera usan el **mismo IS** Q1/Q2, de modo que aun si se alcanzan estos resultados, **no son evidencia OOS independiente** ni autorizan trading en vivo. Q3 debe permanecer intacto hasta autorización específica posterior.

## 6. Evidencia y archivos

- `scripts/run_pr41_native_asset_evolution.py`: entrenamiento propio, función de fitness, métricas y genomas SHA256, 20 generaciones.
- `tests/test_pr41_native_asset_evolution.py`: ausencia de genes EUR, tamaño exacto de búsqueda, causalidad GARCH y estricta aceptación.
- `scripts/run_pr41_filtered_portfolio.py`: solamente activos que pasaron PF/DD en las 3 semillas, patrimonio/stop/cupo compartidos.
- `.github/workflows/pr41-native-20gen-assets.yml`: 12 jobs de neuroevolución, JSON individual y consolidación, posterior replay condicional de cartera sin Q3.
- `reports/pr41-independent-assets/{symbol}-seed-{seed}.json`: score de cada generación, PF Q1/Q2, DD, frecuencia ejecutada, cotas propias, genoma firmado.
- `reports/pr41-20gen-summary.json`: total 30.720 evaluaciones verificadas, 12 filas y activos elegibles.
- `reports/pr41-filtered-portfolio-summary.json`: **solo si** hay dos o más activos elegibles.

## 7. Cierre documental

La ausencia de operaciones, PF nulo o un stop temprano son resultados negativos a declarar con precisión. Ningún CI verde por sí solo certifica rentabilidad. Sin optimización en Q3, sin señales M1, sin promociones automáticas y sin fusionar investigación no aprobada.


## 8. Cierre experimental verificado: CI #42576852134

[GitHub Actions final #42576852134](https://github.com/Asriel550/Forex/actions/runs/42576852134) **SUCCESS**, incluidos los doce entrenamientos individuales, el consolidador cross-seed y los tres replays de cartera. [JSON 12 casos](https://github.com/Asriel550/Forex/actions/runs/42576852134/artifacts/13655966449) y [JSON de la cartera](https://github.com/Asriel550/Forex/actions/runs/42576852134/artifacts/13655782276).

**Presupuesto cumplido:** 20 generaciones × 128 genomas por generación × 4 activos × 3 semillas = **30.720 evaluaciones genéticas**. Todos los datasets son releases selladas H1 M5+M15/GARCH de cada símbolo. Cada activo inicia genes propios y semilla propia; se mantienen compartidos solamente el producto binario M5, Gate p85, las restricciones de cuenta y el esquema de sus ocho features M5/M15. No se copiaron pesos EURUSD de entrada ni dirección a GBP/AUD/CAD. Q3 no se consultó.

### 8.1. Individuos elegidos (PF Q1/Q2, frecuencia ejecutada y DD H1)

| Símbolo | Semilla | Gen local | PF Q1 | PF Q2 | Contratos/h H1 | DD H1 | Banda GARCH propia (relativa) | Pasa PF/DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| EURUSD | 42 | 19 | 1.933 | 1.576 | 0.0497 | 3.16% | 1.280/1.970 | Sí |
| EURUSD | 43 | 4 | 1.459 | 1.288 | 0.0864 | 7.72% | 0.853/2.125 | Sí |
| EURUSD | 44 | 17 | 3.467 | 1.508 | 0.0334 | 5.50% | 0.972/1.935 | Sí |
| GBPUSD | 42 | 20 | 1.647 | 1.348 | 0.0682 | 4.62% | 0.897/1.658 | Sí |
| GBPUSD | 43 | 7 | 1.600 | 1.527 | 0.0160 | 2.29% | 0.788/2.231 | Sí |
| GBPUSD | 44 | 18 | 1.600 | 1.446 | 0.0758 | 4.81% | 0.970/2.322 | Sí |
| AUDUSD | 42 | 8 | 4.000 | 1.300 | 0.0394 | 5.51% | 1.054/2.434 | Sí |
| AUDUSD | 43 | 19 | 1.867 | 1.462 | 0.0387 | 6.91% | 1.033/2.077 | Sí |
| AUDUSD | 44 | 18 | 1.543 | 1.490 | 0.0432 | 5.09% | 0.563/2.250 | Sí |
| USDCAD | 42 | 18 | 2.031 | 1.371 | 0.0556 | 5.26% | 1.008/2.832 | Sí |
| USDCAD | 43 | 17 | 4.200 | 1.482 | 0.0421 | 3.50% | 1.086/2.106 | Sí |
| USDCAD | 44 | 20 | 1.347 | 1.322 | 0.0600 | 5.81% | 1.111/2.256 | Sí |

**Filtro individual aplicado literalmente:** PF estrictamente >1.20 en ambos trimestres, DD trimestral y semestral <15%, PnL positivo y ninguna parada de cuenta. **Los cuatro activos pasaron en las tres semillas**. El filtro NO exigía frecuencia mínima individual; por eso muchas candidatas conservadoras alcanzaron PF elevados con actividad minúscula. No deben calificarse como sistemas comercialmente densos ni como alfa OOS probado.

Frecuencias individuales extremas: **mínimo 0.0160/h** (GBPUSD seed43, ~una operación cada 62 horas observadas), **máximo 0.0864/h** (EURUSD seed43, ~una operación cada 12 horas observadas). Los PF excepcionalmente altos, como **4.2** (USDCAD seed43 Q1) o **4.0** (AUDUSD seed42 Q1), requieren análisis posterior del número real de operaciones y de sus intervalos de confianza antes de interpretarse como ventaja robusta; se han optimizado/seleccionado en el mismo IS Q1/Q2.

### 8.2. Cartera compartida condicional de cuatro mercados

Al haber pasado **todas las divisas las tres semillas**, se activó el replay de las cuatro conjuntamente, sin optimización adicional de componentes ni descarte retrospectivo. Se mantuvieron cuenta única, límite **global 2 contratos/h**, un único stop y liquidaciones simultáneas M5.

| Semilla | PF cartera Q1 | PF cartera Q2 | Contratos/h realmente ejecutados H1 | DD H1 | Cumple objetivo completo de cartera |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 1.786 | 1.394 | 0.2008 | 4.53% | No |
| 43 | 1.683 | 1.406 | 0.1695 | 5.26% | No |
| 44 | 1.600 | 1.430 | 0.2042 | 5.79% | No |

**Resultado:** PF Q1 y Q2 de la cartera **>1.20 en las tres semillas**, DD semestral **<6%** en las tres. Pero la frecuencia ejecutada quedó **0.1695–0.2042/h**, muy lejos del objetivo previo **≥1.00/h**. `all_qualified=false` por **frecuencia insuficiente**. Los objetivos individuales de PF/DD se consiguieron dentro del IS, pero no se solucionó la densidad.

La frecuencia global no se calcula sumando 4 promesas independientes; resulta del replay completo con choques de horario, capital compartido y stops. Los PF trimestrales proceden de replays trimestrales individuales y el DD/frecuencia de la cuenta semestral continua. No confundir indicadores entrenados/seleccionados IS con resultados fuera de muestra.

### 8.3. Incidentes y reproducibilidad

- [Primer run #42576192854](https://github.com/Asriel550/Forex/actions/runs/42576192854): falló en la primera generación por PF `None` en la interfaz NumPy del motor genético; sin resultados económicos registrados.
- [Segundo run #42576424566](https://github.com/Asriel550/Forex/actions/runs/42576424566): las veinte generaciones se ejecutaron pero falló la exportación del historial, que es `dict` plano y no `record` anidado; sin métricas finales exportadas.
- [Tercer run #42576852134](https://github.com/Asriel550/Forex/actions/runs/42576852134): corregido el transporte de PF indefinido sin inventar ganancias, y validado el schema plano con prueba de regresión, **todos los jobs y comparativas finales SUCCESS**. 30.720 evaluaciones genéticas contabilizadas del tercer run; no sumar repeticiones fallidas como resultados independientes.
- Los archivos `reports/pr41-independent-assets/{symbol}-seed-{seed}.json` contienen cromosomas SHA256, las **20 generaciones completas** con fitness/mediana/PF/frecuencia/DD por gen, fuente SHA256, variación por semilla y test de selección; los resúmenes están disponibles en Actions (retención típica 90 días).

### 8.4. Interpretación técnica y decisión

La independencia GARCH + dirección/entrada/umbrales **sí mejoró la selección de operaciones en IS** frente al trasplante de pesos EURUSD PR40: el PF de los modelos independientes no se parece al PF subunitario del trasplante. **No se puede afirmar que la mejora se deba exclusivamente a la independencia**: cambió la arquitectura de la cabeza, la función objetivo, los pesos y el filtro de admisión; hacen falta ablaciones controladas.

El coste principal es **parálisis operativa**. El algoritmo descubre subconjuntos de señales selectos, rentables retrospectivamente, pero no alcanza una operación por hora en la cartera. Las tres semillas corren sobre el mismo semestre y la cartera se construye seleccionando por esos mismos datos, lo que introduce **sesgo de selección IS** y no equivale a verificación independiente. **No abrir Q3** para buscar ganadores; cualquier trabajo posterior debe fijar en IS un plan de frecuencia mínima/intervalos de confianza y examinar densidad por régimen antes de nueva búsqueda.

**Decisión final:** individual PF/DD aprobado en cuatro pares, cartera PF/DD también aprobado en H1 IS, **frecuencia 1/h no aprobada**, aprobación de trading **NO**, PR41 sigue *draft*, cero promoción y Q3 intacto.
