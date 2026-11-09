# Bitácora integral del ciclo EURUSD M5+M15 — reproducibilidad, saturación y búsqueda de alfa (PR #34–#38)

**Estado:** investigación IS cerrada; código y resultados auditables; **ningún campeón aprobado o promovido**.  
**Cierre del flujo PR #38:** 2026-11-09 02:38:40 UTC ([GitHub Actions 42551086458](https://github.com/Asriel550/Forex/actions/runs/42551086458)).  
**Ámbito:** EURUSD, decisión y vencimiento M5; contexto M15 completamente cerrado; enero–marzo 2026 (**Q1**) y abril–junio 2026 (**Q2**) como *in-sample*. Julio–septiembre 2026 (**Q3**) permanece en cuarentena OOS.

> **Hallazgo central.** Las estrategias de más frecuencia no consiguieron rentabilidad ni control de riesgo; las estrategias filtradas por **régimen GARCH** sí redujeron el drawdown a 6,40–8,27 % y acercaron ambos PF trimestrales a 1,20, pero operaron únicamente ~0,20–0,22 contratos/h. **La combinación exigida de PF, frecuencia, DD y consistencia entre las tres semillas no apareció en ningún experimento.**

## 1. Alcance, criterios y terminología

La fase histórica Gen376–497 exigía **1,2 contratos/h**, PF>1,20 en ambos trimestres, DD<15 % y sin paradas. En la investigación de hipótesis **PR #38 se ajustó explícitamente la meta de frecuencia a ≥1,0 contratos/h semestrales**, manteniendo PF estrictamente >1,20 por trimestre, DD semestral <15 %, PnL positivo, sin stop y **éxito simultáneo en semillas 42, 43 y 44**. No comparar las dos metas como si fueran iguales.

- **Frecuencia ejecutada real:** número de contratos efectivamente registrados tras gate, límite horario y paradas de cuenta, dividido por horas observadas en el semestre con una **cuenta continua**. Es distinta de propuestas upstream/h y de admisiones por gate antes del cupo.
- **PF Q1 / Q2:** cociente de ganancias brutas sobre pérdidas brutas en el replay operativo separado por trimestre. Un PF superior a 1 en un experimento IS no implica generalización.
- **DD (%):** máximo drawdown de la cuenta, en el semestre para las tablas PR #38; resultados con DD superior o igual a 15 % no califican.
- **Economía simulada:** contratos binarios M5 con pago de +80 % por acierto y −100 % por fallo, stake, cupo y stops heredados; **no** órdenes spot/CFD con TP/SL reales.
- **Entrenamiento:** neuroevolución con 128 genomas por generación; las nuevas funciones de pérdida se usan como términos del fitness, **sin backpropagation**.
- **Semillas:** 42, 43 y 44; la repetición entre ellas es una comprobación de estabilidad algorítmica **sobre el mismo IS**, no tres muestras OOS independientes.
- **M1:** puede existir como materia prima para reconstruir OHLCV M5/M15; **no existen señales, indicadores ni decisiones M1 en la versión de trading válida**. La antigua variante PR #36 con proxies M1 fue retirada y sus resultados no se toman como evidencia para M5+M15.

## 2. Datos congelados, firmas y cadena de custodia

| Recurso | Versión / SHA256 del archivo principal | Finalidad |
| --- | --- | --- |
| [Snapshot GARCH IS canónico](https://github.com/Asriel550/Forex/releases/tag/data-m5-is-2026h1-garch-v1) | `7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504` | Enero–junio, GARCH/MC, 2.859 propuestas de marzo de referencia |
| [Universo elegible v1, *retirado para señales*](https://github.com/Asriel550/Forex/releases/tag/data-m5-is-2026h1-full-universe-v1) | `d8de896967ec419b89e92be61b0f6cad45165ddff297b8353fbf936db53a0d8d` | Archivo histórico de reconstrucción, con proxies M1 que no deben llegar a la cabeza operativa |
| [Universo M5+M15 v2 **vigente**](https://github.com/Asriel550/Forex/releases/tag/data-m5-m15-is-2026h1-full-universe-v2) | `05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9` | **35.261 M5 elegibles**, 9.968 propuestas legacy reconciliadas, variables M5+M15 |
| [Etiquetas supervisadas solo IS PR #38](https://github.com/Asriel550/Forex/releases/tag/data-m5-m15-is-2026h1-alpha-labels-v1) | `5163af1212217e4a05ad2cdd35928720eafb6031c8fba79c12b934975a2a469d` | Targets ATR, R:R y marcadores de baja señal; solamente para entrenamiento |

Las Releases privadas no caducan automáticamente como los artefactos Actions, pero **pueden ser eliminadas por administradores**. El CI verifica hashes y se niega a consumir datos sustituidos; la copia externa con protección contra borrado sigue siendo recomendable.

La genealogía de la fuente GARCH corresponde al runner forense «A»; **no** se ajusta GARCH otra vez durante las evoluciones ordinarias. Las máscaras originales, timestamps, outcomes, volatilidad e inputs direccionales se comprueban contra el snapshot. Se validó Q3 como `oos_accessed: false` y los constructores restringen las lecturas a meses `202601`–`202606`; **no se incorporaron datos Q3** en esta fase.

## 3. Antecedentes: PR #34–#37 (qué motivó PR #38)

### PR #34 — Hash distinto de marzo y escala del fitness

Dos reconstrucciones independientes de idénticas fuentes dieron los mismos hashes M1/M5, pero el optimizador GARCH tuvo distinta convergencia en el **refit de la fila 16.928**. Un runner conservó parámetros previos, perdiendo exactamente la propuesta de **2026-03-16 11:55 UTC**: **2.858** frente a las **2.859** originales de marzo. No fue una carrera de escritura M1/M5 demostrada. Se fijó el snapshot canónico y una calibración nueva del fitness que mezcla retorno operativo/gate pre-stop y penalizaciones por DD y densidad. No se confundió mejora de score con mejora de PF. [Evidencia forense](https://github.com/Asriel550/Forex/actions/runs/42535814697) · [CI PR34](https://github.com/Asriel550/Forex/actions/runs/42536753310).

### PR #35 — Persistencia del dataset y cuello de botella upstream

Se publicó el GARCH en una Release duradera. El embudo legacy era `entry>0` y margen de dirección **>5,220727**. De **35.261** M5 elegibles en H1, **12.519** tenían entry positivo y solo **9.968** llegaron a propuestas. Q2 tenía **17.974** M5 elegibles, **5.360** tras `entry>0` y **4.374** tras margen; capacidad física previa al gate y posterior al cupo de 2/h **0,951/h** en Q2. El filtro `entry>0` era la restricción más severa, independientemente de la red direccional. [CI PR35](https://github.com/Asriel550/Forex/actions/runs/42538864576).

### PR #36 — Cabeza upstream evolutiva corregida a M5+M15

Se retiraron expresamente los proxies M1 introducidos por error; se desarrollaron **8 señales causales de OHLC M5/M15 cerradas** y 11 genes para entrada/margen adaptativo. El universo v2 conserva sin alteración los 9.968 registros legacy y 35.261 filas elegibles. En el primer smoke válido Gen376/43/p85: **12.741 propuestas** (frente a 9.968), **1.850 admisiones** p85 antes del cupo (frente a 1.447), capacidad previa al gate **1,230/h** (frente a 1,086/h); no obstante, ejecución semestral **0,036/h**, PF **0,713 / 0,873** (Q1/Q2), DD trimestral superior al 20 %. [CI smoke M5+M15](https://github.com/Asriel550/Forex/actions/runs/42542842719).

Después se ejecutaron **Gen376–397 inclusive, 22 generaciones × 128 × 3 semillas = 8.448 evaluaciones** bajo p85, misma cabeza M5+M15:
- Seed 42: mejor **Gen392**, PF Q1 **0,677**, Q2 **0,910**, ejecución **0,033/h**.
- Seed 43: mejor **Gen392**, PF Q1 **0,709**, Q2 **0,927**, ejecución **0,034/h**.
- Seed 44: mejor **Gen386**, PF Q1 **0,649**, Q2 **0,931**, ejecución **0,033/h**.

Ninguno calificó. [CI Gen376–397](https://github.com/Asriel550/Forex/actions/runs/42544173112).

### PR #37 — 100 generaciones adicionales y prueba de meseta

Se prolongó desde el mejor genoma de cada semilla, con RNG nuevo (no la población completa ni estado exacto del RNG Gen397): **Gen398–497 inclusive, 100 × 128 × 3 = 38.400 evaluaciones nuevas**.

- Seed 42: mejor **Gen462**; fitness **−0,586262 → −0,468859** (Δ **+0,117403**), 6 mejoras, 35 generaciones sin avance final; PF Q1/Q2 **0,675 / 0,943**; ejecutada **0,0323/h**; DD Q1/Q2 **20,57 / 20,04 %**.
- Seed 43: mejor **Gen424**; fitness **−0,610543 → −0,506909** (Δ **+0,103634**), 5 mejoras, 73 sin avance; PF **0,681 / 0,939**; ejecutada **0,0344/h**; DD **20,38 / 20,38 %**.
- Seed 44: mejor **Gen483**; fitness **−0,569426 → −0,423387** (Δ **+0,146039**), 5 mejoras, 14 sin avance; PF **0,659 / 0,925**; ejecutada **0,0320/h**; DD **20,88 / 20,49 %**.

**Conclusión del tramo previo:** hubo progreso en el fitness de las 3 ramas, **sin mejoría económica suficiente**; no es correcto afirmar un techo matemático absoluto. Las 122 generaciones por semilla aportaron **46.848 evaluaciones** (8.448+38.400). [CI Gen398–497](https://github.com/Asriel550/Forex/actions/runs/42546596752).

## 4. PR #38 — Fase 1: sensibilidad del Gate de Convicción Pura

**Hipótesis:** un gate p85 excesivamente rígido ocultaría alfa que emergiera con p80, p75 o un percentil dinámico por volatilidad.

**Diseño:** 4 modos × 3 semillas × 2 generaciones × 128 modelos = **3.072** evaluaciones nuevas, con control p85; cada escenario utiliza el mismo fundador congelado de Gen376–397 por semilla. El modo dinámico aplica **p75** si la volatilidad relativa es baja, **p80** en rango medio y **p85** si es alta; referencia histórica de hasta 48 observaciones anteriores, percentiles de convicción con ventana causal de 500 y liquidaciones vencidas. El criterio de paso exploratorio era PF>1,0 junto con frecuencia ≳1,0/h; p75 con deterioro severo de PF (<0,85) implicaba abandonar esta hipótesis.

**Todos los ganadores del smoke (G=generación local 1 o 2, no Gen498):**

| Seed | Gate | G | PF Q1 | PF Q2 | Real/h | DD semestre % | Aceptadas gate pre-cupo | Cumple |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | p85 | 2 | 0.690 | 0.919 | 0.0327 | 20.42 | 1692 | No |
| 42 | p80 | 2 | 0.738 | 0.903 | 0.0429 | 20.19 | 2276 | No |
| 42 | p75 | 2 | 0.760 | 0.939 | 0.0534 | 20.00 | 3151 | No |
| 42 | p75/p80/p85 vol. | 2 | 0.808 | 0.900 | 0.0694 | 20.32 | 2429 | No |
| 43 | p85 | 2 | 0.698 | 0.916 | 0.0354 | 20.42 | 1790 | No |
| 43 | p80 | 2 | 0.756 | 0.898 | 0.0483 | 20.16 | 2349 | No |
| 43 | p75 | 2 | 0.780 | 0.927 | 0.0534 | 20.42 | 3048 | No |
| 43 | p75/p80/p85 vol. | 2 | 0.732 | 0.933 | 0.0466 | 20.12 | 2440 | No |
| 44 | p85 | 1 | 0.649 | 0.931 | 0.0330 | 20.82 | 1810 | No |
| 44 | p80 | 2 | 0.780 | 0.928 | 0.0545 | 20.00 | 2307 | No |
| 44 | p75 | 2 | 0.789 | 0.932 | 0.0483 | 20.56 | 3178 | No |
| 44 | p75/p80/p85 vol. | 2 | 0.738 | 0.973 | 0.0422 | 20.61 | 2407 | No |

**Lectura:** p75 elevó el número de admisiones antes del cupo a **3.151 / 3.048 / 3.178** respectivamente; la frecuencia **real** continuó solo entre **0,0483 y 0,0534/h** con PF Q1 inferior a **0,85**. El gate dinámico tampoco produjo beneficio consistente. Además, un campeón de p75 ha mutado sus genes upstream durante el smoke: la atribución puramente causal al percentil debe basarse en el subinforme `parent` de genoma fijo, **no** en comparar dos campeones genéticos independientes.

**Decisión:** insuficiencia de alfa del sistema bajo estas perturbaciones del gate; **seguir a Fase 2**, no relajar p85 en producción. [Run fase 1](https://github.com/Asriel550/Forex/actions/runs/42549591449) · [Resumen JSON](https://github.com/Asriel550/Forex/actions/runs/42549591449/artifacts/13644712136).

## 5. PR #38 — Fase 2: objetivos alternativos y pérdida asimétrica

**Hipótesis:** la mera dirección de la siguiente M5 no contiene la estructura de movimiento necesaria para mejorar beneficio/riesgo. Se probaron **tres** variantes, en cada caso **16 generaciones × 128 modelos × 3 semillas** = **6.144** evaluaciones por variante, **18.432** en total.

**Etiquetas selladas exclusivamente para entrenamiento:**
- ATR14 M15 calculada con **M15 cerradas**; horizonte futuro de **6 M5, 30 minutos**; solo se permite una etiqueta con horizonte completo dentro de H1 IS y continuidad de las M5.
- `atr_expansion`: excursión dominante favorable superior a **1× ATR M15**; clases +1/−1/0.
- `rr15`: evento de tocar **take 0,75× ATR** antes de **stop 0,50× ATR** (relación 1,5), regla conservadora de **stop primero** ante ambos niveles en una misma M5. **Esta etiqueta no representa una operación real con R:R 1,5**.
- `asymmetric_fp`: dirección M5 con penalización aumentada ante falsos positivos de convicción alta y bajo momentum M5/M15. El gradiente no se utiliza; es un regularizador de fitness evolutivo.
- De 35.261 filas, **33.683** tuvieron etiqueta utilizable; los conteos brutos de los 35.261 renglones fueron ATR +1 **7.961**, −1 **8.248**, 0 **19.052**; R:R +1 **10.117**, −1 **10.448**, 0 **14.696**. La clase cero total incluye etiquetas no válidas; no debe interpretarse como distribución exclusiva de los 33.683 válidos.

La cabeza de dirección residual añade **13 genes** a los **11** genes upstream. La evaluación del rendimiento no cambia el contrato binario de vencimiento M5, ni su gate p85, cupo o stops. **Las etiquetas de futuro se usan solo para la función objetivo en IS; nunca como inputs disponibles en el instante de entrada.**

**Ganadores de cada variante:**

| Target | Seed | G | PF Q1 | PF Q2 | Real/h | DD semestre % | Cumple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| atr_expansion | 42 | 9 | 0.810 | 1.011 | 0.0558 | 20.04 | No |
| atr_expansion | 43 | 8 | 0.893 | 0.988 | 0.1062 | 20.23 | No |
| atr_expansion | 44 | 7 | 1.051 | 1.008 | 0.4543 | 19.34 | No |
| rr15 | 42 | 8 | 0.985 | 1.004 | 0.4166 | 20.08 | No |
| rr15 | 43 | 12 | 0.713 | 0.972 | 0.0357 | 20.42 | No |
| rr15 | 44 | 14 | 0.881 | 1.029 | 0.0493 | 20.03 | No |
| asymmetric_fp | 42 | 16 | 0.818 | 1.017 | 0.0602 | 20.27 | No |
| asymmetric_fp | 43 | 15 | 0.821 | 1.002 | 0.0531 | 20.55 | No |
| asymmetric_fp | 44 | 13 | 1.111 | 1.046 | 0.4499 | 16.16 | No |

**Lectura:** `atr_expansion` seed44/G7 logró PF **1,051 / 1,008** y **0,4543/h**, pero DD **19,34 %**. `asymmetric_fp` seed44/G13 logró PF **1,111 / 1,046**, **0,4499/h** y DD **16,16 %**, el mejor equilibrio aislado de esta fase, **no reproducido en seeds 42/43**. `rr15` no aportó una señal consistente. Ninguna variante obtuvo **PF>1,10 en ambos trimestres y en las tres semillas** ni alcanzó aceptación final.

**Decisión:** Fase 2 fallida según el criterio predefinido; **activar Fase 3**. [Run fase 2 y 3](https://github.com/Asriel550/Forex/actions/runs/42551086458) · [Resumen fase 2](https://github.com/Asriel550/Forex/actions/runs/42551086458/artifacts/13645247715).

## 6. PR #38 — Fase 3: estructura M15 y selección por régimen GARCH

**Hipótesis:** el problema central sería mezclar regímenes no operables con otros en los que hay ventaja, o un contexto M15 insuficientemente condicionado.

**Diseño:** **2 variantes × 3 semillas × 15 generaciones × 128 modelos = 11.520** evaluaciones; p85 y payoff M5 se mantienen. Se añadió un residuo direccional de M5/M15:
- `m15_attention`: ponderación *causal* sencilla entre el cuerpo de la última M15 cerrada y el de la M15 anterior. No es un transformador ni atención multi-cabeza; es un mecanismo escalar de atención.
- `garch_regime`: veto upstream **por banda de volatilidad** de GARCH M5 actual respecto a la mediana histórica desplazada, con límites optimizables; utiliza GARCH precomputado congelado. Restringe operaciones a regímenes seleccionados; no prueba que exista alfa en todos los estados del mercado.

**Ganadores de cada variante:**

| Variante | Seed | G | PF Q1 | PF Q2 | Real/h | DD semestre % | Cumple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| garch_regime | 42 | 10 | 1.272 | 1.224 | 0.2083 | 6.40 | No |
| garch_regime | 43 | 15 | 1.231 | 1.173 | 0.2011 | 7.47 | No |
| garch_regime | 44 | 15 | 1.255 | 1.210 | 0.2209 | 8.27 | No |
| m15_attention | 42 | 15 | 0.707 | 1.037 | 0.0388 | 20.77 | No |
| m15_attention | 43 | 10 | 0.810 | 0.973 | 0.0555 | 20.04 | No |
| m15_attention | 44 | 11 | 1.026 | 1.024 | 0.4291 | 19.45 | No |

**Lectura de la variante GARCH:**
- **Seed 42, G10:** PF Q1 **1,272** y Q2 **1,224**; DD **6,40 %**, ejecutada **0,2083/h**. Supera PF y DD, pero **fracasa frecuencia ≥1,0/h**.
- **Seed 43, G15:** PF **1,231 / 1,173**; DD **7,47 %**, ejecutada **0,2011/h**. Fracasa **PF Q2** y frecuencia.
- **Seed 44, G15:** PF **1,255 / 1,210**; DD **8,27 %**, ejecutada **0,2209/h**. Supera PF y DD, fracasa frecuencia.

El condicionamiento GARCH **cambió de modo apreciable el perfil de riesgo IS** y es la hipótesis que merece más estudio. No se puede afirmar que haya conseguido los criterios, alfa OOS ni que la frecuencia potencial pueda extrapolarse: opera solo alrededor de una quinta parte de la frecuencia requerida. La atención M15 simple no resolvió PF, DD ni actividad.

**Decisión:** **0 variantes aceptadas en tres semillas**. [CI completo](https://github.com/Asriel550/Forex/actions/runs/42551086458) · [Resumen final](https://github.com/Asriel550/Forex/actions/runs/42551086458/artifacts/13645433770).

## 7. Contabilidad integral de experimentos

| Segmento | Generaciones | Genomas por gen. | Semillas | Configuraciones | Evaluaciones |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR36 Gen376–397 | 22 | 128 | 3 | 1 | **8.448** |
| PR37 Gen398–497 | 100 | 128 | 3 | 1 | **38.400** |
| PR38 Fase 1 (4 gates) | 2 | 128 | 3 | 4 | **3.072** |
| PR38 Fase 2 (3 targets) | 16 | 128 | 3 | 3 | **18.432** |
| PR38 Fase 3 (2 arquitecturas) | 15 | 128 | 3 | 2 | **11.520** |
| **Total de tramos principales** | — | — | — | — | **79.872** |

Las **46.848** primeras corresponden a la evolución histórica Gen376–497; las **33.024** posteriores corresponden a PR38. El total **no** cuenta smokes exploratorios de PR34/PR36 ni evaluaciones adicionales de comparación con los fundadores; tampoco equivale a 79.872 muestras de mercado independientes. G1/G2/G7/G15 de PR38 son generaciones **locales de cada variante**, no prolongación de la numeración global Gen497.

## 8. Cronología del CI e incidencias

1. **GARCH de marzo, PR34:** divergencia en un refit numéricamente inestable; misma M1/M5; se congeló la reconstrucción correcta y se verificó el origen de 2.859 propuestas.
2. **Persistencia PR35:** migración a Releases privadas con SHA256 y verificación del universo semestral.
3. **Corrección PR36:** se retiraron inputs M1 introducidos incorrectamente; nueva Release M5+M15 v2 y nuevo replay. Las métricas M1 experimentales se consideran **anuladas** como evidencia de estrategia.
4. **Meseta PR37:** ejecución de 100 generaciones adicional y registro de mejoras de fitness que no se tradujeron en resultados operativos.
5. **Fase 1 PR38:** [42549591449](https://github.com/Asriel550/Forex/actions/runs/42549591449) completada con éxito; 12 smokes de gate con sus controles genéticos.
6. **Primer intento de Fases 2–3:** [42550731368](https://github.com/Asriel550/Forex/actions/runs/42550731368) **falló antes de iniciar la evolución**, porque faltaban `--golden` y `--golden-manifest` al invocar la verificación del dataset de etiquetas en los jobs. **No fue un fracaso de un target ni una discrepancia de datos.**
7. **Corrección y reejecución:** se reparó la línea de comando; [42551086458](https://github.com/Asriel550/Forex/actions/runs/42551086458) terminó **SUCCESS**: sellado/validación de etiquetas, **9** trabajos de Fase 2, evaluación de condiciones, **6** trabajos condicionales de Fase 3 y consolidación final. Los hashes no se modificaron.
8. **Resultado del test final:** `all_criteria_met=false`, `accepted_variants=[]`, `oos_accessed=false`. Las PR siguen en borrador, sin despliegue ni promoción automática.

## 9. Riesgos metodológicos y límites de la interpretación

**a. Selección reiterada sobre el mismo IS.** Se exploraron 46.848 evaluaciones históricas y 33.024 adicionales, con múltiples reglas. Cualquier mejora observada sobre enero–junio podría reflejar sobreajuste o selección entre hipótesis. Semillas diferentes **no sustituyen** un periodo independiente.

**b. Q3 intocable.** El límite OOS no se abrió para evaluar variantes ni elegir ganadores. El control descansa en filtros de meses, snapshots sellados, validaciones por fecha y el campo `oos_accessed=false`; éste último es una declaración de la ejecución, no una prueba estadística de ausencia de fuga por sí solo.

**c. Denominadores de frecuencia.** Propuestas elegibles, admisiones del gate antes de cupo, operaciones gate-only sin stops y contratos **ejecutados** tras stop son métricas distintas. La mejora de PF de GARCH puede venir acompañada de un menor número de contratos y no autoriza inferir que se mantendrá el PF al multiplicar la frecuencia por cinco.

**d. Proxy de target contra producto ejecutado.** ATR, R:R1.5 y excursión de 30 min son targets de aprendizaje; el instrumento económico evaluado siguió siendo la opción binaria de cinco minutos con payout 0,8. El sesgo conservador «stop primero» no reproduce la verdadera secuencia intrabar sin precios tick.

**e. Evidencia parcial de alfa por régimen.** PF>1,20 y DD<15 % en 2/3 semillas del filtro GARCH son indicios IS; faltan número de operaciones, intervalos de confianza por régimen, estabilidad entre bloques temporales, estabilidad de parámetros y pruebas de sensibilidad antes de atribuir significancia.

**f. Comparaciones genéticas versus controles fijos.** En Fase 1 cada modo evolucionó la cabeza de admisión durante dos generaciones: un mejor PF observado entre campeones mezcla el efecto del gate y el efecto del genoma. El análisis `parent` a genoma fijo es la comparación apropiada para sensibilidad pura del percentil.

**g. Controles de riesgo.** El DD cae a 6–8 % bajo la selección GARCH porque se admiten muchas menos operaciones. Esto es deseable como control de riesgo, pero **no** satisface densidad. No bajar los requisitos de PF/frecuencia/DD para declarar éxito.

**h. Reproducibilidad técnica.** Los hashes sellan los inputs y las salidas. No implican robustez de los estimadores frente a cambios de spread, latencia, ejecución real, costes, proveedor de datos, régimen futuro o borrado administrativo de Releases.

## 10. Decisiones técnicas para la siguiente investigación (sin ejecutarlas)

1. Conservar como **punto de partida experimental**, no como estrategia aprobada, la cabeza `garch_regime` de PR38. Registrar por semilla banda GARCH, número de contratos, proporción de rechazo por gate, stops y PF por subperiodos. Sin estos denominadores, PF y DD aislados pueden sobrestimar la calidad.
2. Diseñar **frontera actividad–calidad** condicionada al régimen (más oportunidades dentro de bandas válidas versus pureza del filtro), con cuotas por horas observadas y PF trimestral, manteniendo fijo el criterio final ≥1,0/h. Evitar aumentar frecuencia abriendo sin control los regímenes malos.
3. Antes de nuevos miles de generaciones, introducir **walk-forward interno con embargo temporal** en el IS de Q1/Q2, separar generación/selección/verificación y medir incertidumbre. Congelar el protocolo y sus umbrales *antes* de cualquier evaluación OOS.
4. Mantener Q3 julio–septiembre 2026 **cerrado** hasta una autorización y un protocolo final aparte; no utilizarlo como generador de hipótesis. No fusionar/activar en vivo ninguno de los genomas de estas fases.
5. Preservar copias independientes de los cuatro activos canónicos con sus SHA256, y conservar además los reportes Actions (los últimos son temporales, típicamente retención de 90 días).

## 11. Referencias primarias para reproducibilidad

**Código y protocolos:** [PR #34](https://github.com/Asriel550/Forex/pull/34) · [PR #35](https://github.com/Asriel550/Forex/pull/35) · [PR #36](https://github.com/Asriel550/Forex/pull/36) · [PR #37](https://github.com/Asriel550/Forex/pull/37) · [PR #38](https://github.com/Asriel550/Forex/pull/38).

**Scripts:** `src/upstream_filters.py`, `src/conviction_regime_gate.py`, `scripts/build_m5_full_is_universe.py`, `scripts/build_m5_m15_alpha_targets.py`, `scripts/run_m5_dynamic_upstream.py`, `scripts/run_m5_gate_sensitivity.py`, `scripts/run_m5_m15_target_experiments.py`.

**Workflows de evidencia:** [replay M5+M15](https://github.com/Asriel550/Forex/actions/runs/42542842719) · [22 generaciones](https://github.com/Asriel550/Forex/actions/runs/42544173112) · [100 generaciones adicionales](https://github.com/Asriel550/Forex/actions/runs/42546596752) · [Fase 1](https://github.com/Asriel550/Forex/actions/runs/42549591449) · [primer fallo de argumentos](https://github.com/Asriel550/Forex/actions/runs/42550731368) · [Fases 2/3 y consolidación final](https://github.com/Asriel550/Forex/actions/runs/42551086458).

**Fuente numérica de este informe:** JSON consolidados de [Fase 1](https://github.com/Asriel550/Forex/actions/runs/42549591449/artifacts/13644712136), [Fase 2](https://github.com/Asriel550/Forex/actions/runs/42551086458/artifacts/13645247715) y [Fase final](https://github.com/Asriel550/Forex/actions/runs/42551086458/artifacts/13645433770). Los datos de las tablas fueron transcritos automáticamente desde las líneas `PR38_PHASE1_SUMMARY`, `PR38_PHASE2_SUMMARY` y `PR38_FINAL` de los logs de GitHub Actions, redondeando solo su presentación.

---

**Veredicto documental:** se demostró que el algoritmo puede mejorar **su comportamiento IS al seleccionar regímenes GARCH**, no que se haya obtenido un sistema de 1,0/h con PF>1,20 en dos trimestres, DD<15 % y tres semillas consistentes. **Los cinco experimentos sustantivos y el barrido de gates se cierran como no aprobados, sin promover genomas y sin tocar Q3.**
