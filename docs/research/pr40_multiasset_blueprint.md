# PR #40 — Blueprint y primera implementación: escalado horizontal EURUSD + GBPUSD + AUDUSD + USDCAD

**Estado:** diseño reproducible y motor de cartera implementados; validación empírica **BLOQUEADA** hasta adquirir/sellar datos reales M5+M15/GARCH de los tres pares nuevos. No existen PF, frecuencia ni DD verificados de portfolio multiasset. No se despliega a producción. **Q3 julio–septiembre de 2026 totalmente intocado.**

> **Aclaración de fuente, posterior al primer blueprint:** el usuario solicitó explícitamente descargar los otros pares desde **HistData.com**. HistData distribuye datos de barras M1 o tick, **no M5/M15 nativos**. Por tanto, la procedencia válida será **ZIP mensual M1 como insumo técnico de agregación → velas M5 y M15 cerradas y verificadas → GARCH M5 causal → inputs para la cabeza PR38 congelada**. Ninguna variable M1 entra como señal. [Nuevo workflow de adquisición](https://github.com/Asriel550/Forex/actions/workflows/pr40-acquire-histdata.yml) y [script](../../scripts/acquire_pr40_histdata.py). El periodo de descarga permitido es exclusivamente enero–junio de 2026, y cualquier minuto de junio local que caiga desde 2026-07-01 00:00 UTC se descarta antes de agregar.

## 1. Decisión: conservar el núcleo PR38, NO la evolución PR39

Se conserva, sin entrenamiento ni recalibración, el campeón **`garch_regime` PR38** de cada semilla, [Actions #42551086458](https://github.com/Asriel550/Forex/actions/runs/42551086458):

| Seed | Gen original PR38 | PF EUR Q1 | PF EUR Q2 | EUR real/h H1 | EUR DD H1 | Banda/convicción |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 10 | 1,272 | 1,224 | 0,2083 | 6,40 % | GARCH 0%, p85 |
| 43 | 15 | 1,231 | 1,173 | 0,2011 | 7,47 % | GARCH 0%, p85 |
| 44 | 15 | 1,255 | 1,210 | 0,2209 | 8,27 % | GARCH 0%, p85 |

Se conserva la red direccional Gen375/semilla43/p80 como parte interna original, la cabeza residual y upstream del genoma PR38 correspondiente a la semilla, el **Gate de Convicción Pura p85** con historia causal y calentamiento original, la banda GARCH **sin ensanchar (0%)**, expiración M5, payout supuesto +80%/−100%, stake y stops originales. No se transfieren los genomas ni el fitness de PR39; no se busca un nuevo campeón con resultados de los otros pares.

**Solo temporalidades de señal M5 y M15 cerradas.** Ningún input, indicador ni decisión de M1. No alimentar las cabezas con precios M1 ni etiquetas de futuro. Si un proveedor usa datos M1 para agregación previa, debe demostrar cierre/continuidad de velas M5/M15 y no exponer M1 a inferencia; la fuente de PR40 es HistData **Generic ASCII M1 exclusivamente como materia prima**, reconstruida en velas OHLCV **M5 y M15 completas**. El modelo nunca recibe indicadores M1.

## 2. Cesta propuesta y realidad de los datos

La canasta inicial (cuatro instrumentos) se fija **antes** de consultar rentabilidades de otros pares:

1. EURUSD, control ya verificado.
2. GBPUSD, candidato nuevo, **sin H1 sellado**.
3. AUDUSD, candidato nuevo, **sin H1 sellado**.
4. USDCAD, candidato nuevo, **sin H1 sellado**.

No se incluye automáticamente NZDUSD como quinto por el mero hecho de que los cuatro no alcancen 1,0/h: ese paso requeriría nueva fuente verificada y protocolo previo, no selección oportunista usando el IS.

**Auditoría hecha en la rama PR39:** `datasets/` contenía únicamente `eurusd/`, el directorio `data/` solo `.gitkeep` y las cuatro Releases de datos disponibles eran **todas EURUSD** (GARCH PR35, universo completo v1 retirado, universo M5/M15 v2 y etiquetas ATR/R:R PR38). No había manifests ni Release de GBPUSD/AUDUSD/USDCAD.

[Inventario versionado](../../datasets/multiasset/pr40_asset_inventory.json): `EURUSD=verified` y `GBPUSD/AUDUSD/USDCAD=missing`. El informe de readiness escribe **null** para PF/frecuencia/DD, nunca cero (cero daría a entender que se midió y falló).

**El volumen agregado de cuatro pares es una hipótesis, no una medición:** `4×(0,20–0,25/h)=0,80–1,00/h` sumando cuentas aisladas, y cinco hipotéticos pares `1,00–1,25/h`. En una **cuenta común** pueden existir señales simultáneas, límites horarios y stops globales, de modo que la frecuencia ejecutada será diferente de la suma (y puede ser inferior). Correlaciones de shocks USD pueden elevar el DD conjunto incluso cuando el DD de los pares aislados sea pequeño.

## 3. Evidencia mínima antes de ejecutar un backtest real

Para cada par adicional se necesita:

- OHLCV M5 y M15 **reconstruidos a partir de HistData mensual, completamente cerrados** para 2026-01-01 00:00 UTC a 2026-07-01 00:00 UTC, con proveedor, instrumento, zona horaria, definiciones de volumen, timestamps, continuidad/gaps y hashes de origen registrados.
- Serie GARCH de M5 **estimada causalmente en ese mismo activo** con la receta/MC alineadas al motor original (no copiar la volatilidad EUR a GBP, AUD o CAD). Reportar parámetros/refits fallidos, warmup y huellas de reproducibilidad.
- **Los mismos 32 inputs de dirección** y **las mismas ocho features M5/M15** en el mismo orden que la arquitectura PR38; aplicar **el normalizador original congelado Gen375**. Registrar porcentaje de clipping/fuera de distribución, especialmente por cambios de escala de precios/tick-volume, sin reentrenar el modelo.
- `row_times_ns`, `outcome` de expiración M5, `volatility`, `entry`, `margin`, `timeframe`, `direction_inputs`, `legacy_mask` (schema `pr40_m5_m15_symbol_universe_v1`), con SHA256 binario del NPZ y de todos los arrays, ventanas Q1 y Q2 y horas observadas por mes.
- Manifiesto con `source_timeframes=["M5","M15"]`, `signal_timeframes=["M5","M15"]`, `q3_accessed=false`, `start_utc=2026-01-01T00:00:00+00:00`, `end_exclusive_utc=2026-07-01T00:00:00+00:00`, seis meses H1, `garch_m5_causal_refit=true`, `gen375_normalizer_frozen=true`, hashes del OHLCV nativo, receta GARCH y normalizador original.
- Revisión independiente de procedencia y Release privada sellada para cada símbolo, con `universe_sha256` copiado al inventario PR40. No tratar una bandera `verified` escrita a mano como sustituto de inspeccionar fuente/preparación. El validador comprueba hash, forma, meses y contrato; **una firma de archivo no demuestra por sí sola que la ingeniería de features sea correcta**.

Solo al finalizar esas tareas se permite cambiar cada elemento `missing` a `verified` en el inventario. La fase de portfolio **se omite** por diseño cuando falte cualquiera de las tres fuentes; no se imputan precios ni oportunidades. Las coberturas mensuales de mercado de cada par deben coincidir dentro del **10%** de las horas observadas EURUSD; se reportan gaps, sin rellenarlos con información futura.

## 4. Arquitectura multiasset implementada

### A. Inferencia por par, idéntico PR38

`scripts/run_m5_multiasset_pr40.py` carga el genoma original **por semilla** desde los artefactos originales PR38 y reproduce en EURUSD **PF Q1, PF Q2, DD y frecuencia** antes de intentar otros símbolos. Sobre cada universo nuevo, usa los pesos congelados Gen375, la cabeza residual PR38, las once variables de admisión y la **banda de régimen GARCH original 0%**; aplica el Gate p85 sobre el **historial causal de ese instrumento**, con calentamiento de 500 propuestas.

No hay evolución, reentrenamiento, ajuste del gate ni normalizadores nuevos. Cada par genera una tabla de **decisiones fechadas**, `side`, `outcome` (solo para liquidar posteriormente) y `gate_reason` (causal). La matriz de señales no puede incluir columnas M1 ni etiquetas futuras como features de inferencia.

### B. Una sola cuenta de cartera

`src/m5_multiasset_portfolio.py` une las decisiones de todos los símbolos en una única línea temporal de cierres M5. Si varios pares proponen a la misma hora, resuelve colisiones con prioridad **determinista rotatoria por slot M5**, independiente del acierto y del profit futuro.

- **Cupo global único de 2 contratos por hora móvil**, no 2/h multiplicado por cada par.
- **Un único balance**, stake original por contrato, reserva de stake de operaciones pendientes y **un único stop de drawdown heredado** (`min(account.max_drawdown,risk.drawdown_limit)`). No se aumentan el presupuesto ni la exposición para fabricar frecuencia.
- Liquidaciones de contratos M5 simultáneos **en conjunto**, al expirar; el DD se evalúa tras el cierre del grupo antes de admitir nuevas entradas. Impide que el orden arbitrario de dos pares cambie el resultado de stop.
- Se calculan tres contabilidades separadas: **Q1 con su cuenta inicial, Q2 con su cuenta inicial y H1 con una sola cuenta continua**, preservando el stop de Q1 para Q2 en el ledger semestral.
- Se reportan **contribuciones por activo realmente ejecutadas en el portfolio** (no suma de cuentas hipotéticamente independientes): operaciones, wins/losses, PnL, PF, admisiones, choques de cupo, contratos concurrentes y stops. También se reporta el replay standalone de cada símbolo para comparar desvíos por competencia de capital.
- Un **único denominador de horas H1 verificado** del calendario EURUSD original, con cobertura mensual comparable en las nuevas fuentes. La capacidad real de cartera se calcula después del gate global, de los stops y de las colisiones.

Un replay de un único símbolo debe coincidir con el contrato `src/replay_engine.replay_contracts` original para PF, DD, PnL y frecuencia. Las regresiones de CI impiden abrir capital o superar la restricción 2/h al combinar activos.

## 5. Condiciones ex ante de aceptación

Se examinan semillas 42/43/44 **por separado**, con los genomas PR38 sin mezclar semillas ni seleccionar el mejor activo mediante resultados del mismo IS.

| Métrica | Regla propuesta de PR40 |
| --- | --- |
| PF cartera Q1 | Estrictamente **>1,20** |
| PF cartera Q2 | Estrictamente **>1,20** |
| PF cartera H1 continuo | Estrictamente **>1,20** |
| Frecuencia ejecutada en H1 | **≥1,00/h**; rango buscado 1,00–1,25/h |
| DD máximo cartera H1 | **<12%**, heredado como umbral estricto del protocolo PR39 (el stop operativo original no se altera) |
| PnL Q1/Q2 | Ambos positivos |
| Stops | Ninguno en Q1, Q2 o H1 |
| Replicabilidad | Cumplimiento en **las tres semillas originales**, no mezcla de parámetros entre ellas |
| Cuarentena OOS | `q3_accessed=false`, sin archivos posteriores al 2026-06-30 |

El límite <12% es **criterio de investigación**, más exigente que el <15% del protocolo histórico PR38; no se cambia el stop operativo original para hacer pasar o fallar casos. Si cualquiera de los activos empeora significativamente los resultados, se informa; no se excluye retroactivamente el activo basándose en PF del IS. La posibilidad de un quinto par necesita un nuevo plan predeclarado.

## 6. Riesgos concretos del escalado

**Transferencia de pesos entre activos:** los genes y el normalizador entrenados sobre EURUSD pueden estar fuera de distribución sobre GBPUSD, AUDUSD o USDCAD; la normalización original y ratios de GARCH no garantizan ventaja direccional transferible. Las 4 series deben obtenerse de fuentes nativas compatibles con el mismo payout supuesto. La pérdida es total de la apuesta ante fallo y el payout hipotético 80% puede no ser el disponible en un broker real.

**Correlación y clústeres USD:** GBPUSD y AUDUSD comparten USD como divisa cotizada; USDCAD tiene USD como divisa base. No son riesgos independientes. Las divisas mayores pueden reaccionar juntas a la misma noticia. Un DD individual de 6–8% no implica DD combinado similar; por eso el replay compartido requiere reconciliar PnL y eventos simultáneos.

**Overfitting de la cesta:** se fijan los cuatro símbolos antes de ver datos IS, sin selección retrospectiva de subconjuntos ganadores. Las tres semillas prueban variabilidad del mismo semestre y **no sustituyen OOS**. Antes de cualquier cambio de producción se necesitarían análisis de sensibilidad, costos/latencia y evaluación futura autorizada aparte.

**Reproducibilidad y disponibilidad:** los datos de EURUSD están en Release con SHA256, pero los artefactos Actions de PR38 pueden caducar; hay que preservarlos antes de la ventana de retención. Una Release es persistente **mientras no se borre administrativamente**.

## 7. Código, pruebas y ejecución

- `src/m5_multiasset_portfolio.py`: ledger con una cuenta, stop único y 2/h global.
- `src/m5_multiasset_inputs.py`: lector de universos nuevos con esquema, SHA, M5/M15, normalización, GARCH, periodos y cuarentena Q3.
- `scripts/run_m5_multiasset_pr40.py`: genomas PR38 congelados, gate p85 separado por par, reconciliación EURUSD baseline, benchmarks standalone y cartera.
- `scripts/check_pr40_data_readiness.py`: certifica qué símbolos están realmente disponibles; emite métricas de cartera **null** mientras falten.
- `datasets/multiasset/pr40_asset_inventory.json`: lista exacta de fuentes/Release y estado.
- `tests/test_m5_multiasset_portfolio.py`: regresiones de equivalencia con EUR individual, colisiones globales, stop común, no reseteo semestral, rechazo de Q3 y features M1, y bloqueo ante ausencia de datos. **Las fixtures sintéticas NO son resultados financieros de trading.**
- `.github/workflows/pr40-m5-multiasset.yml`: primero tests + readiness, y **solo si existen los cuatro archivos verificados** habilita tres jobs reales de cartera y un resumen cross-seed. No crea PF ficticios para símbolos faltantes.

**Estado final de la primera iteración PR40:** implementación del protocolo y verificación estructural, **sin evaluación de los nuevos activos por falta de datos H1**. No se puede asegurar capacidad ≥1,0/h, PF ni DD hasta construir los tres archivos. El workflow puede terminar con tests verdes y los jobs de backtest real omitidos; eso **no significa aprobación financiera**.

**Q3 permanece completamente sellado.**


## 8. Primera ejecución de CI — validación técnica, datos externos aún ausentes

[Run #42565772995](https://github.com/Asriel550/Forex/actions/runs/42565772995):
**18 pruebas aprobadas**, resultado técnico `SUCCESS`. Se verificaron
equivalencia del replay de un solo activo frente al contrato original,
cupo global entre pares, liquidaciones M5 simultáneas, stops de DD
compartidos, una cuenta H1 continua y rechazo de Q3 e inputs M1.

[Reporte de readiness](https://github.com/Asriel550/Forex/actions/runs/42565772995/artifacts/13650579025):

- `verified_assets=["EURUSD"]`
- `missing_assets=["GBPUSD","AUDUSD","USDCAD"]`
- `ready_for_real_backtest=false`
- `portfolio_pf_q1=null`, `portfolio_pf_q2=null`
- `portfolio_executed_contracts_per_hour=null`
- `portfolio_max_dd_pct=null`
- `q3_accessed=false`

Los jobs reales por semilla 42/43/44 y su consolidación se marcaron
**SKIPPED deliberadamente**. No existe un backtest real de cuatro pares,
ni evidencia que acredite 1.0–1.25/h, PF o DD combinados. **CI verde es
prueba de seguridad del diseño, no validación de la estrategia.**

Próximo trabajo de datos: obtener y verificar por separado los seis meses
de OHLCV M5/M15 derivado de archivos HistData mensuales y GARCH causal de GBPUSD, AUDUSD y USDCAD;
reconstruir exactamente las 32 entradas normalizadas por el Gen375 original;
comparar cobertura, clipping y firmas; publicar Releases selladas;
actualizar el inventario y entonces ejecutar por primera vez la validación
financiera de cartera H1. Q3 seguirá fuera del flujo.

## 9. Adquisición HistData ejecutada a solicitud del usuario

La implementación en `scripts/acquire_pr40_histdata.py` permite descargar los **18 ZIP oficiales mensuales** de GBPUSD/AUDUSD/USDCAD para los seis meses 202601–202606, valida par/mes/formato y SHA256, normaliza el huso del proveedor (EST fijo sin DST), excluye timestamps UTC posteriores o iguales a julio antes de construir las velas, descarta grupos incompletos M5 y M15, calcula huellas de OHLCV y publica un `ohlcv.npz` junto con los manifiestos y hashes. La Release se crea por par solo si los seis ZIP se recibieron y los controles pasaron; **no** inventa operaciones de cartera. El GARCH por par y los inputs normalizados PR38 son un paso posterior imprescindible para el backtest económico.

## Adenda — adquisición HistData ejecutada

Se implementó `scripts/acquire_pr40_histdata.py` y el workflow
[PR40 — Acquire real HistData GBP AUD CAD](https://github.com/Asriel550/Forex/actions/workflows/pr40-acquire-histdata.yml).
El proveedor distribuye únicamente archivos base **M1 o tick**, no velas M5/M15 ya cerradas. Para estos pares se descargan los seis ZIP HistData enero–junio, verificando nombre CSV, CRC, SHA256 y huecos, y de ahí se agregan M5 y M15; **M1 no se usa como rasgo de trading**.

La descarga exige identificar `/get.php` en el HTML; HistData también ofrece `/getStatus.php` con campos muy similares. Los primeros reintentos fallaron por seleccionar ambos formularios; [CI actual de adquisición](https://github.com/Asriel550/Forex/actions/runs/42568580038) incluye la corrección y su test.

El archivo `ohlcv.npz` tendrá ambas temporalidades, fingerprints OHLCV y fuentes firmadas. Es **solo un dataset de velas**, no una validación de la red ni de la cartera: GARCH por símbolo y construcción del universo neuronal siguen siendo fases distintas.
