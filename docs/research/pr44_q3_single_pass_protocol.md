# PR44 — Pre-registered single-pass audit OOS Q3 2026

## Estado real de disponibilidad (verificado antes de entrar a Q3)

Al inspeccionar **únicamente el repositorio Forex**, en 2026-11-09
las Releases publicadas consistían en los universos **H1 Q1/Q2** de
EURUSD, GBPUSD, AUDUSD y USDCAD. **No se encontró ninguna Release
Q3 M5+M15/GARCH** para los cuatro símbolos. Los ZIP
`datasets/eurusd/EURUSD_M15_UTC_2026.zip` son M15 de un solo par,
no un sustituto de los universos de inferencia M5/M15/GARCH requeridos
por los genomas de 22 genes. Tampoco existen Q3 sellados para GBP/AUD/CAD.

Por tanto, **no hay métricas Q3 reales que informar todavía**.
El workflow se define **manual-only / trigger deliberado**; primero
verifica las cuatro Releases Q3 **sin instalar paquetes**. Si falta
cualquiera, se interrumpe inmediatamente, sin producir falsos PF,
DD, “PASS” ni reemplazar la fuente con los datos H1.

## Registro previo: modelos, fechas y controles

- Exactamente **24 cromosomas congelados**: 12 controles PR41/PR42
  (4 pares × seeds42/43/44) y 12 descendientes de PR43
  (100 generaciones adicionales).
- SHA256 de **cada uno de los 24 genomas fijado en código** usando el
  consolidado PR43 SUCCESS #42653546285. Se verifica el SHA
  contra los artefactos finales de
  PR41 run #42576852134 y PR43 run #42647437793 antes de abrir Q3.
- No se muta ningún genoma; sin selección, crossover,
  reentrenamiento, búsqueda, promoción ni post-hoc.
- Período exacto: **2026-07-01 00:00Z ≤ decisión < 2026-10-01
  00:00Z**. Ninguna ejecución puede necesitar la liquidación en Q4;
  decisiones con vencimiento desde 2026-10-01 se excluyen.
- Datos del mismo símbolo: M5 y última M15 completamente cerrada;
  GARCH causal sobre historial propio M5, con receta preexistente,
  normalizador H1 congelado y 8 inputs en el mismo orden. M1
  únicamente podría servir de fuente de velas OHLCV ya cerradas,
  nunca como feature de inferencia.
- Cada **modelo** usa su propio replay contable independiente:
  cuenta simulada 1.000 USD, stake 10 USD, payout +80%/−100%,
  tope 2/h local, stop DD 20% y Gate p85 causal. No existe cuenta
  compartida, cap conjunto ni reordenamiento por señales de otro par.
- El p85 se computa cronológicamente mediante el mismo
  `AdaptiveGate` original: cada resultado entra a su memoria solo
  cuando ha vencido la operación. **No hay inspección previa de
  resultados futuros** para decidir cuál rama optimizada usar.
  El warmup de 500 propuestas comienza en Q3 para todas las
  condiciones, idéntico entre control y evolucionado.

## Reglas de éxito estrictas (predeclaradas, no modificables con Q3)

Se calcula **un veredicto individual por cada uno de los 24 genomas**,
sin elegir un ganador ni reoptimizar. Todos los requisitos se
exigen simultáneamente:

1. **PF Q3 finito > 1,15**, con al menos una operación ejecutada.
   PF indefinido por ausencia de pérdidas, de operaciones o por otras
   causas **no se transforma en infinito aprobatorio**.
2. **DD máximo Q3 < 15%**, cuenta simulada independiente
   inicializada una sola vez al comienzo de Q3, sin stops.
3. **Degradación PF Q3 vs PF H1 < 40%** (estricta):
   `degradacion = 1 - PF_Q3 / PF_H1`, y se exige
   `degradacion < 0.40`. La referencia H1 **no se elige
   mirando Q3**: se reconstruye del PF y PnL positivo
   de Q1 y Q2 del MISMO genoma congelado, usando
   `perdidas_brutas_i = pnl_i/(PF_i - 1)` y
   `PF_H1 = sum(PF_i * perdidas_brutas_i)/sum(perdidas_brutas_i)`.
   Es posible reconstruir pérdidas/ganancias por trimestre porque
   ambos PF son finitos >1 y el PnL individual es positivo.
   Este agregado usa Q1/Q2 de la misma evaluación IS y no mezcla
   divisas, semillas o grupos.

La frecuencia y el número de operaciones se **reportan**, pero
no se añaden umbrales nuevos ni se escogen modelos según Q3.
Toda comparación y tabla queda fijada para los 24 individuos.
El término “alfa” será una interpretación prudente: una evaluación
OOS no elimina el riesgo de selección múltiple en H1, muestras
pequeñas o errores de procedencia.

## Contrato mínimo de datos Q3 antes del replay

Los cuatro archivos requeridos deben ser publicados **antes de la
única ejecución** como Releases privadas con nombres EXACTOS:

- `data-pr44-q3-eurusd-m5-m15-garch-universe-2026q3-v1`
- `data-pr44-q3-gbpusd-m5-m15-garch-universe-2026q3-v1`
- `data-pr44-q3-audusd-m5-m15-garch-universe-2026q3-v1`
- `data-pr44-q3-usdcad-m5-m15-garch-universe-2026q3-v1`

Cada Release debe contener `universe.npz` y `manifest.json`:
`manifest.schema=PR44_Q3_native_M5_M15_GARCH_universe_v1`,
`symbol`, `signal_timeframes/source_timeframes=["M5","M15"]`,
`months=["202607","202608","202609"]`, `start_utc`,
`end_exclusive_utc`, `frozen_h1_input_sha256`,
`source_m5_ohlcv_sha256`, `source_m15_ohlcv_sha256`,
`garch_m5_recipe_sha256`, `frozen_h1_normalizer_sha256`,
`source_zip_sha256` con hashes por mes, `universe_sha256`,
`arrays_sha256` por campo, `months_observed_hours`,
y banderas obligatorias
`raw_m1_signal_features=false`,
`garch_native_causal_refit=true`,
`frozen_h1_normalizer=true`,
`m15_closed_no_lookahead=true` y
`q3_full_source_coverage=true`.

El NPZ lleva los ocho arrays tipados del contrato PR41:
`row_times_ns`, `outcome`, `volatility`, `entry`,
`margin`, `timeframe` (N×8),
`direction_inputs` y `legacy_mask`. **Estos arrays deben
ser generados usando la receta original y datos reales Q3,
no derivarlos de PF, predicciones u otros resultados H1**.
La bitácora del proveedor y la verificación de datos originales
son requisitos independientes de que una manifestación de SHA
sea internamente consistente.

No es correcto poner `q3_full_source_coverage=true` sin haber
verificado los tres meses y la cobertura de barras; tampoco
debe inventarse ninguna SHA. Si falta provenance, el evaluador
aborta con error.

## Implementación y presupuesto

- `scripts/run_pr44_q3_single_pass_audit.py`: único procesador
  económico de los 24 genomas, leyendo cada universo Q3 una sola
  vez por símbolo; sin ramas de entrenamiento, sin retocar modelos.
- `tests/test_pr44_q3_single_pass_audit.py`: pruebas
  exclusivamente sintéticas de política de riesgo, fechas y
  causalidad del gate, sin abrir Q3.
- `.github/workflows/pr44-q3-single-pass-audit.yml`:
  **un solo job**, preflight antes de pip, sin PyTorch/PPO,
  descarga 24 archivos de modelos y **cuatro universos**, ejecuta
  una sola evaluación y exporta `q3-verdict.json`.
- Si los cuatro universos no están disponibles, **no disparar la
  auditoría**, para proteger la cuota de GitHub Actions.
  Este código NO adquiere ni recalibra velas por su cuenta.

**La PR44 queda como protocolo congelado.** Se puede abastecer
por separado el Q3 desde datos históricos reales en un trabajo
de adquisición/procedencia auditable; eso no constituye
entrenamiento genético ni autoriza cambios en los 24 genomas
o los tres criterios OOS.
