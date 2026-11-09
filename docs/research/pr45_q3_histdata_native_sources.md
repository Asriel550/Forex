# PR45 — Datos oficiales HistData Q3, continuidad causal H1→Q3

La PR44 requiere cuatro datasets de 2026-07-01 a 2026-09-30:
EURUSD_M5_Q3_2026.parquet, GBPUSD_M5_Q3_2026.parquet,
AUDUSD_M5_Q3_2026.parquet y USDCAD_M5_Q3_2026.parquet.

## HistData M1 como fuente oficial, no como indicador

HistData sirve ZIP Generic ASCII M1; los Parquet M5 se derivan
mediante agrupación a OHLCV **M5/M15 completamente cerrados**
sin interpolar, normalizar precios, rellenar huecos ni enviar
inputs M1 a los modelos. Cada fuente conserva URL, mes y SHA256
y también se archiva en su Release definitiva.

Para GBPUSD, AUDUSD, USDCAD se reutilizan las releases H1
de OHLCV M5/M15 de PR40, y se descargan los ZIP HistData
de junio–septiembre. El ZIP junio aporta el 1 de julio
00:00–04:59 UTC, ya que HistData interpreta EST fijo UTC−05.
Se verifica un solapamiento de junio en precios, volúmenes y fechas
entre H1 original y el proveedor antes de concatenar H1 y Q3.
Si difieren, la preparación aborta.

Para EURUSD, sin Release H1 OHLCV M5, se adquieren enero–septiembre
y se reconstruye el histórico continuo mediante la misma receta
M5/M15 PR40. La volatilidad de las últimas 48 propuestas H1
proviene exclusivamente del universo original PR41 firmado.

## Features causales y estado al cierre de junio

Se ejecuta el código original del proyecto:
enrich_features, causal_garch_volatility, timeframe_features,
m15_closed_context y los normalizadores Gen375 congelados.
Los GARCH por activo se calculan sobre toda la historia conocida
H1→Q3 antes de extraer filas Q3. M15 solo se utiliza después
de su cierre efectivo; la etiqueta de outcome se liquida después
de la decisión y no participa como predictor.

El filtro GARCH relativo de los 22 genes de PR41 emplea
una mediana de volatilidad sobre 48 propuestas anteriores.
El auditor PR44 hereda exactamente los últimos 48 sigma Q2
del universo H1 firmado del MISMO símbolo, guardados y
autenticados en el manifiesto Q3. La cuenta/riesgo/p85 Q3
se reinician conforme al protocolo OOS, sin trasladar PnL
de Q2. Ningún genoma de PR42/43 participa en la preparación.

## Entregables y disparo

Cuatro Releases con etiquetas data-pr44-q3-{par}-m5-m15-garch-universe-2026q3-v1
contienen el Parquet solicitado, universe.npz, manifest.json,
source-manifest.json y ZIP de proveedor para inspección independiente.

El Parquet contiene M5 open/close UTC, OHLCV bid, GARCH causal y
las ocho features M5/M15. El universo NPZ proporciona las
ocho matrices idénticas al contrato PR41: row_times_ns, outcome,
volatility, entry, margin, timeframe, direction_inputs, legacy_mask.
El manifest guarda SHA256 de la fuente, arrays, Q2 warmup y
normalizador congelado, además de conteo por mes y horas.

Workflow: .github/workflows/pr45-q3-native-data.yml. Solo
prepara datasets; **NO invoca la evaluación OOS PR44**,
NO calcula PF, NO elige genomas ni reentrena. Para hacer
el juicio final, deberán existir y validarse las cuatro
Releases antes de iniciar el workflow PR44 de pasada única.

Los meses faltantes, solapes incompatibles y gaps de barras
no se inventan ni se cubren con precios artificiales.

## Recuperación numérica GBPUSD pre-OOS

En la primera adquisición, EURUSD, AUDUSD y USDCAD fueron publicados sin cambios. La receta GARCH sobre H1 completo para GBPUSD encontró dos refits consecutivos sin convergencia **en el propio historial H1**, antes de evaluar ningún genoma Q3. La política GARCH de retener solo un refit fallido permanece intocable. Para preparar GBPUSD se reintenta con exactamente la misma rutina y configuración GARCH causal, pero un warmup fijo del último mes Q2 (desde 2026-06-01) antes de la primera barra Q3. Esta diferencia de historial debe quedar registrada explícitamente en su manifest mediante garch_history_start_utc y garch_q2_30day_fallback_used; los otros tres pares conservan su serie completa H1. Esta recuperación no usa ni consulta PF, drawdown, resultados PR44 ni los 24 genomas. Si el nuevo ajuste tampoco converge, no se publica GBPUSD.

No se sustituyen las tres Releases certificadas ni se ejecuta automáticamente el juicio Q3. La diferencia de calentamiento debe interpretarse como limitación metodológica en la comparación final.
