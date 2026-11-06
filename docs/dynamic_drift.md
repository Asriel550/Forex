# Drift de momentum causal para GBM

Actualización de escala/umbral: [sensibilidad controlada](drift_threshold_sensitivity.md). Esta página conserva la fórmula y los resultados de referencia de PR #6 (escala efectiva 1.0); los nuevos defaults se documentan en la actualización.

La simulación conserva sus 10.000 rutas independientes entre sí, 32 pasos,
PCG64/semilla 42 y common random numbers. Lo que cambia por origen es mu[t],
no las innovaciones ni el recuento de escenarios. La asimetría del precio GBM
ya era lognormal; con drift cero la dirección quedaba cerca de 50%. El 50,45%
observado también refleja el banco finito de shocks, no una ventaja de mercado.

## Estimador y unidades

QuantConfig usa drift_mode=momentum por defecto. causal_drift estima una media
logarítmica por vela usando exclusivamente L=8 retornos conocidos al cierre t.
Los parámetros iniciales son reglas previas, no una optimización del holdout.

```text
r[j]       = log(close[j]/close[j-1]), j=t-L+1..t
r_clipped  = clip(r, -3*sigma[t], +3*sigma[t])
m          = mean(r_clipped)
se         = max(std(r_clipped, ddof=1), sigma[t]) / sqrt(L)
a          = sign(m) * max(abs(m) - 1.96*se, 0)
log_mu[t]  = clip(0.75*a, -0.75*sigma[t], +0.75*sigma[t])
mu[t]      = log_mu[t] + 0.5*sigma[t]^2
```

Se separan correctamente drift aritmético GBM y retorno logarítmico esperado:
S[k] = close[t] * exp((mu[t]-sigma[t]^2/2)*k + sigma[t]*W[k]).
La compensación sigma²/2 evita restarla dos veces a un momentum ya logarítmico.
Sin señal se obtiene mediana plana (log_mu=0). La volatilidad sigue siendo
sigma[t+1|t] del GARCH original, sin usar datos posteriores ni anualizarla.

El recorte de outliers, la contracción y la banda de ruido evitan usar un solo
salto como tendencia. La banda se usa como heurística conservadora; NO se
presenta momentum_z como prueba estadística de rentabilidad ni probabilidad
predictiva calibrada. Con L retornos insuficientes o un hueco distinto de M15
dentro de ellos, se reinicia log_mu a cero hasta reunir ocho retornos contiguos.
Los NaN del warmup GARCH se conservan, sin rellenarlos con datos futuros.

## Probabilidades, percentiles y tensor

project_monte_carlo usa la MISMA mu para:

- P05/P95 terminales a 32 barras y dist_p05/dist_p95 con signo.
- mc_prob_up/down/tie, contadas en el PRIMER paso de esas mismas rutas.
- mc_band_position = (spot-P05)/(P95-P05), sin recortarla a [0,1].
- mc_band_bias: +1 si spot está debajo de P05, -1 si está encima de P95, 0 dentro.

Ambas bandas pueden quedar arriba o abajo del spot. Esa geometría refleja el
sesgo del escenario a 32 velas; NO se convierte un cruce de banda a ocho horas
en una probabilidad de acierto binaria a quince minutos. El filtro de 70% de
PR #5 sigue usando únicamente la probabilidad direccional del primer paso y
su control de edge conservador. No se reduce ese umbral ni se modifica payout.

Las distancias y probabilidades existentes ya forman parte del tensor. Se
mantienen sus 32 dimensiones; se necesitan un normalizador y PPO entrenados
sobre las nuevas features para aprender su distribución. Las columnas mc_drift,
mc_log_drift, momentum_mean, momentum_se, momentum_z y las dos métricas de
geometría se guardan para auditoría; no se añaden a escondidas al tensor.

## Continuación de inferencia y compatibilidad

Telegram usa causal_drift y project_monte_carlo, los mismos helpers que el batch.
En cada vela nueva utiliza su ventana de cierres, timestamps y sigma causal.
La fase de refit GARCH, el horizonte, número de rutas y semilla siguen congelados
en el bundle. Si falta historia necesaria para el momentum, se rechaza el
baseline incompleto; no se inventa una señal ni se cambia su ventana.

Los nuevos parámetros quedan en quant_config del Parquet y del manifiesto v3.
QuantConfig.from_metadata conserva el significado de bundles previos: si no
declaraban drift_mode, se interpretan como constant con su drift_per_bar original.
No se introduce momentum silenciosamente en pesos entrenados con drift cero.
El formato v3 permanece porque no cambia la forma de las observaciones.

Si se reconstruyen probabilidades ausentes en datos de momentum, ensure_probabilities
exige mc_drift archivado. No usa un cero por defecto ni reestima un momentum
con una ventana que pudo truncarse tras el warmup.

## Ejecución

```bash
python -m src.quant_engine --drop-warmup --drift-mode momentum --drift-window 8 --drift-shrinkage 0.75 --drift-noise-z 1.96 --drift-cap-sigma 0.75 --drift-clip-sigma 3
python -m src.train --total-timesteps 100000
python -m src.backtest
```

El workflow ya llama al módulo y toma estos nuevos defaults tras fusionar el
cambio. Hay que ejecutar mode=train para generar features, normalizador y pesos
con momentum. Reproducción del motor antiguo:

```bash
python -m src.quant_engine --drop-warmup --drift-mode constant --drift-per-bar 0
```

Un drift constante no nulo requiere --drift-mode constant explícito. Los
parámetros no se cambian desde la ruta de alertas.

## Verificación y límites de interpretación

Las pruebas comparan las probabilidades y percentiles contra las matrices reales
10.000 x 32, verifican prefijos ante cambios futuros, equivalencia batch/incremental,
huecos, warmup, ruido alternante, outliers, conversión de unidades y metadatos
antiguos. Tendencias sintéticas controladas producen probabilidades direccionales
aproximadamente entre 55% y 75%, y las más fuertes pueden pasar el filtro del 70%.
Estos escenarios verifican capacidad matemática; no son resultados de mercado.

En el snapshot real se recalcularon 135.194 filas y 1.409 ajustes GARCH. El
diagnóstico se limitó a las 94.624 filas de entrenamiento, sin seleccionar
parámetros por resultados futuros: 202 filas superaron 55%; máximo 65,08%;
ninguna superó 70%. No se rebajó la banda de ruido para fabricar operaciones.
Con estos defaults no se debe prometer que desaparezca la abstención del agente.

Un porcentaje alto bajo GBM con momentum no prueba una ventaja estadística real.
La persistencia futura del momentum y su extrapolación constante a 32 barras
son hipótesis. La incertidumbre del drift y los cambios de régimen no quedan
resueltos por generar más rutas. Para validar/calibrar las probabilidades hace
falta medir Brier/log-loss, calibración y expectativa de payout en validación
cronológica/walk-forward, y confirmar en datos futuros no utilizados. El holdout
antiguo ya se examinó; no se usa para afinar estos parámetros ni se presenta
como una nueva prueba independiente. No se garantiza un rango de confianza ni
rentabilidad, y no se añaden precios de mercado actuales al snapshot estático.
