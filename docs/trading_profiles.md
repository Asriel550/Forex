# Perfiles de investigación EUR/USD

En GitHub Actions, abrí el workflow **Forex M15 research and Telegram**,
seleccioná la rama `main` y usá `mode=train`. Elegí un perfil y dejá
`timeframe=auto` y `expiry_candles=0` para aplicar sus valores.

| Perfil | Temporalidad | Expiración binaria por defecto | Confianza nominal | Entropía PPO | Coste de omitir un setup | Ventana de tendencia |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| scalping | M5, admite M1 | 1 vela, admite 1 a 3 | 55%, admite 53–55% | 0,03 | 0,20 | 8 barras |
| intraday | M15 | 1 vela | 60% | 0,02 | 0,05 | 8 barras |
| daily | H1 | 2 velas | 63% | 0,02 | 0,05 | 12 barras |
| swing | H4 | 2 velas | 65% | 0,02 | 0,05 | 16 barras |

`trading_profile=auto` elige el perfil de la temporalidad explícita. Con ambos
campos en automático se aplica intraday. El formulario rechaza combinaciones
incompatibles, como scalping con M15. `expiry_candles` acepta un entero y usa
cero como selector del preset. Forex spot conserva su contrato con
`expiry_candles=1` y evalúa posiciones de mercado sin vencimiento binario.
Los inputs de confianza, entropía y coste neutral en `auto` aplican el preset.

## Histórico separado para scalping

Por defecto el runner obtiene el ZIP anual M1 de EUR/USD de 2025 mediante el
formulario oficial de HistData. Comprueba el SHA256 registrado en
`datasets/eurusd/manifest.json` y reutiliza el ZIP verificado desde la caché
de Actions. El año fijo permite repetir el experimento. El script separa el
Parquet M1 o M5, las ausencias observadas y la auditoría de origen. El manifiesto
del modelo identifica el ZIP original y la transformación aplicada.
[Histórico oficial M1 de 2025](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2025)

- Para M1 se usan las velas de un minuto publicadas por HistData, convertidas de EST fijo a UTC.
- Para M5 el camino automático agrega grupos completos de cinco velas M1 reales. Descarta grupos incompletos y registra los huecos. M5 también admite cargar OHLCV nativo de cinco minutos directamente.
- Las variables de Actions `FOREX_M1_DATA_DIR` y `FOREX_M5_DATA_DIR` permiten usar directorios estáticos propios. El cargador usa `config/closures_m1.json` o `config/closures_m5.json` si existe y exige documentar cualquier hueco. Separá las resoluciones por directorio.
- Intraday, daily y swing usan la fuente estática M15 existente. H1/H4 se forman solo con grupos completos de esa fuente.

El archivo verificado produjo 372.024 velas M1 y 73.775 grupos M5 completos.

La ingesta comprueba la resolución declarada, los timestamps, la coherencia
OHLCV y los hashes. M15 carece de los cierres intermedios para construir M1/M5.
El pipeline exige fuentes con granularidad suficiente. Las expiraciones
conservan los límites de los segmentos y de cada partición.

M1 usa una ventana GARCH máxima de 3.072 observaciones y reajusta cada
480 barras, equivalente a la cadencia de 96 barras M5 en minutos observados.
Esta ventana más amplia resolvió dos ajustes consecutivos que no convergían
con la ventana de 2.016 observaciones de la configuración base. En M1/M5, si un
reajuste GARCH aislado falla por falta de convergencia, se
mantienen los últimos parámetros válidos durante un intervalo y se continúa la
recursión causal. El fallo queda registrado en `garch_retained_refits`. El
entrenamiento aborta si falta un ajuste previo o si fallan dos consecutivos.

## Medir la frecuencia

Scalping aumenta la entropía y el coste de omitir setups válidos. La meta es
5–10 contratos por hora observable. Con payout 0,80 el acierto de equilibrio
es 55,56%. El filtro de ventaja conservadora sigue aplicando el margen por
payout y Wilson además del umbral nominal. El agente solo cobra o pierde el
payout simulado al liquidar un contrato.

El reporte `validation_metrics.json` muestra `trades_per_observed_hour`,
`eligible_setups_before_policy`, `trades_per_hour_upper_bound` y
`frequency_target_met`. La frecuencia divide los contratos por las horas
de segmentos operables de toda la validación, incluyendo el tiempo sin operar.
La cota superior incorpora los setups que pasan el filtro y la duración del
contrato. M5 con expiración de tres velas tiene un límite temporal de cuatro
contratos por hora sin solapamiento.

Cada ejecución publica un bundle de investigación con nombre como
`forex-research-bundle-v4-scalping-M1-binary-e1`. El nombre y el manifiesto
distinguen perfil, temporalidad, tipo de mercado y expiración. La ejecución
original del v4 M15 conserva su propio artefacto. La partición de validación
respeta el corte histórico congelado y Telegram selecciona sus bundles por
su contrato de producción.
[Bundle v4 M15 original](https://github.com/Asriel550/Forex/actions/runs/42331180138)


## Prueba con el histórico de 2025

Se completaron ingesta, features y 16.384 pasos PPO con semilla 42, umbral
nominal 55% y expiración de una vela en ambos perfiles de scalping.

| Validación simulada | M1 | M5 |
| --- | ---: | ---: |
| Setups antes de PPO | 344 | 37 |
| Contratos del checkpoint seleccionado | 4 | 8 |
| Contratos por hora observable | 0,0032 | 0,0066 |
| Cota de contratos por hora con el filtro | 0,2786 | 0,0307 |

La meta de 5–10 contratos por hora queda fuera de la capacidad del filtro en
esta muestra. El incentivo de recompensa solo actúa sobre setups elegibles.
M1 registró tres reajustes GARCH aislados retenidos con el estado causal.
El holdout quedó reservado y se conservaron separados los pesos de prueba.
