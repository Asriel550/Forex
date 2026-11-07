# Perfiles de investigación EUR/USD

En GitHub Actions, abrí el workflow **Forex M15 research and Telegram**,
seleccioná la rama `main` y usá `mode=train`. Elegí un perfil y dejá
`timeframe=auto` y `expiry_candles=0` para aplicar sus valores.

| Perfil | Temporalidad | Expiración binaria por defecto | Confianza nominal | Entropía PPO | Coste de omitir un setup | Ventana de tendencia |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| scalping | M5, admite M1 | 1 vela, admite 1 a 3 | M5 binario 57%. M1 y spot 54%, admiten 52–54% | M5 binario 0,012. M1 y spot 0,03 | 0,20 | 4 barras M5 / 6 barras M1 |
| intraday | M15 | 1 vela | 60% | 0,02 | 0,05 | 8 barras |
| daily | H1 | 2 velas | 63% | 0,02 | 0,05 | 12 barras |
| swing | H4 | 2 velas | 65% | 0,02 | 0,05 | 16 barras |

`trading_profile=auto` elige el perfil de la temporalidad explícita. Con ambos
campos en automático se aplica intraday. El formulario rechaza combinaciones
incompatibles, como scalping con M15. `expiry_candles` acepta un entero y usa
cero como selector del preset. Forex spot conserva su contrato con
`expiry_candles=1` y evalúa posiciones de mercado sin vencimiento binario.
Los inputs de confianza, escala de momentum, entropía y coste neutral en `auto` aplican el preset.

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

El objetivo de frecuencia ejecutada es de 1,5 a 3 contratos por hora
observable para M5 binario y de 5 a 10 para M1 binario. El momentum usa 6
barras en M1 y 4 en M5. Con `drift_scale=auto`, M1 usa 1,8 fijo. M5 binario
interpola entre 1,4 y 2,0
según la relación entre la media de retornos recientes y su error estándar.
Usa el mínimo hasta una relación de 1 y el máximo desde 2,5. Ambos valores se
calculan con velas cerradas, antes de proyectar. En M5 binario se puede
seleccionar `auto` o `2.0` para el techo. P05/P95 y las
probabilidades direccionales se obtienen de la misma simulación de 10.000
trayectorias. Intraday, daily y swing conservan sus presets.

Con payout 0,80, el acierto de equilibrio es 55,56%. M5 binario exige
probabilidad direccional superior al 57% nominal y aplica Wilson con
`z=2.24`. La ventaja esperada calculada con ese límite debe superar el 1%
de la apuesta, lo que requiere un límite inferior mayor al 56,11% con ese
payout. M1 conserva
`z=1.96` y margen adicional cero. Wilson mide el error de muestreo de Monte
Carlo bajo el GBM. La validación mide por separado el acierto observado.

Los logs de validación y los reportes JSON y Markdown separan tres etapas.
Se cuenta la dirección MC en cada ventana con entrada y vencimiento válidos,
las propuestas que superan la confianza nominal y las que superan el filtro
de Wilson y payout. Los contratos ejecutados por PPO se muestran aparte.
Las tasas usan las mismas horas de toda la validación en segmentos operables,
incluso si el stop termina la ejecución de la política antes del final.
Las propuestas son oportunidades por vela. La cota de contratos por hora
incorpora también el vencimiento sin solapamiento.

## Alineación y gate de Scalping M5 binario

En el formulario, elegí `mode=train`, `trading_profile=scalping`,
`timeframe=M5`, `trading_mode=binary` y `expiry_candles=1` para reproducir
la prueba de una vela. Usá `auto` para confianza, drift y entropía. El preset
fija el umbral nominal en 57% y la entropía en 0,012. Se acepta un
`ent_coef` explícito de 0,01 a 0,012.

Una solicitud contraria a un setup que pasa los filtros recibe −1,25 en
la recompensa de PPO. Una operación alineada ganadora recibe una base de
+1,75 antes de aplicar costes de actividad, penalizaciones por drawdown
y stop. El saldo conserva la liquidación por payout y el filtro mantiene
bloqueada la dirección contraria.

El gate exige que el checkpoint entrenado cumpla todos los objetivos.

| Métrica de validación | Requisito |
| --- | ---: |
| Acierto sin reembolsos | ≥ 56% |
| Profit Factor | > 1,20 |
| Contratos ejecutados por hora observable | 1,5–3,0 |
| Contratos ejecutados sobre todos los setups elegibles | ≥ 30% |

La selección prioriza los checkpoints entrenados que pasan el gate y
desempata con el score habitual. Si ninguno pasa, Actions conserva los
reportes JSON, Markdown y CSV y marca el job como fallido antes de publicar
el bundle M5. El manifiesto registra `validation_gate_passed`. La conversión
usa todos los setups de validación, incluidos los posteriores a un stop,
y el reporte desglosa coincidencias, solicitudes contrarias y abstenciones
entre los setups que PPO llegó a evaluar.

Los bundles de investigación tienen nombres como
`forex-research-bundle-v4-scalping-M1-binary-e1`. El nombre y el manifiesto
distinguen perfil, temporalidad, tipo de mercado y expiración. La ejecución
original del v4 M15 conserva su propio artefacto. La partición de validación
respeta el corte histórico congelado y Telegram selecciona sus bundles por
su contrato de producción.
[Bundle v4 M15 original](https://github.com/Asriel550/Forex/actions/runs/42331180138)


## Referencia de la PR #11 con el histórico de 2025

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


## Referencia de sensibilidad de la PR #12

Dos corridas con el histórico de 2025, semilla 42, 16.384 pasos PPO, umbral
nominal 54% y una vela de expiración. La evaluación se limitó a validación y el
holdout quedó reservado.

| Métrica de validación | M1 | M5 |
| --- | ---: | ---: |
| Direcciones MC por hora antes del filtro | 59,99 | 12,00 |
| Propuestas por hora tras confianza nominal | 25,39 | 5,31 |
| Setups por hora tras Wilson y payout | 21,55 | 4,76 |
| Setups elegibles en toda la fase | 26.606 | 5.740 |
| Contratos ejecutados por PPO por hora | 0,290 | 0,222 |
| Acierto sin reembolsos | 53,47% | 51,15% |
| PnL simulado | −US$130 | −US$206 |
| Drawdown máximo | 20,04% | 20,60% |

En M5 la cota aumentó de 0,031 a 4,76 setups por hora sobre la misma fase. En
ambas corridas el stop de drawdown se activó y el acierto quedó por debajo del
55,56% necesario con payout 0,80. El ajuste amplió el conjunto de candidatos
para investigación. La frecuencia de ejecución de 5–10 por hora sigue pendiente.

## Comparación de drift de la PR #13

Se reutilizaron las volatilidades GARCH causales del histórico M5 de 2025
para aislar el efecto del drift. Con la misma validación y payout 0,80,
se evaluó la dirección de mayor probabilidad en cada setup elegible.

| Filtro y drift | Setups | Setups/h | Acierto resuelto del motor | Profit Factor del motor |
| --- | ---: | ---: | ---: | ---: |
| PR #12, 54%, fijo 2,2 | 5.740 | 4,76 | 48,46% | 0,752 |
| PR #13, 58%, fijo 2,2 | 5.315 | 4,41 | 48,28% | 0,747 |
| PR #13, 58%, adaptativo 1,4–2,2 | 4.112 | 3,41 | 47,91% | 0,736 |

El drift adaptativo reduce el número de setups en un 22,6% frente a la
escala fija con el mismo filtro. En esta validación, el acierto del motor
también baja. Las dos configuraciones de la PR #13 admitieron los mismos
candidatos después del umbral nominal y del filtro económico. El holdout
quedó reservado.

Con semilla 42 y 100.000 pasos solicitados por corrida, se seleccionaron
los checkpoints con el criterio habitual de validación ajustado por riesgo.
Ambas corridas usaron 58%, el mismo Wilson y el mismo presupuesto PPO.

| Checkpoint seleccionado en validación | Fijo 2,2 | Adaptativo 1,4–2,2 |
| --- | ---: | ---: |
| Paso | 75.000 | 50.000 |
| Contratos | 53 | 63 |
| Contratos/h | 0,044 | 0,052 |
| Acierto sin reembolsos | 58,49% | 55,74% |
| Profit Factor | 1,127 | 1,007 |
| PnL simulado | +US$28 | +US$2 |
| Drawdown máximo | 6,20% | 6,79% |
| Objetivos cumplidos | 1 de 3 | 0 de 3 |

A payout 0,80 y apuesta fija, PF superior a 1,2 requiere más de 60% de
acierto en los contratos resueltos. El drift adaptativo reduce propuestas
y deja una cota de 3,41 setups/h. La frecuencia efectivamente ejecutada
por PPO y la calidad conjunta siguen por debajo de los objetivos.

## Prueba de alineación de la PR #14

Se usó la fuente M5 de la corrida de la PR #13 y se reproyectó el drift
causal con techo de 2,0. Con semilla 42, 100.000 pasos solicitados y
vencimiento de una vela, el candidato diagnóstico seleccionado tuvo 219
contratos, 50,94% de acierto sin reembolsos, PF 0,831 y PnL de −US$176.
El stop de drawdown del 20% terminó la simulación. Hasta ese momento PPO
coincidió en 219 de 275 setups, se abstuvo en 56 y pidió cero direcciones
contrarias. Sobre la fase completa fueron 0,182 contratos por hora y
4,94% de conversión sobre 4.429 setups. El gate quedó rechazado y el
holdout reservado.

## Evolución genética de Scalping M5

En **Run workflow**, seleccioná `main` y `mode=evolution`. Los valores
`trading_profile=auto`, `timeframe=auto`, `trading_mode=binary` y
`expiry_candles=0` resuelven Scalping M5 binario con una vela de vencimiento.
Elegí `population_size` entre 4 y 64 y `generations` entre 1 y 20. Los
valores por defecto son 24 individuos y 10 ciclos posteriores a la
generación inicial, para un total de 11 generaciones evaluadas.

Cada individuo tiene un vector de pesos y un sesgo sobre features de
mercado normalizadas. Una puntuación positiva acepta la dirección de
Monte Carlo si supera confianza, Wilson y payout, y coincide con la
dirección H1. Cada H1 se forma con 12 velas M5 completas. La dirección es
el signo de EMA8 menos EMA24 del cierre H1 fuera de una banda neutral del
10% de la media exponencial de 14 rangos H1. Las horas incompletas se
descartan. El filtro requiere el warmup completo y un último cierre H1
de menos de 60 minutos. El sistema solo ejecuta CALL con H1 alcista y
PUT con H1 bajista.

Los cierres posteriores se usan para liquidar, con las mismas reglas de
caja, huecos y reembolsos del entorno M5. El stop del 20% se aplica al
liquidar y el gate exige un drawdown menor al 15%. Las horas incluyen
el tramo observable completo.

Cada individuo recibe un puntaje finito. Sean `w` el acierto sin
reembolsos, `f` los contratos por hora y `d` el drawdown como fracción.
La aptitud suma `w` y el término de PF, ponderados por `min(1, f/1.5)`
para dar menos peso a resultados de baja frecuencia. Se resta la distancia
de `f` al rango [1,5–3,0], dividida por 1,5, el drawdown `d` y la
penalización suave `0.5 × max(0, 0.56 − w)`. El término de PF es
`PF/(1+PF)`, vale uno si hay ganancias sin pérdidas y cero si el ratio
es indefinido. Un acierto indefinido aporta cero al cálculo.

El normalizador se ajusta en train. El primer 75% de validación
cronológica se divide en cuatro bloques consecutivos de tamaño similar.
Cada bloque empieza con saldo propio y liquida sus contratos dentro de
sus límites. El reporte muestra la distribución H1 observada en cada
bloque, con estados alcista, bajista, lateral y sin cotización vigente.
La aptitud combina 50% del puntaje medio y 50% del peor bloque. Resta
0,10 veces la proporción de pesos efectivos, medida como
`(sum(abs(w))**2 / sum(w**2)) / número_de_pesos`. El cálculo excluye el
sesgo y reescalar todos los pesos mantiene la misma penalización.

El grupo de padres conserva el tamaño del 20% superior redondeado
hacia arriba. Da prioridad a quienes terminan en positivo con drawdown
menor al 15% en los cuatro bloques. Solo esos individuos pasan intactos
como élites. Las plazas restantes permiten que candidatos con menor
consistencia aporten genes a descendientes mutados. El CSV distingue
padres, élites y resultados de cada bloque.

La evaluación de búsqueda completa transporta el saldo entre bloques
para medir el riesgo de todo el tramo. Tras una vela de purga, se evalúa
una sola vez al campeón preseleccionado en confirmación. El gate final
exige consistencia en los cuatro bloques y, en búsqueda completa y
confirmación, acierto sin reembolsos ≥56%, PF >1,20, 1,5–3,0 contratos por
hora y drawdown <15%. El holdout histórico queda reservado.

Actions publica el JSON, el CSV por individuo y el Markdown bajo
`forex-reports-*`. El artefacto `forex-evolution-candidates-*` contiene
un ZIP independiente por generación, desde `bundle_scalping_gen_0.zip`.
Cada ZIP contiene pesos, normalizador y manifiesto con SHA256. Se publica
también el representante de diagnóstico de una generación rechazada.
El gate puede marcar la corrida como fallida y conservar los reportes y
candidatos. Los artefactos de evolución quedan separados de Intradía M15
y del flujo de Telegram.

### Evaluación de consistencia y tendencia H1

La prueba local conservó los 40 individuos, 10 ciclos, semilla 42 e
histórico M5 de 2025 de la PR #16. Hubo ocho padres en cada generación y
cero élites. El campeón de la generación 10 perdió en los cuatro bloques
(−US$170, −US$200, −US$188 y −US$90). Su genoma usó 14,68 pesos efectivos
de 25. El filtro H1 dejó 1.543 de 3.311 setups en búsqueda y 491 de 1.118
en confirmación. Sus tasas máximas fueron 1,706 y 1,630 contratos por hora.

| Métrica del campeón | Búsqueda | Confirmación |
| --- | ---: | ---: |
| Contratos ejecutados | 303 | 239 |
| Acierto sin reembolsos | 52,36% | 50,87% |
| Profit Factor | 0,879 | 0,828 |
| Contratos por hora | 0,335 | 0,794 |
| Drawdown máximo | 20,80% | 20,51% |
| PnL simulado | −US$170 | −US$194 |

En una comparación posterior con el genoma fijo, el filtro elevó el
PF en confirmación de 0,500 a 0,828 y el drawdown pasó de 20,00% a
20,51%. Ambas variantes activaron el stop. El gate rechazó al campeón.
Se reutilizó la confirmación histórica para el diagnóstico.
[Corrida de referencia de la PR #16](https://github.com/Asriel550/Forex/actions/runs/42409133378)
