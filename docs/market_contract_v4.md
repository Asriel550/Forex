# PR #10: contratos de mercado v4 y entrenamiento por oportunidades

## Qué cambia en Actions

En la rama por defecto, Run workflow → mode=train permite:

| Input | Opciones | Default |
| --- | --- | --- |
| timeframe | M5 / M15 / H1 / H4 | M15 |
| trading_mode | binary / forex_spot | binary |
| expiry_candles | entero 1..32 | 1 |
| neutral_opportunity_penalty | 0 / 0.005 / 0.01 / 0.05 / 0.10 | 0.05 |

GitHub Actions ofrece type:number, no un tipo integer. El código exige entero y
rango antes de ingerir datos; no redondea 1.5 ni sustituye 0 por 1. Forex spot
requiere expiry_candles=1 como marcador: sus posiciones no tienen expiración.
Los parámetros se pasan por variables de entorno entrecomilladas, sin interpolar
expresiones de usuario directamente en comandos de shell.

Ejemplos: binary/M15/2 vence a 30 minutos; binary/H1/2 a 2 horas. Las bandas MC
siguen siendo a 32 barras; ahora las probabilidades direccionales se extraen de
la columna de las 10.000 rutas correspondiente a expiry_candles. No se utiliza
una probabilidad a 1 barra para filtrar un contrato a 2 barras. La volatilidad
GARCH y las ventanas de momentum se expresan en barras del timeframe elegido.
32 barras H4 no equivalen a 8 horas; se conservan 32 barras de negociación.

## Datos: límite que no se puede resolver con resampling

El snapshot incluido es M15. M15, H1 y H4 funcionan con esos archivos. **M5
requiere archivos estáticos M1: no se puede obtener OHLC M5 genuino de OHLC M15.**
La opción M5 está implementada para M1; con el dataset actual falla de forma
explícita antes del cálculo cuantitativo. No se inventan velas ni se añade una
API en vivo para aparentar soporte. M1 usa nombres HistData o input_timeframe=M1. La variable de repositorio
FOREX_MARKET_DATA_DIR permite apuntar train a un directorio estático M1 separado;
no mezclar esos archivos con el snapshot M15.
Una fuente M1 nueva también requiere sus exclusiones observadas de huecos;
las exclusiones M15 existentes no certifican automáticamente otra fuente.

El cargador combina y deduplica los CSV/ZIP primero, verifica la fuente en su
resolución original contra el calendario y las exclusiones documentadas, y
agrega en UTC con límites de medianoche (epoch): open primero, high máximo,
low mínimo, close último, volume suma. Solo conserva grupos completos y
cerrados. Los grupos parciales se descartan y cuentan en dropped_partial_bars;
no se convierten en velas más cortas ni se rellenan con precios anteriores.

La firma de caché separa resolución, configuración y archivos por SHA256. Los
nuevos inputs se guardan en data/eurusd_market.parquet y features en
 data/eurusd_market_features.parquet dentro del workflow. El entorno y el motor
comprueban el timeframe y el horizonte archivados, además de timestamps.

Comprobación sobre el snapshot: 31.665 barras H1 y 6.239 barras H4 completas.
No son señales ni resultados de rentabilidad. La pérdida de grupos incompletos
crea huecos explícitos; no se abren contratos que crucen esos huecos.

## BinaryOptionsEnv

0=PUT, 1=NEUTRO, 2=CALL. La decisión usa el cierre conocido de t; un contrato
liquida en t+expiry_candles. Un contrato aceptado salta exactamente N barras;
NEUTRO y solicitudes bloqueadas esperan una barra. No hay contratos solapados,
no se evalúan filas fuera del segmento o split, ni expiraciones sobre fines de
semana/huecos. Un episodio termina si no queda una ventana completa para N.

El timestamp fuente es apertura UTC; decisión=timestamp[t]+duración de barra,
liquidación=timestamp[t+N]+duración. El normalizador usa exclusivamente train.
Payout sigue siendo 0.80 por defecto (beneficio neto del stake); empate se
reembolsa por defecto. Las penalizaciones afectan reward, nunca el saldo.

La nueva recompensa mantiene PnL normalizado por riesgo, penalización de
actividad, incremento de drawdown y stop del 20%. Omitir un setup que pasa
confianza y edge conservador por payout cuesta 0.05 por defecto, frente a 0.005
antes. Se determina antes de leer el resultado futuro y cuesta lo mismo si
luego gana o pierde. Esperar sin setup sigue siendo gratuito. En v4 una solicitud
bloqueada donde ninguna dirección pasa el filtro tampoco cuesta reward; pedir
la dirección contraria a un setup válido conserva blocked_penalty. Esto evita
penalizar al agente por no ejecutar un contrato que el propio filtro impide.
El score de selección es la recompensa acumulada por setup en la exposición,
no por decenas de miles de esperas. Esa recompensa ya incluye drawdown
incremental, costes y actividad; no vuelve a restar 10×maxDD, que duplicaba el
castigo y favorecía la política inicial sin contratos. Una política que alcanza
el stop recibe un score fuertemente negativo. No usa resultados del holdout.

## Abordar la escasez de setups durante el aprendizaje

La penalización ya existía. Subirla por sí sola no resuelve que menos del 0,2%
de las filas habiliten un contrato. El entrenamiento v4 binario usa
OpportunityTraining: empieza episodios en setups elegibles de TRAIN y omite
solo esperas gratuitas sin setup. Esas filas se saltan mediante índices; saldo,
decaimiento de actividad, límites de episodio y estado posterior equivalen a
esperar explícitamente. No filtra por resultados futuros ni por validación.

PPO sigue pudiendo elegir las tres acciones; no se fuerzan CALL/PUT, no se
aplican máscaras que eliminen NEUTRO y no se paga una ganancia ficticia por
operar. Un NEUTRO ante un setup válido siempre devuelve su coste configurado.
Si train no tiene setups elegibles, el entrenamiento falla sin bajar umbrales.

El descuento gamma se aplica por decisión elegible (reloj de oportunidades),
no por minuto. Esto es un cambio de objetivo explícito y se archiva en el
manifiesto. Validación usa TODAS las barras cronológicas sin este wrapper, con
sus filtros, saldo, huecos y stop. --no-opportunity-training permite una ablación
con reloj de barras; no se selecciona por el holdout. El wrapper solo se aplica
si el filtro de riesgo está activo.

El muestreo repite los pocos setups de train: no crea observaciones económicas
independientes. Más operaciones o mayor captura no prueban una ventaja. Un
setup aprobado es una hipótesis del GBM no calibrada, no una oportunidad rentable
confirmada. No se promete eliminar definitivamente el colapso con una recompensa.

## ForexSpotEnv

0=SHORT, 1=FLAT/cerrar, 2=LONG, como posición objetivo. Repetir LONG/SHORT mantiene
la posición sin volver a pagar spread. Revertir cierra la anterior y abre la
nueva. Lote fijo 0.1 (10.000 EUR), cuenta USD, leverage 1:100, spread 1.5 pips.
Se asigna el coste completo de ida/vuelta a la apertura. Se marca equity en cada
cierre; la recompensa usa su variación ajustada por volatilidad, no +0.80/−1.
Las observaciones incluyen posición, PnL flotante en pips, utilización de margen,
spread y actividad, en lugar de fingir el estado de un contrato binario.

El filtro spot usa confianza direccional, no edge por payout. Una apertura sin
margen suficiente se rechaza; si la equity de una posición abierta cae bajo el
margen requerido se liquida y termina. También rige el límite de drawdown.
Fin de segmento/episodio liquida posiciones antes de un hueco. PF/acierto usan
operaciones CERRADAS y Sharpe/equity incluyen mark-to-market; CSV separa ambos.
No se modelan financiación overnight, slippage, liquidación intrabar o ejecución
del broker. Un salto entre cierres puede exceder el stop e incluso dejar deuda;
no se supone protección de saldo negativo. Es simulación de investigación, no trading real conectado.

## Pesos, reportes y compatibilidad

Nuevos modelos: manifest_version=4, market_config, reward_version,
observation_features, parámetros de riesgo/spot, reloj de entrenamiento,
selected_checkpoint_timesteps/selected_checkpoint_trained y
SHA256 de datos/modelo/normalizador. No se cargan pesos previos para warm-start.

El workflow publica forex-research-bundle-v4, separado de forex-alert-bundle.
**No promueve estos modelos ni cambia el bot Telegram activo v3.** La inferencia
v3 no admite pesos v4; su selección ignora el artefacto de investigación. La
promoción y una inferencia multi-timeframe requieren otra revisión explícita,
con datos recientes. La prueba telegram_test y sus secretos siguen funcionando.

Los artefactos de reportes incluyen manifiesto, auditoría estocástica de PPO,
historial de validación y validation_metrics.json / validation_decisions.csv /
validation_trades.csv / validation.md. El modo diagnostics conserva el protocolo
PR #8 (M15/binary/1) para mantener comparables sus barridos y bundle archivado;
rechaza otro contrato en vez de ignorar los inputs. Para probar v4 usar train.

La frontera histórica protegida de PR #9 queda fijada en
2026-04-07T15:15:00Z (apertura UTC). Si el resampling o nuevos datos desplazan
el percentil 90 hacia adelante, train/validación se vuelven a partir SOLO sobre
el prefijo anterior a esa frontera, conservando su proporción 70:20 y la purga.
Se excluye también cualquier barra gruesa cuyo cierre la atraviese. No se
recicla inadvertidamente el antiguo holdout al seleccionar H1/H4.

El holdout histórico continúa congelado: ningún paso del workflow invoca backtest.
Los tests de backtesting existentes son sintéticos. Las APIs Python antiguas sin
market conservan el contrato v3 para auditoría de modelos previos; el CLI y
mode=train usan el contrato v4 explícito. No se reinterpretan los pesos v3 con
los nuevos defaults.

## Comprobación de aprendizaje

[Ablación corta de 16.384 pasos](research/v4_opportunity_ablation.md): en una
semilla, el checkpoint seleccionado por oportunidades ejecutó 27 de 48 setups
en validación, frente a 7 con reloj de barras. Ambos tuvieron entrenamiento
efectivo. El JSON conserva hashes y métricas; no es una garantía de solución
universal ni se interpreta el resultado como rentabilidad confirmada.
