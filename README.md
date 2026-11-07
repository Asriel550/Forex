Holdout congelado: [diagnóstico MC/PPO, barrido de semillas y prueba Telegram](docs/train_validation_diagnostics.md). Investigación solo manual; `mode=diagnostics` no publica pesos. [Resultados MC/PPO y tres semillas](docs/research/pr8_filter_policy_diagnosis.md).

Exploración PPO configurable: [entropía, incentivos y auditoría](docs/ppo_exploration.md). Default `ent_coef=0.02`; [comparación local con resultados y límites](docs/research/ppo_exploration_comparison.md).

# Forex EUR/USD M15 — research and Telegram signals

Escala/umbral controlados: [sensibilidad 58–65% y payout](docs/drift_threshold_sensitivity.md). Nuevos defaults: escala 1.5, umbral 60%.

Drift de momentum causal: [fórmula, parámetros, compatibilidad y límites](docs/dynamic_drift.md).

Cambios v3 de inacción/riesgo: [diseño, uso y límites](docs/risk_filter.md). Los bundles v2 requieren regenerar features y reentrenar.

Sistema de investigación en Python para opciones binarias de una vela M15:
archivos históricos CSV/ZIP, GARCH(1,1), Monte Carlo GBM, Gymnasium, PPO, holdout y
alertas Telegram. No coloca órdenes ni utiliza MetaTrader, Windows o Yahoo.

## Ejecución local

Python 3.11/3.12; se valida con Python 3.12 y Torch CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.data_loader --input-dir datasets/eurusd
python -m src.quant_engine --drop-warmup
python -m src.train --total-timesteps 100000
python -m src.backtest
```

## Contrato de simulación

0=PUT, 1=NEUTRO, 2=CALL. La decisión usa información conocida al cierre de t,
con entrada de referencia en close[t] y expiración en close[t+1], quince minutos
después. Un acierto paga +0.8 stakes netos; una pérdida resta un stake. Empates
se reembolsan por defecto y se pueden configurar como pérdida. No hay contratos
que crucen huecos de datos. El timestamp del Parquet es apertura de vela UTC.

El resultado es una simulación al cierre, sin modelar latencia o cotización
contractual real de un broker. Los percentiles MC son terminales a 32 barras;
no son niveles garantizados de soporte/resistencia ni una probabilidad de acierto.

## Causalidad y evaluación

GARCH se ajusta a ventanas históricas con refits periódicos; entre refits se
actualiza su recursión. MC usa 10,000 rutas de 32 pasos y common random numbers
para obtener percentiles empíricos sin un tensor gigante por cada vela.
Las observaciones combinan RSI/ATR, spread opcional, retornos, features cuantitativas
y estado simulado de cuenta.

Tras el warmup se separa cronológicamente 70% train, 20% validación y 10% test,
con una barra adicional de purga en los límites. Solo train ajusta el normalizador
y pesos; validación selecciona el checkpoint por score de retorno, drawdown e intentos bloqueados. El holdout se evalúa después,
sin seleccionar hiperparámetros con él. SHA256 vincula datos, pesos y normalizador.
Los hashes son checks de consistencia, no una firma ni garantía sobre datos externos.

Reportes: reports/metrics.json, trades.csv, signals.csv, equity.csv,
equity_drawdown.png y signals.png. Los ratios indefinidos se representan con null
y un estado explícito. Sharpe incluye barras inactivas y usa la convención 252*96.

## GitHub Actions

El workflow .github/workflows/pipeline.yml contiene pruebas, investigación e
inferencia separados. Las PR solo ejecutan pruebas sin secretos. La investigación
lee los archivos estáticos de datasets/eurusd, calcula features, entrena, evalúa
el holdout y publica el artefacto forex-alert-bundle. No descarga cotizaciones.

1. Los archivos HistData de 2021–2025 y enero–octubre de 2026 ya están incluidos en datasets/eurusd/. No hace falta subirlos manualmente ([procedencia](datasets/eurusd/manifest.json)).
2. Añadir los secretos de repositorio TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID.
3. Dar al bot permiso para publicar en el chat/canal configurado.
4. Ejecutar manualmente el workflow con mode=train y esperar a que termine bien.
5. Usar mode=alerts o el cron de alertas cada quince minutos.

Los runners recuperan el último bundle de una ejecución exitosa de este workflow
en la rama por defecto. No se entrena cada quince minutos. El bot continúa GARCH
desde el estado archivado y solo usa la última vela cerrada y aún vigente.

GitHub Actions no garantiza puntualidad. Con retraso superior a 180 segundos,
mercado cerrado o datos incompletos se omite la alerta. La outbox y concurrencia
reducen duplicados; la caché no garantiza exactly-once. Una entrega incierta queda
pending para evitar reintentar un POST que quizá ya se entregó.

La ingesta es exclusivamente local: CSV, ZIP con CSV y Parquet. HistData usa
EST fijo (UTC-5), sin horario de verano. Los M1 con nombre oficial se agregan a
M15 con 15 minutos completos; para M1 renombrados usar --input-timeframe M1.
Los CSV ya convertidos a UTC deben incluir Z/offset o usar --timezone UTC.
No se interpolan huecos. Cierres comprobados se declaran en config/closures.json.

El snapshot incluido contiene 135,706 velas M15 UTC. Se descargaron 15 ZIP
oficiales de HistData y se agregaron únicamente grupos de 15 minutos completos.
Se excluyeron minutos contradictorios y grupos incompletos; config/closures.json
registra los huecos observados del feed, sin afirmar que fueran cierres de
mercado y sin rellenarlos. La validación sigue rechazando huecos no documentados.
El entorno separa los segmentos y no expira contratos atravesando un hueco.

Un histórico estático sirve para investigar y entrenar. Para señales vigentes
hay que actualizar los archivos con la última vela cerrada; el cron no obtiene
precios nuevos y omite datos caducados. Los secretos Telegram siguen vigentes.

## Documentación y alcance de pruebas

- [Ingesta estática y carga desde Android](docs/static_ingestion.md)
- [Inferencia y alertas](docs/telegram_alerts.md)
- [Entorno y PPO](docs/binary_options_training.md)
- [Backtest y métricas](docs/backtesting_pipeline.md)

Las pruebas deterministas incluyen lectura CSV/ZIP sin red, conversión EST/UTC,
agregación M1, integridad de caché, un entrenamiento PPO mínimo con datos
sintéticos, holdout, causalidad e inferencia incremental. Telegram se simula:
las pruebas no envían mensajes reales. Los reintentos HTTP restantes pertenecen
exclusivamente a Telegram; no existe cliente HTTP de datos de mercado.

Sin un histórico/modelo reales y un backtest independiente completado, estas
pruebas no acreditan rendimiento financiero de mercado.

