> Operación actual: [diagnóstico y prueba técnica de Telegram](train_validation_diagnostics.md). Ya no se ejecuta backtest histórico automáticamente.

# Inferencia Telegram con archivos estáticos

Este ZIP es una ACTUALIZACIÓN para aplicar en la raíz del proyecto completo
anterior. No elimina ni reemplaza backtest.py/metrics.py ni el resto de módulos.
Incluye los archivos completos modificados, los dos tests nuevos y este manual.

## Archivos y primer arranque

1. Copiar src/, tests/, .github/, requirements-alerts.txt y docs/ sobre el proyecto.
2. En GitHub, Settings > Secrets and variables > Actions, crear secretos de
   repositorio TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID. No escribirlos en código,
   YAML, logs o issues. El bot debe poder publicar en el chat/canal configurado.
3. El snapshot HistData de 2021–octubre de 2026 ya está incluido en datasets/eurusd/, con su manifiesto y exclusiones; no requiere carga manual.
4. Ejecutar el workflow manual con mode=train. Esto vuelve a generar features,
   entrena, selecciona por validación y publica el artefacto forex-alert-bundle. El holdout histórico queda congelado.
5. Tras una ejecución exitosa, ejecutar manualmente mode=alerts o dejar el cron.

El bundle contiene models/best_ppo_agent.zip, normalizer.json,
training_manifest.json y data/eurusd_m15_features.parquet. Los runners no
comparten filesystem: el job de alertas busca y descarga el último bundle de
una ejecución exitosa de pipeline.yml sobre la rama por defecto del mismo
repositorio. No usa artefactos de PR, modelos parciales ni checkpoints al azar.

train.py registra hashes de datos, pesos y normalizador. quant_engine.py añade
source_rows/origin_timestamp para continuar la fase exacta de sus refits.
Los bundles antiguos sin estos campos deben regenerarse una vez; no se añade
un origen supuesto a modelos previos. Después el cron no vuelve a entrenar
cada quince minutos: solo lee archivos estáticos y hace inferencia. No obtiene nuevas cotizaciones.

## Ingesta sin API pública

Se eliminó el cliente Dukascopy: no hay GET, paginación, pacing ni reintentos
para datos de mercado. La investigación y el bot leen datasets/eurusd/ con la
misma resolución y zona configuradas en el workflow. Ver [ingesta estática](static_ingestion.md).
Los archivos deben conservar la vela de anclaje del bundle y la continuidad
hasta el último cierre disponible. Si cambian los precios del anclaje se exige
regenerar el bundle; no se mezclan silenciosamente fuentes distintas.

La actualización de históricos no es un feed en tiempo real. Mientras no se
aporten archivos recientes, no se enviarán alertas vigentes. No se relajó el
control de frescura para convertir un histórico viejo en una señal actual.

## Qué significa una señal

El bot analiza el cierre de t y predice la dirección para t+1, con el mismo
orden de features y normalizador del entrenamiento. No utiliza la vela que
sigue en formación. Nunca genera una vela futura ficticia para satisfacer
el constructor del entorno ni toma por error la penúltima observación.

Continúa GARCH desde sigma[t+1|t] y omega/alpha/beta del baseline archivado.
Reestima solo en las mismas filas globales que quant_engine, usando su ventana
histórica. Los tests comparan esa continuación con recalcular todo el histórico.
MC conserva semilla, número de rutas, horizonte y drift del modelo original.
La preparación RSI/ATR/lags usa únicamente el pasado disponible.

La observación de cuenta es una cuenta simulada inicial sin posición, con el
capital, stake y payout del manifiesto y dirección/recompensa previa cero.
No afirma conocer un saldo real ni presupone que el usuario ejecutó la señal
anterior. No abre contratos ni coloca órdenes en un broker.

El MC se refiere a 32 barras (normalmente ocho horas de negociación), mientras
que la expiración binaria es de una barra. P05/P95 son percentiles terminales
estadísticos, no barreras garantizadas de soporte/resistencia.

Ejemplo de formato, no señal real:

```text
🔔 SEÑAL EUR/USD M15
Acción: CALL (Comprar)
Volatilidad Cond.: 0.0018
Soporte/Resist. (MC): P05: 1.0820 | P95: 1.0910
Cierre analizado: 2026-11-06 10:00 UTC
Expiración teórica: 10:15 UTC
```

La entrada del backtest era al cierre. Una alerta enviada después incluye
ese cierre de referencia y su expiración teórica: no promete que un broker
ofrezca todavía aquel precio ni un contrato con esa hora exacta de expiración.

## Frescura y estado de envío

El cron de alertas es 1,16,31,46 * * * * en UTC: un minuto después de cada
cierre. El de entrenamiento se mantiene semanal, lunes 06:15 UTC. La ejecución
manual usa mode=alerts o mode=train. Las PR solo ejecutan pruebas sin secretos.

GitHub Actions puede retrasar o descartar ejecuciones. Por defecto solo se
envía si la última vela cerrada corresponde al cuarto de hora actual y tiene
como máximo 180 segundos de antigüedad. Se revisa otra vez justo antes del
POST y después de esperas 429. Mercado cerrado, señal caducada o datos atrasados
producen un skip; un error de integridad/datos/Telegram produce fallo.
Un bundle cuyo histórico termine hace más de catorce días también se rechaza.

Los secretos se inyectan exclusivamente en el paso de envío. El script usa
requests.post a sendMessage, sin parse_mode ni interpolar respuestas remotas
en logs. Los errores no imprimen token, URL del bot ni tracebacks de requests.
Se envían también las predicciones NEUTRO, de acuerdo con el espacio de acciones.

La outbox data/telegram_state.json registra un hash del chat+vela antes de
solicitar envío y queda en sent tras confirmación. Un timeout/5xx ambiguo
permanece pending y no se reintenta automáticamente, porque Telegram no ofrece
una clave de idempotencia para sendMessage. Solo se reintentan rechazos 429
explícitos con retry_after. Rechazos confirmados de configuración se marcan
rejected y permiten corregirlos y reintentar mientras siga vigente la vela.

Un grupo de concurrencia y actions/cache transportan la outbox entre runners.
Esto reduce duplicados, pero NO garantiza exactly-once: una pérdida de caché
o un crash entre el POST y la persistencia de caché puede permitir duplicados.
Una entrega incierta también puede omitir un mensaje si el POST nunca llegó.
Para garantías más fuertes se requiere un almacén transaccional duradero y
un servicio de ejecución con disponibilidad controlada. Actions es best effort.

## Ejecución local y pruebas

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-alerts.txt
# Requiere el bundle coherente y datos recientes; no envía a Telegram:
python -m src.telegram_bot --dry-run
# Con variables de entorno ya configuradas de forma privada:
python -m src.telegram_bot
python -m pytest -q tests/test_rate_limits.py tests/test_telegram_bot.py
```

--input-dir, --history, --model-dir, --state y --live-data permiten rutas distintas.
--input-timeframe y --timezone tienen la misma semántica que data_loader.
--closures admite los mismos cierres documentados del proveedor que la ingesta.
--max-delay-seconds puede cambiar el umbral entre 1 y 899, pero no permite
usar una vela de otro cuarto de hora. Los archivos tienen que incluir
la última vela cerrada; no hay espera de publicación ni descarga automática.

Las pruebas simulan HTTP y Telegram: no envían mensajes reales. Incluyen
429 de Telegram con retry_after, fechas HTTP, igualdad GARCH incremental/completo,
última observación, caducidad,
formato, redacción de secretos y deduplicación/outbox. No se probaron secretos
reales ni se ejecutó el workflow remoto en esta sesión.


## Contexto mínimo para continuar GARCH

En una reestimación se exige exactamente min(window, fila_global) retornos,
además del mínimo de observaciones. Un bundle corto que haya eliminado su
warmup y ya no conserve esa ventana se rechaza: no se ajusta silenciosamente
un modelo con menos historia que el utilizado en la referencia completa.
Los bundles multianuales del pipeline conservan suficiente contexto.

