> Actualización PR #10: [contrato de mercado v4](market_contract_v4.md). `train` ahora publica un bundle de investigación aislado; `diagnostics` conserva el protocolo PR8.

# Diagnóstico sin volver a inspeccionar el holdout

## Política operativa

El holdout histórico queda congelado: pipeline.yml ya no invoca backtest.py.
Los tests unitarios de backtest siguen usando datos sintéticos, sin evaluar el
histórico reservado. Push/PR ejecutan pruebas; el cron semanal ejecuta pruebas,
y el cron M15 conserva las alertas. La investigación requiere workflow_dispatch
sobre la rama por defecto. Esto evita que fusionar un diagnóstico reentrene o
reemplace automáticamente el bundle activo.

| mode | Trabajo | Publica bundle | Evalúa holdout |
| --- | --- | --- | --- |
| diagnostics | Semillas PPO 42, 43, 44 y diagnóstico MC/PPO | No | No |
| train | Entrenamiento manual, selección por validación | Sí | No |
| alerts | Inferencia causal con bundle exitoso | No | No |
| telegram_test | Mensaje técnico sin precios/modelo | No | No |

El bundle de train sigue siendo un artefacto de investigación seleccionado por
validación; publicarlo no certifica rentabilidad. No se elimina backtest.py ni
los reportes anteriores, pero ninguna rama del workflow lo ejecuta. No se cambia
la frescura exigida a señales reales.

## Ejecutar el estudio desde Android

1. Tras fusionar, abrir Actions → Forex M15 research and Telegram → Run workflow.
2. Elegir la rama por defecto y mode=diagnostics. El barrido usa exactamente las
   semillas 42, 43 y 44 y 100.000 pasos solicitados por semilla (PPO completa el
   rollout, normalmente 100.352 pasos efectivos).
3. Primero descarga el último bundle exitoso de la rama por defecto y verifica
   sus hashes. Lo diagnostica en train/validación con sus parámetros archivados,
   sin reentrenar. El reporte queda en diagnostics/active_bundle/.
   Después el barrido fija PR #8: ent_coef=0.02, confianza=0.60, drift_scale=1.5,
   trade_penalty=0.025, blocked_penalty=0.05, neutral_opportunity_penalty=0.005,
   payout=0.80 y stop de drawdown=20%. Ignora los inputs/variables de estrategia
   de mode=train para no mezclar un barrido de semillas con un barrido de parámetros.
4. La semilla Monte Carlo queda fija en 42 y las features se calculan una sola
   vez; varían solo la inicialización/muestreo PPO y los resets del entorno.
5. Descargar forex-reports-<run>-<attempt>. Contiene reports/diagnostics/
   diagnostics.json, diagnostics.csv y diagnostics.md; por semilla, auditorías,
   historial de validación, manifiesto y CSV de decisiones train/validation.
   Los pesos experimentales no se publican como forex-alert-bundle.

CLI equivalente:

```bash
python -m src.data_loader --input-dir datasets/eurusd --closures config/closures.json
python -m src.quant_engine --drop-warmup --seed 42 --drift-scale 1.5
python -m src.diagnostics --seeds 42 43 44 --total-timesteps 100000
```

El CLI permite predefinir 1..10 semillas únicas; no elige una ganadora. Para
repetir localmente usar un --report-dir nuevo: no se sobreescriben experimentos.
Cada semilla normaliza solo train y selecciona su checkpoint solo en validación.
No se usa el score o PnL del holdout, ni se buscan objetivos de número de contratos.

## Interpretación de los denominadores

- **MC raw**: dirección con mayor probabilidad antes de umbral/edge.
- **MC elegibles (fase completa)**: contratos que el motor propondría tras el
  umbral y edge conservador, sin PPO ni stop de cuenta. Solo transiciones M15
  consecutivas. Se cuentan sin leer el precio de liquidación.
- **MC baseline**: economía de esa regla con su propio saldo/stop. Distinguir
  eligible_before_stop de los contratos que realmente alcanza a simular.
- **PPO raw**: acción predict(deterministic=True) antes de que el entorno la
  filtre; usa el estado de cuenta derivado de las acciones ejecutadas.
- **PPO ejecutado**: propuesta tras el filtro. Se registran bloqueos por
  confianza o edge, NEUTRO voluntario y la tabla cruzada MC/PPO.
- **Exposición PPO**: velas evaluadas hasta su propio stop. Después del stop no
  se inventan propuestas ni saldos. Se reportan los huecos y filas omitidas.
- **Captura de oportunidades**: propuestas PPO alineadas con una dirección MC
  elegible / oportunidades MC durante esa misma exposición. Las otras se
  separan entre NEUTRO voluntario y propuesta en dirección contraria.

Pocas oportunidades MC ya antes de PPO indican un filtro muy restrictivo sobre
estas features. Muchas oportunidades elegibles con baja captura indican que PPO
no las toma. Son conteos observacionales: por sí solos no prueban que la causa
sea la recompensa, la falta de señal, la arquitectura o un atractor local.

Se reportan operaciones, acierto sin reembolsos, PF, PnL y captura por semilla,
con media/mediana/desviación muestral/mínimo/máximo. PF infinito por ausencia de
pérdidas y PF sin contratos conservan estados explícitos, no se convierten a cero
ni se mezclan en la media de PF finitos. El acierto sin contratos es N/D.
Tres semillas solo aportan evidencia exploratoria sobre estabilidad; no son
muestras independientes del mercado ni demuestran una propiedad inherente.

## Prueba real de Telegram sin depender del mercado

1. Elegir Run workflow sobre la rama por defecto, mode=telegram_test.
2. Usar telegram_test_id=mobile-v1 (1..64 letras ASCII, números, guiones o guion
   bajo). Se permite ejecutar con mercado cerrado y sin cotizaciones recientes.
3. Este modo no descarga bundle ni calcula señales. Usa los secretos ya
   configurados, el mismo send_telegram y la misma función deliver_once de
   producción. Envía un mensaje marcado PRUEBA DE ENTREGA, sin CALL/PUT.
4. Buscar en logs el JSON status=sent y message_id. Eso confirma que Telegram
   aceptó sendMessage para el chat configurado; la recepción física requiere
   que el usuario vea el mensaje en su móvil. El bot debe tener permisos en el canal.
5. Ejecutar de nuevo con el MISMO ID: debe aparecer already_sent sin un POST
   nuevo. Una respuesta incierta conserva pending y falla la prueba sin reenvío.
   IDs distintos conservan archivos distintos; no borran la deduplicación previa.

La outbox técnica está aislada en data/telegram_tests/<hash>.json. No registra
el token ni chat_id en claro. Se persiste antes del HTTP y se guarda incluso si
el paso falla. Comparte el grupo de concurrencia de alertas para serializar los
runners. No tocar ni borrar una entrada pending para forzar un segundo envío.

Telegram no ofrece una clave de idempotencia para sendMessage. Solo se reintenta
429 explícito dentro del presupuesto existente; timeout/5xx/cuerpo ambiguo no
se reintentan. 401/403 u otro rechazo explícito permiten volver a probar el mismo
ID después de corregir token/permisos. Los logs evitan imprimir URL/token y cuerpo
remoto. La prueba usa sendMessage para comprobar token y destino a la vez.

La caché de Actions puede expirar, ser desalojada o no guardarse tras un fallo
abrupto del runner. La deduplicación se garantiza solo mientras se conserva el
estado; no hay una promesa de exactly-once entre fallos de persistencia. Revisar
que Persist technical test sent or uncertain outbox guarde correctamente; no
repetir una entrega incierta si se perdió la caché sin revisar primero Telegram.
Para garantías más fuertes haría falta un almacén durable transaccional externo.

Esta entrega agrega y prueba el mecanismo mediante HTTP simulado. Un resultado
de CI verde no se presenta como confirmación de que llegó al dispositivo móvil.

## Resultados de esta entrega

[Diagnóstico del bundle activo y semillas 42/43/44](research/pr8_filter_policy_diagnosis.md), con JSON de hashes, métricas y dispersión. No se consultaron resultados del holdout.
