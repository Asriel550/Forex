# Validación de integración — 6 de noviembre de 2026 (Lima)

Ejecución de referencia: https://github.com/Asriel550/Forex/actions/runs/42287033782
Commit: 01e12ed8ed6a4a94042aa3196e8a6230a7a8c18c.
Evento: push a main. Recorrió los mismos pasos de investigación que mode=train,
con resultado success; no fue una ejecución manual workflow_dispatch.

## Entrenamiento y holdout comprobados

- Ingesta estática: 135.706 velas M15; no hubo descarga de mercado.
- Features causales: 135.194 filas después del warmup; 1.409 ajustes GARCH.
- Monte Carlo: 10.000 rutas, horizonte 32, semilla 42, drift cero.
- PPO: 100.000 pasos solicitados, 100.352 realizados por completar el rollout.
- Preparación para Gymnasium: 135.178 filas, observaciones de 28 valores.
- Train: 94.624 filas; validación: 27.035; holdout: 13.517. Una barra de purga
  al entrar en validación y otra al entrar en test.
- El normalizador solo usa train; el checkpoint se selecciona por beneficio
  de validación. El holdout no interviene en el aprendizaje ni en esa selección.
- El backtest comprueba SHA256 y reconstruye los límites del split antes de
  evaluar. Evaluó 13.104 decisiones; omitió 412 transiciones sobre huecos.

El holdout cubre del 7 de abril al 30 de octubre de 2026. Hubo 1.615 contratos,
837 aciertos, 758 pérdidas y 20 reembolsos. Beneficio neto: -US$884 sobre
US$1.000; saldo final: US$116; win rate con reembolsos: 51,83%; profit factor:
0,8834; drawdown máximo: 90,67%; Sharpe: -2,05. La validación técnica del flujo
no valida rentabilidad: este checkpoint tiene pérdidas sustanciales.

## Bundle real descargado y verificado

Artifact forex-alert-bundle, ID 13538716061, de la ejecución anterior en main.
Contiene pesos, normalizador, training_manifest.json y el Parquet enriquecido.
Los hashes internos y el digest del ZIP coinciden con los artefactos descargados.
SHA256 comprueba integridad; no es una firma digital del modelo.

| Archivo | SHA256 |
| --- | --- |
| ZIP forex-alert-bundle | 6b79d8db1e7712e43d0c91718d27ae1bc9ecf9d069bd346d362e78267af27302 |
| models/best_ppo_agent.zip | 73def0f0bab44aa9e3d5a716c09751c370f52fbedcf7568125babc0f0ac389da |
| models/normalizer.json | 170087158e21d7ce261859217e5e912a2fe52eec197eba87acff6c0bf85308b5 |
| models/training_manifest.json | 0bcd98b7b0d1add7a3f01bd7f202f714cf38f88339950a6054c80a1f1b3f8c92 |
| data/eurusd_m15_features.parquet | 6fa5ab1e6608338dd3ddde65f71df2b158b89ccb43bb98b36a0d008bed300690 |

## Inferencia y Telegram

El workflow selecciona el último bundle no expirado de una ejecución exitosa
del mismo workflow y repositorio, sobre la rama por defecto. No usa PR ni un
entrenamiento fallido. Los artefactos se conservan 30 días.

La continuación de GARCH mantiene la fila global y la fase de refit originales:
actualiza la varianza cada nueva vela y reestima cada 96 observaciones, solo
con pasado. Sobre las últimas 200 barras del bundle real, la diferencia máxima
frente a las features archivadas fue 3,87e-10, dentro de tolerancia numérica
(relativa 1e-6/absoluta 1e-11 entre entornos; no se afirma igualdad bit a bit).
La inferencia histórica con reloj controlado y dry_run cargó el PPO real y
produjo una observación compatible; no envió HTTP ni escribió una outbox.

El control con fecha actual rechazó el snapshot: la última apertura es el
30 de octubre a las 15:30 de Lima y el cierre a las 15:45. Con datos estáticos
hasta esa fecha no existe una vela cerrada actual para generar una alerta.
Reentrenar sobre el mismo snapshot no crea cotizaciones nuevas. Se mantiene
el límite de frescura de 180 segundos y la comprobación inmediatamente antes
del envío; no se presenta una predicción histórica como una señal actual.
Un bundle cuyo último dato supere 14 días también se rechaza, aunque se haya
entrenado de nuevo recientemente con los mismos archivos antiguos.

La outbox escribe pending antes de sendMessage y sent tras message_id confirmado.
Una entrega incierta por timeout o 5xx conserva pending y no repite ese chat/vela.
Se corrigió la clasificación de 5xx con ok:false: tampoco son rechazos seguros.
Solo 429 explícitos permiten backoff con retry_after y presupuesto limitado;
se vuelve a comprobar la frescura antes del siguiente intento. Un rechazo
confirmado de credenciales/permisos queda rejected.

Los secretos TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID se referencian únicamente en
el paso de envío. Las pruebas usan credenciales ficticias y HTTP simulado; esta
revisión no acredita autenticación real ni entrega al móvil. No se expusieron
ni cambiaron los secretos del repositorio.

Concurrencia y caché de Actions reducen duplicados, pero no garantizan exactly-once:
si se pierde la caché o el runner cae antes de persistirla tras enviar, puede
haber duplicados. Pending también puede omitir una entrega que nunca llegó.
Una garantía más fuerte exige un almacén duradero para ese estado.

## Ejecución manual desde Android

1. Abrir https://github.com/Asriel550/Forex/actions/workflows/pipeline.yml.
2. Elegir Run workflow, Branch=main y mode=train. El valor por defecto del modo
   es alerts, por lo que hay que cambiarlo expresamente.
3. Pulsar Run workflow. Esperar unit-tests y research-pipeline en verde.
4. Comprobar que Read static EURUSD CSV or ZIP files, Causal quant features and
   continuation state, Train with untouched holdout y Holdout backtest terminan
   con success. El job telegram-alert omitido en train es normal.
5. En Artifacts de esa ejecución deben aparecer forex-alert-bundle y forex-reports-….
   En los reportes revisar metrics.json, trades.csv y las gráficas. El bundle
   se vincula automáticamente a alertas por su ejecución exitosa; no se copia a Git.
6. Para probar la ruta de alertas, ejecutar el mismo workflow con main y mode=alerts.
   La descarga del bundle puede funcionar y el envío omitirse por datos caducados:
   un job verde con ese skip no acredita una notificación ni autenticación Telegram.

Los pasos UI corresponden al flujo documentado por GitHub:
https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow
