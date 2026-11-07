# Diagnóstico PR #8: motor MC, PPO archivado y tres semillas

Se conservaron ent_coef=0.02, confianza=60%, escala momentum=1.5×, payout=0.80,
los costes de PR #8 y el stop de drawdown 20%. **No se evaluó el holdout ni se
promovieron pesos.** El JSON adjunto contiene hashes, versiones, auditorías y
dispersión. Estas corridas locales son investigación previa al workflow.

## Motor antes de PPO

| Fase | Transiciones M15 consecutivas | Oportunidades elegibles | Tasa elegible | Baseline MC: PnL USD | PF | Acierto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 90.827 | 82 | 0,0903% | −64 | 0,84 | 51,22% |
| Validación | 26.395 | 48 | 0,1819% | +96 | 1,60 | 66,67% |

El baseline simula todas las propuestas elegibles con su propio saldo y stop;
ninguno de estos dos subconjuntos alcanzó el stop. MC no representa certeza
calibrada. El rendimiento divergente entre train y validación no demuestra una
ventaja estable ni autoriza buscar parámetros en el histórico reservado.

## Bundle activo de PR #8

Se descargó el artefacto 13549426467 de la ejecución 42312722964, se verificó el
SHA256 del ZIP y los hashes de datos, normalizador y modelo. Se evaluaron
exclusivamente sus subconjuntos train/validation con el PPO archivado.

| Fase | MC elegibles | PPO alineado y ejecutado | NEUTRO sobre oportunidades elegibles | PPO contrario sobre oportunidades elegibles | Propuestas bloqueadas (toda la fase) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 82 | 1 | 81 | 0 | 68 |
| Validación | 48 | 1 | 47 | 0 | 9 |

En validación PPO propuso en bruto 26.385 NEUTRO, 3 PUT y 7 CALL. El filtro
rechazó 9 propuestas y aceptó 1 CALL. Capturó solo 1/48 = 2,08% de las
configuraciones elegibles durante su exposición. En train capturó 1/82 = 1,22%.
Hay dos restricciones observables: el motor ya habilita menos del 0,2% de velas,
y PPO omite la mayor parte de esas oportunidades. Los conteos no prueban por sí
solos que la recompensa sea la causa; no se simula un PPO monetario sin filtro
que cambiaría retrospectivamente su estado de cuenta.

## Barrido PPO (100.352 pasos efectivos por semilla)

Se regeneraron las features una vez con MC seed42. Solo varió seed PPO/reset.
Cada corrida seleccionó su checkpoint por validación; no se eligió una semilla
ganadora ni se publicaron esos pesos. Las tres comparten input y versiones.

| Seed | Contratos train | PnL train USD | PF train | Contratos validación | PnL validación USD | Acierto validación | PF validación |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 42 | 1 | −10 | 0 | 0 | 0 | N/D | Sin contratos |
| 43 | 1 | +8 | Sin pérdidas | 1 | +8 | 100% (1 contrato) | Sin pérdidas |
| 44 | 3 | +6 | 1,6 | 1 | +8 | 100% (1 contrato) | Sin pérdidas |

En validación: contratos media 0,667; desviación muestral 0,577; mínimo 0 y
máximo 1. PnL medio US$5,33; desviación muestral US$4,62. Ningún PF de validación
es finito: uno sin contratos y dos sin pérdidas. No se promedian esos estados
como cero ni infinito. El 100% de acierto de un contrato no es evidencia robusta.
La captura de oportunidades fue 0%, 2,08% y 2,08%, respectivamente.

La abstención persiste en estas tres semillas. Esto sugiere estabilidad del
comportamiento bajo este protocolo, pero no demuestra un atractor local o una
propiedad inherente. Tres semillas no son tres muestras independientes de mercado.
No se cambian recompensas, umbrales, escala ni entropía como resultado del estudio.

## Reproducibilidad y límites

El barrido local usó pandas 2.2.3; el bundle publicado se entrenó con pandas
2.3.3. Sus Parquets tienen hashes distintos. No se presenta el reentrenamiento
local seed42 como una reproducción bit a bit del checkpoint archivado. Ambos
estudios registran versiones y hashes por separado. Las auditorías del bundle
usan su Parquet original y el normalizador sellado. La confirmación operativa
se obtiene ejecutando mode=diagnostics en Ubuntu con el mismo protocolo.

La prueba Telegram se implementó con el sender/outbox de producción y tests HTTP
simulados. No se usaron ni leyeron secretos en este entorno y no se afirma que
un mensaje haya llegado al móvil. Ejecutar mode=telegram_test tras fusionar y
seguir el manual docs/train_validation_diagnostics.md.
