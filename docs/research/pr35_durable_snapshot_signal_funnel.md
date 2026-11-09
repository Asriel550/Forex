# PR #35 — Snapshot GARCH duradero y auditoría de oportunidades M5

## Fuente duradera privada

Se migra una sola vez el snapshot canónico de la PR #34 desde el artefacto efímero
13639251699 hacia una **GitHub Release privada de Forex**, que no depende de la
retención de GitHub Actions. La Release almacena dos activos binarios/versionados
fuera del árbol Git (no se meten megabytes en commits):

- Tag: `data-m5-is-2026h1-garch-v1`.
- `m5-is-2026h1-v1.npz`: IS enero–junio 2026, 2.859 propuestas de marzo.
- `m5-is-2026h1-v1.json`: fuentes, firmas GARCH/MC, semillas, fechas y auditorías.
- SHA256 del snapshot: `7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504`.
- SHA256 del manifiesto: `d115db5f4799aa3b6920d5d7539778c8b611452844211a98fc3e8ccc7f689f04`.

El job de publicación valida los bytes originales y crea la Release solo si no
existe. Si existe, **no sobrescribe** los activos y exige igualdad byte a byte
con el original. Los jobs de evaluación descargan el archivo por tag mediante
`gh release download` autenticado y fallan si cualquier SHA256 cambia. Cada
versión distinta debe usar un nuevo tag y nuevas huellas.

GitHub Releases es un almacenamiento privado y versionado vinculado al
repositorio, no un repo de datos independiente, Git LFS ni un servicio S3/GCS.
Evita credenciales externas y las expiraciones automáticas de los artefactos de
Actions, pero administradores aún podrían borrar la Release; los consumidores
fallan cerrados por SHA256. Para recuperación ante desastres es aconsejable
replicar este mismo snapshot a un almacén con versionado/retención bloqueada.

Los ZIP M1, la reconstrucción OHLCV M5 y el propio archivo GARCH se verifican
por separado. **La evolución jamás vuelve a ajustar GARCH**, corrigiendo la
inestabilidad numérica del refit 16.928 diagnosticada en PR #34. El workflow de
PR #34 pasa también a usar la Release en vez del artefacto temporal.

El dataset cubre **seis meses IS**, no solo marzo; contiene la máscara de
marzo para reproducir exactamente 2.859 propuestas. Julio–septiembre Q3
permanece fuera del acceso.

## Embudo de señales antes del Gate Adaptativo

La arquitectura heredada de `FrozenDirectionSimulation` puntúa TODOS los
candidatos mediante dos cabezas fijas de un fundador anterior, y solo permite
una propuesta cuando:

1. `entry > 0`.
2. `abs(direction) > 0`.
3. `abs(direction) > activation_threshold` del fundador.

Las Gen 354–376 **evolucionan dirección sobre este subconjunto ya recortado**:
no pueden aumentar la capacidad de entrada ni aunque el fitness recompense
frecuencia. PR #32 midió capacidad máxima **1,226/h Q1 y 0,951/h Q2**
con este conjunto de propuestas y el cupo horario.

`scripts/audit_m5_primary_triggers.py` muestra por mes, Q1, Q2 y semestre:

- Barras M5 elegibles, cuántas sobreviven el filtro `entry > 0`, el margen
  y ambos filtros, más sus distribuciones de scores.
- Conteos/capacidad con cupo horario de la regla original, reducción del
  umbral de margen a la mitad, margen cero, entry ligeramente relajado y
  supresión independiente de cada barrera.
- Comprobación del mismo SHA256 de propuestas frente al snapshot de la
  Release. El nuevo informe es **no comparable** si el GARCH reconstruido
  varía; nunca crea una nueva referencia de trading con esas señales.

Son **sensibilidades de admisión, no backtests**: no incluyen Gate Adaptativo,
PnL ni Profit Factor para señales nuevas. Tampoco se considera que una
oportunidad adicional sea automáticamente una señal con valor. Antes de
ampliar la entrada real habrá que entrenar una cabeza de admisión y auditar
calibración de confianza, calidad del feed M1→M5→M15, pérdidas, drawdown y
densidad bajo el mismo gate causal, con nuevo experimento IS y OOS intacto.

El histórico HistData es OHLCV M1 agregado; **no proporciona por sí solo**
spread real de cotizaciones o libro de órdenes. Indicadores de microestructura
construidos de M1 son proxies, no order flow real.

## Bloqueos que permanecen

No lanzar Gen 376–396 por haber ampliado un contador de señales.
El Gate de Convicción Pura no cambia. PF>1,20 y DD<15% siguen requeridos
antes de promover genomas; la meta 1,2/h no puede alcanzarse en Q2
bajo el embudo congelado actual. Q3 permanece aislado.
