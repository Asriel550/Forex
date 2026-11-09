# PR #35 — Snapshot GARCH duradero y auditoría de oportunidades M5

## Fuente duradera privada

Se migra una sola vez el snapshot canónico de la PR #34 desde el artefacto efímero
13639251699 hacia una **GitHub Release privada de Forex**, que no depende de la
retención de GitHub Actions. La Release almacena dos activos binarios/versionados
fuera del árbol Git (no se meten megabytes en commits):

- Tag: `data-m5-is-2026h1-garch-v1`.
- Asset `a.npz` (etiqueta `m5-is-2026h1-v1.npz`): IS enero–junio 2026, 2.859 propuestas de marzo.
- Asset `a.json` (etiqueta `m5-is-2026h1-v1.json`): fuentes, firmas GARCH/MC, semillas, fechas y auditorías.
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

## Resultados medidos (CI PR35)

[Workflow aprobado, reconstrucción exacta de los seis meses](https://github.com/Asriel550/Forex/actions/runs/42538864576).
[JSON completo de la auditoría](https://github.com/Asriel550/Forex/actions/runs/42538864576/artifacts/13640940430).

Las huellas GARCH/MC y las máscaras de entrada de **cada mes enero–junio**
coincidieron con el snapshot canónico. La admisión principal resulta:

| Tramo | M5 elegibles | Cumplen entry>0 | Pasan entry + margen | Tras cupo (2/h) | Capacidad/h |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 17.287 | 7.159 | 5.594 | 1.766 | 1,226 |
| Q2 | 17.974 | 5.360 | 4.374 | 1.425 | 0,951 |
| Semestre | 35.261 | 12.519 | 9.968 | 3.191 | 1,086 |

El `activation_threshold` congelado de la cabeza primaria vale
**5,220727152415592**. Todas las decisiones de dirección evaluadas tenían
margen no nulo, por lo que esa cláusula del filtro no explica la pérdida.

**Q2: ablation de cobertura (antes de p85, sin PnL y sin stop):**

| Filtro primario | Propuestas Q2 | Tras cupo (2/h) | Capacidad/h |
| --- | ---: | ---: | ---: |
| Original (entry>0 y margen>5,2207) | 4.374 | 1.425 | 0,951 |
| Margen reducido a la mitad | 4.858 | 1.482 | 0,989 |
| Eliminar umbral de margen | 5.360 | 1.535 | 1,025 |
| Relajar ligeramente entry (−0,25 × std del logit) | 5.738 | 1.746 | 1,166 |
| Eliminar el filtro entry sin alterar margen | 14.799 | 3.003 | 2,005 |
| Sin ninguno de ambos filtros | 17.974 | 3.090 | 2,063 |

**Diagnóstico:** en Q2 el filtro de entrada elimina 70,2% de velas
elegibles *antes* de aplicar el margen, frente al 18,4% adicional que
descarta el margen entre las velas que superaron entry. El filtro primario
de entrada aporta la restricción más intensa. Reducir solo el umbral
de margen no alcanza 1,2/h ni siquiera antes del gate adaptativo.

Estos valores son **capacidad potencial superior** con las propuestas y el
cupo horario, no una predicción de frecuencia ejecutada bajo p85/p80/p75,
y **no certifican PnL, PF o drawdown** de ninguna nueva propuesta. Abrir
el filtro puede introducir operaciones de menor calidad. Para evolucionar
una cabeza de admisión con nuevas señales se necesita un nuevo pipeline
causal de candidatos y una validación IS por trimestre; no retocar los
genomas congelados ni observar Q3.
