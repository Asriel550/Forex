# PR43 — 100 generaciones ADICIONALES desde los 12 campeones de PR42

**Corrección de alcance**: NO son 100 entrenamientos nuevos por divisa
inicializados desde cero. Se toma cada uno de los **12 genomas
campeones ya aprobados de la PR41/PR42**: 4 pares × semillas 42/43/44.

Cada campeón se carga directamente del artefacto final verificable de PR41
(run `42576852134`), conservando sus 22 genes, SHA256, fuente GARCH nativa,
métrica Q1/Q2 y generación donde resultó ganador. No se modifica
la PR41 ni la PR42. La primera generación de nueva descendencia comienza
en el cromosoma campeón y evoluciona **100 generaciones adicionales**;
identificadores de continuidad `21..120` por linaje.

**Presupuesto**: 12 linajes × 100 generaciones × 128 individuos =
**153.600 nuevas evaluaciones**. Cada linaje mantiene sus parámetros
GARCH propios, entrada y dirección nativas; el crossover/mutación ocurren
solo entre descendientes del mismo símbolo y semilla. Se conserva
el ranking y evaluación económica/filtro original PR41
(PF Q1/Q2 >1,20; DD<15%; PnL positivo y sin stop),
cuenta, payout y cap independientes del activo.

## Precisión de continuidad

PR41 no guardó el estado interno completo de RNG/población/estado de
`sigma` para continuar bit a bit desde su última generación. Por tanto,
la continuación es **genética desde el genoma ganador congelado**,
con RNG separado `(seed, first_generation)` y sigma inicial .12 del
mismo motor; **no** es una reanudación exacta del flujo aleatorio
del experimento anterior. Se preserva la ascendencia de campeón y
no se reinicia desde ceros. Esta condición se registra en el reporte.

## Protección de Q3

Todos los universos H1 están sellados por símbolo con los mismos SHA256
de PR41. Solo existe 2026-01-01 a 2026-07-01 (fin exclusivo).
**Q3 julio–septiembre no se lee ni evalúa en ninguna etapa**.
La PR nueva no dispara broker, alertas, trading real ni promoción.
No se escogen ganadores utilizando datos OOS. Repetir búsqueda
sobre el mismo IS aumenta riesgo de sobreajuste: los mejores PF
obtenidos aquí seguirán siendo in-sample, no alfa confirmado.

## Evidencia

- `scripts/run_pr43_100_native_trainings.py`: carga y reproduce el
  fundador PR42, evoluciona 100 generaciones, archiva genoma original,
  genoma descendiente, mutaciones y métricas.
- `tests/test_pr43_100_native_trainings.py`: prueba sintética de
  descendencia directa del cromosoma campeón, independencia y
  frontera H1.
- `.github/workflows/pr43-100-native-trainings.yml`: 12 jobs,
  uno por símbolo/semilla, consolida exactamente 153.600
  evaluaciones y 12 historiales de 100 generaciones.
- No fusionar ni promocionar a producción antes de verificar
  resultados y establecer protocolo OOS separado, sin abrir Q3.
