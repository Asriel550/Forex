# Experimento de saturación: 100 generaciones más (Gen398–497)

## Pregunta

¿La cabeza primaria M5+M15 sigue mejorando en IS si prolongamos la
neuroevolución con 100 generaciones adicionales de 128 modelos, sin cambiar
arquitectura, gate ni datos? Si no mejora durante muchas generaciones,
eso es evidencia de **meseta de este procedimiento y conjunto IS**, no una
demostración de techo universal o de que el mercado sea inoperable.

## Continuidad y trazabilidad

- **Tres semillas originales:** 42, 43 y 44, en jobs independientes.
- **Fundadores:** mejores genomas verificados por SHA256 de la corrida
  Gen376–397 (run `42544173112`), generaciones ganadoras 392, 392 y 386.
- **Nuevas generaciones:** 398–497 (100 inclusive). El algoritmo original
  `evolve_population` siembra al mejor fundador y reanuda exploración y
  mutación con una nueva secuencia RNG `(seed,398)`. No se reconstruye
  el estado interno de la población Gen397 ni su RNG: **continuación desde
  el mejor genoma, no continuación bit a bit del proceso anterior**.
- **Población:** 128. `100 × 128 × 3 = 38.400` evaluaciones nuevas.
- **Feature set:** exclusivamente M5 y M15 completos, universo sellado
  `data-m5-m15-is-2026h1-full-universe-v2`. M1 no aporta señales.
- **Red direccional:** Gen375/43 congelada. Gate causal Convicción Pura p85,
  cuenta y cupo sin cambios. Q1/Q2 de 2026 siguen siendo IS.
  **Q3 julio–septiembre continúa completamente aislado.**

Cada founder se reproduce antes de cualquier mutación. Una discrepancia
en scores, PF, DD, frecuencia o hash aborta la ejecución.

## Observabilidad de meseta

Por semilla se reportan las 100 generaciones, medianas de fitness, mejor
score acumulado, momentos de mejora, último avance y número de generaciones
sin progreso. Se compara explícitamente antes/después:

- Cambio del fitness (valor absoluto, sin fabricar mejoras estadísticas).
- PF en Q1 y Q2; frecuencia real semestral ejecutada después de p85,
  cupo y stops; drawdown de ambos trimestres.
- Si aparece algún modelo que cumpla todos los criterios financieros
  (PF >1,20 ambos trimestres, frecuencia ejecutada >=1,2/h, DD <15%,
  PnL positivo y sin stops).

Es posible mejorar el fitness sin mejorar PF o frecuencia. La pregunta
de techo financiero requiere analizar ambos, **no solamente fitness**.

Los 38.400 candidatos se puntúan contra un IS ya muy explorado; cualquier
mejora encontrada es diagnóstico retrospectivo, no demostración de
generalización. No se evalúa Q3 ni se promocionan modelos automáticamente.

## Ejecución

El workflow `M5+M15 100 extra generations, Gen 398-497` se lanza
una sola vez por el marcador versionado `research/triggers/gen398-497-plateau-v1.txt`.
Genera tres artefactos individuales y una síntesis conjunta con todos los
scores, PF, DD, frecuencias, deltas y cuenta de generaciones estancadas.

Los resultados deben verificarse antes de cualquier cambio operativo.
