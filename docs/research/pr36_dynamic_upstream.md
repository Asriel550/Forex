# PR #36 — Evolución causal de la entrada primaria M5

## Hipótesis y límite de validez

La PR35 probó que el embudo fijo `entry>0` y margen>5.2207
deja solo 0,951 oportunidades/h en Q2 después del cupo, **antes** del
Gate de Convicción Pura. Las generaciones de dirección no cambian
esa lista. La PR36 evoluciona únicamente la admisión primaria.

**Advertencia crítica sobre snapshots**: el activo congelado PR35
SHA256 `7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504`
contiene las 9.968 propuestas ya aceptadas por el upstream, **no**
las 35.261 velas elegibles. No se pueden calcular nuevas admisiones
ni sus outcomes a partir de solo esas 9.968 filas.

`build_m5_full_is_universe.py` recalcula enero–junio exclusivamente
desde los seis ZIP M1 del manifiesto, exige la **misma serie GARCH/MC**
que el snapshot original (SHA256), reconstruye la lista completa de
M5 elegibles y verifica que el subconjunto legacy reproduce
**bit a bit** timestamps, outcomes, volatilidad y los 32 inputs de
dirección originales. Si cualquiera cambia, la preparación falla y
**no se ejecuta ninguna evolución**. Solo entonces crea un
nuevo archivo de universo elegible sellado y versionado como
`data-m5-is-2026h1-full-universe-v1` en la Release privada del proyecto.

## Cabeza upstream dinámica

M1 es OHLCV agregado (no L2, bid-ask ni órdenes reales).
El momento de decisión de una M5 abierta a `t` es `t+5min`.
Los proxies M1 usan las **cinco velas M1 completas** dentro de
`[t, t+5min)`: presión firmada de cuerpos M1, movimiento del último
minuto, fracción de subidas y presión de volumen firmada.
Los proxies M5 usan solo la vela completa: tamaño de cuerpo, ubicación
del cierre, rango, y cuerpo de la vela anterior. Se rechazan bloques
con menos de cinco M1 o indicadores no finitos.

Los 11 genes optimizables son ocho pesos para los proxies, sesgo de
activación y dos parámetros de ajuste de margen según el logaritmo
de `σ_actual / mediana(σ de las 48 velas previas)`.

- `entry_evolutiva = entry_legacy + 4*tanh(proxies @ pesos + sesgo)`.
- `umbral_margen = umbral_legacy*(1-0.95*max(0,tanh(gene_base + gene_vol*vol_relativa)))`.
- Solo es candidato cuando `entry_evolutiva>0` y
  `abs(margen_dirección)>umbral_margen`.
- Todos los genes en **cero** reproducen el embudo legacy idéntico.
- La volatilidad **no veta** operaciones: solo modula un margen de
  propuesta. El gate p85 no recibe ni conoce esos genes.

## Entrenamiento y métricas

La dirección está fijada al genoma Gen375/43 con la red neuronal
y normalizador archivados; el gate p85 sigue siendo causal y mantiene
su ventana de las 500 propuestas anteriores y asentamientos vencidos.
Población de 128, mutación/cruce/elitismo existentes, semillas
42/43/44. Se evalúa la robustez temporal por IS Q1 y Q2, no por
el mejor mes. Unidades binarias simuladas con pago +80%/-100% por
stake y stop de cuenta, sin comisiones reales de bróker.

Los reportes separan:

1. Cobertura upstream (propuestas y capacidad máxima tras cupo).
2. Gate p85 (warmup, rechazos y admitidas).
3. Ejecutado realmente después de cupo y stops (PNL, PF, DD,
   contratos/hora).
4. Q1 y Q2 con riesgo reiniciado para diagnóstico, y semestre
   con **una única cuenta** continua para aceptación definitiva.

**Criterio de promoción** (todos a la vez): PF estrictamente
superior a 1,20 en Q1 **y** Q2; PnL positivo y DD inferior al 15%
en ambos; ninguna parada por riesgo; DD semestral inferior al 15%
y frecuencia realmente ejecutada semestral al menos 1,2/h.
No se acepta una capacidad teórica ni un PF calculado sin stop
como sustituto de estos indicadores.

## CI y no contaminación OOS

El CI reconstruye y compara el nuevo universo con PR35 antes de
entrenar; solo entonces ejecuta un smoke Gen376, 128 candidatos,
semilla43. La corrida completa de 21 generaciones **no está
habilitada** en este workflow. El resultado se publica aunque
ningún individuo cumpla objetivos; CI verde significa integridad
técnica, no rentabilidad.

Todos los paths y fuentes se limitan al IS
`2026-01-01T00:00:00Z <= t < 2026-07-01T00:00:00Z`.
Julio–septiembre 2026 (Q3) permanece cerrado, sin leer.
Aun cumpliendo ambos trimestres de IS, debe haber un protocolo
OOS posterior antes de cualquier promoción a trading real.

Cada snapshot y su manifest tienen SHA256 y etiqueta de Release;
los artefactos del workflow son copias diagnósticas temporales.

## Resultados ejecutados — primer smoke Gen376 / semilla43

[GitHub Actions, reconstrucción canónica y replay completos](https://github.com/Asriel550/Forex/actions/runs/42540811163).
[Artefacto JSON con ambos modelos](https://github.com/Asriel550/Forex/actions/runs/42540811163/artifacts/13641422046).
[Release del nuevo universo de 35.261 velas](https://github.com/Asriel550/Forex/releases/tag/data-m5-is-2026h1-full-universe-v1).

La reconstrucción real produjo **35.261 M5 elegibles** y confirmó sin
diferencia las **9.968 propuestas originales** del snapshot PR35 (SHA256 del
universo completo `d8de896967ec419b89e92be61b0f6cad45165ddff297b8353fbf936db53a0d8d`).
No se accedió a Q3.

**Replays a p85 (Gen375/43 de dirección congelada):**

| Métrica | Entrada primaria original | Gen376/43 nueva cabeza |
| --- | ---: | ---: |
| Propuestas antes de p85 | 9.968 | 11.247 |
| Gate p85 acepta antes del cupo | 1.447 | 1.630 |
| Capacidad upstream/h después del cupo | 1,086 | 1,176 |
| Q1 PnL ejecutado | −$194 | −$160 |
| Q1 Profit Factor | 0,476 | 0,667 |
| Q1 frecuencia operativa/h | 0,042 | 0,062 |
| Q1 drawdown | 20,51% | 20,45% |
| Q2 PnL ejecutado | −$146 | −$164 |
| Q2 Profit Factor | 0,810 | 0,809 |
| Q2 frecuencia operativa/h | 0,106 | 0,120 |
| Q2 drawdown | 20,63% | 20,08% |
| Semestre PnL con cuenta continua | −$194 | −$160 |
| Semestre frecuencia **operativa**/h | 0,020 | 0,030 |

El aumento de cobertura es real pero **no recupera la rentabilidad**. El
genoma Gen376 del smoke baja el umbral de margen dinámico hasta 2,796 y
mantiene dirección, cupo y stop. Aún así, ambos trimestres activan el stop
con drawdown mayor del 20%. El PF Q2 disminuye marginalmente; por tanto,
no puede afirmarse que el volumen adicional preserve la calidad.

La evolución conservó población **128**, una generación y semilla43.
El fitness IS aumentó de -1,115 a -0,890, pero esos scores no sustituyen
los criterios operativos. **0 candidatos aptos; ninguno promovido.**

El objetivo contractual **≥1,2 contratos/h ejecutados**, PF>1,20 en
ambos trimestres y DD<15% **NO SE CUMPLE**. La capacidad máxima antes
del gate (1,176/h) también queda por debajo de 1,2/h, y el gate p85
reduce aún más admisiones. El criterio de densidad no debe relajarse
para declarar éxito ni se debe acceder a Q3 para buscar otro resultado.

**Recomendación de continuación:** mantener esta PR como investigación
para una cabeza upstream realmente entrenable sobre calidad de
propuestas, ampliar el universo con información causal adicional
validada, y repetir controles IS antes de cualquier evolución extendida.
