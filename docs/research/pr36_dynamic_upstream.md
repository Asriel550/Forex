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
