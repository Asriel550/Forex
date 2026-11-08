# PR #34 — Auditoría reproducible y calibración de fitness IS

## Evidencia previa y alcance

La PR #33 produjo un primer fallo de máscara de marzo en la corrida
[42532836670](https://github.com/Asriel550/Forex/actions/runs/42532836670),
y una reconstrucción posterior exitosa en
[42532891373](https://github.com/Asriel550/Forex/actions/runs/42532891373).
La comparación de commits `9a9fbd8..8cfa5f0` muestra **únicamente
un documento agregado**: el código numérico no cambió entre ambas.
Esto no prueba una carrera; los ZIP originales M1 ya se verifican por SHA256
y el Monte Carlo usa una semilla local. Un refit GARCH/entorno numérico o una
diferencia de reconstrucción son hipótesis que requieren huellas por etapa.

## Auditoría obligatoria antes de nuevas generaciones

El workflow `PR34 audit IS independiente y fitness calibrado` ejecuta
**dos runners limpios** que reconstruyen enero-junio 2026 a partir de los
mismos seis ZIP M1 por SHA256, con versiones fijadas de NumPy, SciPy,
pandas y arch y un único hilo BLAS. Cada runner registra:

- SHA256 de los seis ZIP origen y de los valores/timestamps del feed M1.
- SHA256 de timestamps y OHLCV M5 **agrupado desde M1 completo**, sin
  inventar velas.
- SHA256 de la serie GARCH/Monte Carlo (volatilidad y probabilidades),
  versiones y reajustes GARCH retenidos.
- Cantidades y hashes de las admisiones por mes; en marzo exactamente
  **2.859 propuestas** y hash congelado. Ante discrepancias se enumeran
  timestamps faltantes y nuevos.
- SHA256 de los inputs normalizados que consume la red y sus outcomes.

El primer paso que diverja permitirá ubicar la causa. El flujo aborta antes
de entrenar si hay una diferencia, incluso si coincidieran PnL finales.

Los archivos `a.npz`/`b.npz` y manifiestos `a.json`/`b.json`
quedan guardados como artefactos. Son **snapshots IS con SHA256 verificado**:
la evolución los lee sin volver a ajustar GARCH. La descarga de cada ZIP
se protege con un lock entre procesos y una sustitución atómica; la auditoría
sigue verificando SHA256 incluso para archivos recuperados de caché.
Por diseño no se accede a Q3 (julio-septiembre).

El código original de admisiones mantiene el antiguo cutoff protegido por
defecto; solo los diagnósticos IS declaran el cutoff junio 2026.

## Diagnóstico del problema de densidad

En el smoke previo la Gen 376/43/p80 produjo 0,099 contratos/h en Q1 y 0,417
en Q2 bajo el riesgo **operativo**, con parada de cuenta en ambos trimestres.
Eso no equivale a la densidad de admisión del gate antes del stop. En el
replay sellado de la PR #32, la Gen 375/p80 mostraba ~0,550/h en Q1
y ~0,439/h en Q2 **antes** de aplicar el stop.

El límite físico del universo de propuestas y cupo horario fue **1,226/h
en Q1 y 0,951/h en Q2**; la aspiración de 1,2/h sigue siendo **imposible
en Q2** con las propuestas actuales. Ninguna calibración de pesos podrá
crear señales nuevas.

## Fitness calibrado y trazable (v2)

Se elimina la desproporción anterior: retornos de orden 0,003 quedaban
eclipsados por un término `loss * 0,12` con loss ~1, y una penalización
binaria por stop de 0,15. Ahora el retorno se expresa en **puntos porcentuales
(100×)**, la pérdida pesa 0,025, y el stop pesa 0,05, con riesgo por drawdown.

Para cada trimestre se calcula:

```text
r_real = pnl_ejecutado / (stake × todas_las_propuestas)
r_gate = pnl_de_admisiones_gate_sin_stop / (stake × todas_las_propuestas)
r = 0.40 × r_real + 0.60 × r_gate
fitness = 100 × (0.45 × mean(r) + 0.55 × min(r)) - penalizaciones

penalizaciones =
  0.25 × std(100 × r)
+ 0.10 × var(100 × r)
+ 0.16 × max(drawdown_fracción)
+ 0.06 × mean(drawdown_fracción)
+ 0.08 × (max(0, drawdown_fracción - 0.15) / 0.15)² máximo
+ 0.025 × loss_tanh
+ 0.12 × mean(shortfall_de_densidad_factible)
+ 0.05 × fracción_de_trimestres_con_stop
```

La **densidad de entrenamiento** se mide en el mismo gate antes del stop,
con el cupo aplicado. Su objetivo suave es
`min(1.2, capacidad_física_del_trimestre)`, para no castigar por no
poder realizar operaciones inexistentes. La frecuencia operativa post-stop
y el PnL verdadero siguen registrándose para el resultado final.

El **criterio de promoción no cambia** y se evalúa sobre resultados
operativos en **ambos** trimestres: PF>1,20, 1,2 contratos/h, DD<15%,
PnL positivo y ningún stop. Como Q2 tiene techo 0,951/h, ningún campeón
podrá marcarse elegible para promoción hasta ampliar el universo de
propuestas: no lo ocultamos.

## Garantía de selección y próximas pruebas

Se conserva la población 128, las semillas 42/43/44, mutación, cruce,
elitismo, la red tanh y el Gate de Convicción Pura. La loss de
falsos positivos en volatilidad elevada sigue siendo solo objetivo de
entrenamiento, no filtro de admisión.

La prueba por defecto es una generación y una semilla/percentil; 21
generaciones requieren **snapshots previamente auditados y sellados**.
No se habilita entrenamiento extendido desde el cargador dinámico.
Los resultados IS no son validación externa y Q3 sigue sin examinarse.
