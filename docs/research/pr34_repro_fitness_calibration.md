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

## Causa raíz confirmada por reconstrucciones forenses

La [corrida 42535814697](https://github.com/Asriel550/Forex/actions/runs/42535814697)
ejecutó dos procesos independientes con los mismos paquetes y fuentes.
Encontramos **idénticos SHA256 de los valores M1 y M5**, pero diferente
resultado en GARCH/Monte Carlo:

- Runner A: GARCH/MC `7e796d512a...`, **ningún refit retenido**, 2.859
  propuestas de marzo (hash exacto del linaje).
- Runner B: GARCH/MC `1683798b3c...`: el optimizador GARCH no convergió
  en la **fila 16.928**, reutilizó los parámetros anteriores y rechazó
  la propuesta M5 abierta **2026-03-16 11:55 UTC**. Resultado: 2.858.
- Por tanto **la variación no nació en los ZIP ni en las velas M5**; la
  diferencia es un refit GARCH numéricamente inestable. No se ha probado
  una race condition en los datos, y el lock de adquisición ahora evita
  escrituras concurrentes del mismo ZIP.

## Política de congelación y doble comprobación

Se preservó el snapshot IS de la reconstrucción A como único input
cuantitativo para la evolución temporal:

- [Artefacto canónico de Actions](https://github.com/Asriel550/Forex/actions/runs/42535814697/artifacts/13639251699).
- SHA256 binario `7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504`.
- SHA256 del bloque GARCH/MC `7e796d512ad12d5a546e8b6dacf94e72dd2c23ff7639efc65737687cbe037ef6`.

**Cada runner nuevo reconstruye M1 y M5 independientemente** desde los seis
ZIP oficiales comprobados por SHA256 y compara timestamps y OHLCV con ese
snapshot. **No vuelve a ajustar GARCH ni a regenerar los inputs neuronales**.
La evolución solo puede arrancar cuando ambos informes de fuentes coinciden
y el archivo binario verifica su hash. Esto elimina la dependencia del éxito
aleatorio del reajuste GARCH durante esta búsqueda, sin afirmar haber
resuelto el algoritmo numérico para futuras fechas.

Las 2.859 aperturas históricas de marzo, su hash, el genoma fundador y el
intervalo IS (enero-junio) se verifican antes de entrenar. Q3
(julio-septiembre) permanece fuera del pipeline. El cutoff protegido
de tareas anteriores sigue siendo el predeterminado.

**Retención:** el snapshot proviene de un artefacto de Actions temporal.
Antes de depender de él a largo plazo debe archivarse de forma duradera,
preservando los mismos hashes; no reoptimizar usando otro snapshot sin
abrir un protocolo experimental distinto.

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
