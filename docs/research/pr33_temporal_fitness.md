# PR #33 — Fitness multitrimestre y pérdida direccional Gen 376+

## Hipótesis

La Gen 375/43 fue seleccionada por desempeño en marzo. El replay de la PR #32
mostró PF inferior a 1 en Q1 y Q2 y una capacidad del semestre de 1,086
propuestas ejecutables/hora con el cupo existente. La nueva búsqueda no
debe seleccionar un genoma por su mejor mes.

## División de datos

- **IS de investigación: enero–junio 2026**, particionado en **Q1 y Q2**.
  Ambos trimestres participan en la función de selección de cada individuo.
  Ya son periodos conocidos y no se presentarán como confirmación OOS.
- **Q3 julio–septiembre 2026: OOS cerrado**. El workflow no descarga sus
  archivos ni entrena, calibra o evalúa sobre ese trimestre. Antes de usar Q3
  en una prueba final, auditar si hubo exploración anterior sobre esas fechas.
- M1 oficial agregado solo en barras M5 completas; seis ZIP de 2026 por
  SHA256 del manifiesto. El hash de las **2.859 propuestas de marzo** debe
  reproducirse.
- Gen 375/43/p80 y normalizador archivados, con SHA256. Se conserva el mismo
  pipeline `evolve_population` (128 individuos, cruce, mutación, elitismo,
  semillas 42/43/44), sin alterar el motor de entrada ni el Gate de Convicción
  Pura. Nuevas generaciones 376–396.

## Fitness para cada individuo y cada percentil

Se calcula el gate p85/p80/p75 usando el historial de las **500 propuestas
anteriores** y liquidaciones causales. Su estado continúa durante el semestre,
pero **la cuenta se reinicia por trimestre** para ver quebrantos de Q1 y Q2
por separado. El cupo y el stop permanecen en la capa de riesgo.

Para cada trimestre `q`, se usa `R_q = PnL_q / (stake * propuestas_q)`,
con denominador fijo que incluye rechazos, `DD_q` como drawdown fraccional,
`f_q` contratos por hora observable y `stop_q` como stop de cuenta.

Con constantes **congeladas antes de usar OOS**, la puntuación es:

```text
fitness = 0.40 * mean(R_q)
        + 0.60 * min(R_q)
        - 0.50 * std(R_q)
        - 0.25 * var(R_q)
        - 0.40 * max(DD_q)
        - 0.20 * mean(DD_q)
        - 0.12 * loss_tanh
        - 0.10 * mean(max(0, 1.2 - f_q) / 1.2)
        - 0.15 * fraction_of_quarters_with_account_stop
```

La aptitud jamás se basa solamente en PnL neto de un mes. Es obligatorio
reportar PF, PnL, drawdown y frecuencia de **cada trimestre**, además del
score. La bandera de elegibilidad exige en **ambos** bloques PF>1.20,
frecuencia>=1.2/h, DD<15%, retornos positivos y ausencia de stop.
Los campeones que no pasan quedan etiquetados como **candidatos de
investigación no promovibles**.

## Loss tanh de entrenamiento

La implementación real no tiene SGD ni una rutina de retropropagación para
esta red: es una red tanh entrenada por neuroevolución. Para preservar
exactamente ese proceso, se incorporó en `src/model_training.py` una pérdida
calculada sobre **todas las propuestas resueltas**, conectada directamente
a la puntuación usada por el algoritmo genético:

```text
direction_loss = mean(softplus(-2 * outcome * tanh(score)))
high_vol_fp    = mean(1[high_vol and wrong_side] * tanh(score)^2)
loss_tanh      = direction_loss + 0.40 * high_vol_fp
```

Reembolsos no cuentan en el término direccional. El régimen de volatilidad
alta se define por el percentil 67 de la volatilidad condicional de **Q1 IS**
y permanece fijo al evaluar Q2. **No se reintroduce un filtro de volatilidad
en la admisión**: es únicamente una penalización de entrenamiento para
prevenir confianza errónea en regímenes agitados.

## Ejecución

El workflow `Gen 376+ robustez temporal Q1-Q2` se prueba en cada push con
un **smoke de Gen 376, semilla 43, p80**. La ejecución completa de
**Gen 376–396, 128 individuos, 3 semillas, 3 percentiles** es manual
(`workflow_dispatch`) para controlar costo computacional.

La ejecución completa puede requerir tiempo sustancial; la corrida de
prueba y CI deben estar verdes antes de intentarla. Se publican pesos,
hashes y métricas sin promoción automática. Si ningún candidato logra
el umbral, el resultado aceptable es documentar que **no se resolvió**
la robustez y que el universo de propuestas existente aún limita la
frecuencia máxima por debajo de 1,2/h.

Una eventual transferencia a GBP/USD es un experimento independiente y
no se incluye en esta iteración.
