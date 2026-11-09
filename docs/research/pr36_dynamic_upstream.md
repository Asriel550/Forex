# PR #36 — Cabeza evolutiva de entrada EUR/USD M5 + M15

## Corrección explícita del experimento

La estrategia trabaja **únicamente con señales M5 y M15**. La primera
implementación de PR36 había introducido indebidamente 4 proxies M1. Esa
variante y sus resultados de Gen376 (11.247 propuestas y pérdidas
trimestrales) **se retiran como evidencia para esta estrategia**: no son
directamente comparables con el modelo solicitado M5+M15.

**M1 solo sirve para descargar histórico y agregar velas completas OHLCV
M5/M15** con las mismas verificaciones SHA256 del repositorio. No existen
inputs M1 en la cabeza genética corregida. M5 se decide al cerrarse su
vela; M15 se usa exclusivamente si la vela de 15 minutos ya está cerrada.
Una M15 en formación no se consulta.

## Datos, integridad y aislamiento

- El snapshot original PR35 GARCH/MC tiene SHA256
  `7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504`.
- El universo anterior v1 de 35.261 velas elegibles tenía SHA256
  `d8de896967ec419b89e92be61b0f6cad45165ddff297b8353fbf936db53a0d8d`,
  pero llevaba proxies M1 y **no vuelve a emplearse directamente para evolucionar**.
- Se deriva un universo **v2** conservando intactos todos los arrays
  operativos: timestamps, outcomes, volatilidad GARCH, 32 inputs de dirección,
  scores de entry, márgenes y máscara legacy de **9.968 propuestas**.
- Se sustituyen los proxies antiguos por ocho señales exclusivas de M5/M15.
  La migración usa OHLCV agregado a M5/M15 desde el histórico M1 original,
  pero **sin extraer rasgos ni tomar decisiones con velas M1**.
- El origen v1 se verifica por SHA256 binario, el M5 regenerado debe coincidir
  con la huella OHLCV canónica de PR35; v2 se publica **solo si pasa todas
  las verificaciones**, en una Release nueva e inmutable
  `data-m5-m15-is-2026h1-full-universe-v2`.
- La evolución y cada CI posterior usan v2 sin volver a ajustar GARCH.
  Q1 y Q2 siguen siendo IS. Q3 (julio–septiembre 2026) permanece sin leer.

## Ocho señales de entrada permitidas

Se calculan exclusivamente sobre velas completas:

- M5: cuerpo relativo, posición del cierre dentro del rango, rango
  normalizado y cuerpo de la vela M5 inmediatamente anterior.
- M15: cuerpo relativo, posición del cierre, rango normalizado y cuerpo
  de la última vela M15 completa/anterior.

La cabeza utiliza ocho pesos, sesgo y dos genes del margen dinámico,
que depende de la volatilidad GARCH de M5 ya congelada.
El genoma cero reproduce exactamente los filtros legacy.
**El gate p85 no cambia.** La Gen375/43 de dirección se mantiene congelada.

## Condiciones para aceptar los resultados

Cada candidato se evalúa con la misma simulación de contratos M5,
payout 80%, cupo horario, gate p85 causal y stop original.
Se publica Q1, Q2 y el semestre de riesgo continuo:

- Contratos ejecutados del semestre **>=1,2/h**.
- Profit Factor **>1,20** en **ambos** trimestres.
- Beneficio positivo en ambos, drawdown **<15%** y sin stops.
- Verificación de sello de datos, integridad de la máscara legacy y
  aislamiento absoluto de Q3.

El CI debe publicar datos reales y marcar la búsqueda como **no aceptada**
si falla cualquier condición. Un CI verde solo certifica funcionamiento
técnico, no rentabilidad. No se ejecutan 21 generaciones sin pruebas
robustas y revisión posterior.


## Smoke corregido M5+M15 (Gen376, semilla43, p85)

[CI completo aprobado](https://github.com/Asriel550/Forex/actions/runs/42542842719).
[JSON íntegro](https://github.com/Asriel550/Forex/actions/runs/42542842719/artifacts/13642003115).
[Release M5+M15 v2](https://github.com/Asriel550/Forex/releases/tag/data-m5-m15-is-2026h1-full-universe-v2)
(SHA256 `05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9`).

La migración verificó **35.261 velas elegibles** y las **9.968 propuestas
originales** sin alterar outcomes, dirección, GARCH ni máscaras históricas.
Ninguna feature M1 se trasladó al modelo. Se probaron 128 candidatos.

| Métrica | Filtro original | Cabeza M5+M15 Gen376/43 |
| --- | ---: | ---: |
| Propuestas antes del gate | 9.968 | **12.741** |
| Aceptadas por gate p85, antes del cupo | 1.447 | **1.850** |
| Capacidad upstream/h, antes del gate | 1,086 | **1,230** |
| Q1 PnL / PF | −$194 / 0,476 | **−$158 / 0,713** |
| Q2 PnL / PF | −$146 / 0,810 | **−$154 / 0,873** |
| Q1 frecuencia realmente ejecutada/h | 0,042 | **0,073** |
| Q2 frecuencia realmente ejecutada/h | 0,106 | **0,174** |
| Semestre, una cuenta (PnL / frecuencia) | −$194 / 0,020/h | **−$158 / 0,036/h** |
| DD Q1 / Q2 | 20,51% / 20,63% | **20,11% / 20,04%** |

La cabeza aumenta la **capacidad pre-gate** hasta superar 1,2/h,
pero **no** alcanza 1,2/h ejecutados después del gate y el stop.
La mejora de PF es insuficiente: ambos trimestres siguen por debajo
de 1,20 y tocan el stop por drawdown superior al 20%.
Por tanto **0 candidatos aptos; ninguno promovido**. El CI técnico está
verde, pero los criterios financieros **no**.

Las cifras iniciales de la variante M1 quedan anuladas para este protocolo.
No usar estos resultados IS como prueba OOS o de trading real.
