# PR #32 — diagnóstico semestral EUR/USD M5 y OOS aislado

## Alcance

Esta PR **no inicia nuevas generaciones ni reentrena**. Evalúa los genomas
sellados Gen 354/43 y Gen 375/43 procedentes de la PR #31, con el Gate de
Convicción Pura causal en p85, p80 y p75. El techo externo de volatilidad
permanece retirado. La gestión de riesgo conserva el cupo horario y el stop
de cuenta después del gate.

| División | Fechas UTC | Uso |
| --- | --- | --- |
| Diagnóstico Q1 | 2026-01-01 a 2026-04-01 exclusivo | Replay retrospectivo |
| Diagnóstico Q2 | 2026-04-01 a 2026-07-01 exclusivo | Replay retrospectivo |
| Cuarentena Q3 | 2026-07-01 a 2026-10-01 exclusivo | No cargar, evaluar ni optimizar en PR #32 |

**No llamamos OOS genuino a Q1/Q2**: marzo fue el tramo optimizado, enero/febrero
se han estudiado previamente y abril ya fue observado en las PR anteriores.
El protocolo histórico anterior de PR #9 protegía el tramo desde
2026-04-07T15:15Z; este nuevo diagnóstico entra deliberadamente en ese periodo
ya conocido. Por tanto, no modifica ni invalida las garantías previas, pero
**tampoco puede presentarse como prueba OOS del protocolo original**.

Q3 solo será OOS si una auditoría posterior confirma ausencia de uso previo para
selección de hiperparámetros, filtros o genomas. La implementación se detiene en
el 1 de julio y no solicita ni descarga las fuentes M1 de Q3.

## Fuentes, causalidad y reproducción

- Se descargan exclusivamente los seis ZIP **M1 enero-junio 2026** identificados
  por SHA256 en `datasets/eurusd/manifest.json`. Se agregan a M5 solo grupos
  completos, sin interpolar cotizaciones ni atravesar huecos.
- Se comprueba la **máscara exacta de marzo** (2.859 propuestas y SHA256 ya
  archivado). Si cambia, el replay falla: no se permiten comparaciones con
  propuestas distintas como si fueran equivalentes.
- Se usa la normalización y los genomas *sellados*; ninguna etapa ajusta parámetros
  a Q2 ni selecciona campeones usando datos semestrales.
- El gate arranca sin un estado anterior a enero y necesita **500 propuestas**
  antes de admitir contratos. Se contabilizan esas abstenciones y las horas
  observadas: el arranque reduce la tasa agregada y debe declararse.
- Cada decisión usa solo scores previos; las liquidaciones se incorporan cuando
  vence el contrato, incluso si la propuesta fue descartada. El estado del gate
  permanece continuo de Q1 a Q2.
- La cuenta se reinicia al evaluar Q1 y Q2 individualmente; además se calcula el
  semestre con una **única cuenta y stop continuo**. El informe distingue ambas
  convenciones de riesgo.

## Métricas y aceptación

En `semester.json` se registran para cada genoma y percentil: PnL, PF,
contratos/hora observable, drawdown, acierto, admisiones, capacidad sin filtro de
convicción pero con cupo, cortes distintos, estados elásticos y causas de rechazo.
En `admissions.csv` se conserva el detalle de contratos del semestre.

Objetivos de investigación, **no supuestos de éxito**:

- Profit Factor **estrictamente > 1.20**.
- Densidad **>= 1.2 contratos/hora**.
- Drawdown **< 15 %**.
- Verificar estabilidad en **ambos trimestres**, no solamente en el agregado.
- Comprobar que el percentil móvil cambia autónomamente. La apertura
  excepcional hasta p0 sigue exigiendo evidencia de maestría.

Un CI verde demuestra que el programa y sus sellos funcionan, **no** que el
modelo cumpla objetivos financieros ni que las simulaciones representen
precios ejecutables de un bróker.

## Próxima fase

Si los diagnósticos demuestran estabilidad suficiente, definir una prueba OOS
auditada en Q3 sin reoptimizar después de observar resultados. La transferencia
posterior a otra divisa principal se hará con fuente independiente y auditoría
de calidad, separando dos experimentos: (1) aplicar los pesos congelados sin
reentrenar; (2) evolución nueva con las mismas semillas 42/43/44 y las mismas
reglas de gate y riesgo. No hay resultados de portabilidad en esta PR.
