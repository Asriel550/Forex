# PR39 — GARCH Sweet Spot: ensanchamiento controlado, PF × frecuencia y gate local

**Alcance:** EURUSD M5 (ejecución) + M15 cerrada (contexto), **IS 2026 Q1 y Q2 exclusivamente**.
Q3 julio–septiembre 2026 permanece intocado. Este protocolo no aprueba trading real.

## Antecedentes cuantitativos

La [PR38](https://github.com/Asriel550/Forex/pull/38) encontró en la cabeza
`garch_regime` un avance real del PF IS, manteniendo p85:

| Semilla | Gen PR38 | PF Q1 | PF Q2 | Ejecutadas/h H1 | DD H1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 10 | 1.272 | 1.224 | 0.2083 | 6.40% |
| 43 | 15 | 1.231 | 1.173 | 0.2011 | 7.47% |
| 44 | 15 | 1.255 | 1.210 | 0.2209 | 8.27% |

El objetivo nuevo es demostrar simultáneamente, **en una misma
configuración para las tres semillas**, PF **estrictamente >1.20** tanto en
Q1 como Q2, DD semestral **<12%**, frecuencia ejecutada H1 **>=0.50/h**
(con preferencia por 0.50–0.80/h o superior si PF y DD se sostienen), PnL
Q1/Q2 positivo y ausencia de paradas de cuenta. *No* basta capacidad de
entrada o densidad anterior al stop.

**Datos:** GARCH snapshot
`7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504`
y universo completo M5+M15 v2
`05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9`.
La Release v1 tenía proxies M1 retirados; no son señales del modelo.
Ningún workflow vuelve a ajustar GARCH. Se comprueba la reproducción
exacta del campeón anterior por seed (PF Q1/Q2, PnL trimestral, frecuencia,
DD y SHA256 del genoma) antes de las perturbaciones.

## Diseño: tres fases secuenciales con controles

### A — Grid de apertura de banda GARCH, genoma original fijo

Para un GARCH de M5 dado, se calcula el cociente entre la volatilidad
actual al cierre de M5 y la mediana de las **48 observaciones anteriores**.
El genoma original define las cotas de la banda GARCH operable:

`low = 0.35 + sigmoid(gene[10])`, `high = 1.2 + 2*sigmoid(gene[11])`
(indices relativos a la cabeza direccional).

El barrido aplica cinco relajaciones **relativas a las cotas**, sin modificar
genes ni p85: 0%, +7,5%, **+15%**, +22,5% y +30%.

Por ejemplo, +15% significa `low_nuevo = low_original × 0.85` y
`high_nuevo = high_original × 1.15`. El número de registros dentro de
la banda es monótono con la relajación, pero el número admitido por el gate
y el PF no tienen por qué serlo. Se miden además los números de propuestas,
admisiones por gate, frecuencia **ejecutada** y DD semestral.

**Hipótesis A:** bandas adyacentes pueden contener nuevas señales con PF
rentable; si sube frecuencia y cae PF, no hay justificación para abrir más.

### B — Evolución de fitness explícito PF × actividad

Se ensaya la banda **+15% fija** y gate **p85 original**. Cada seed
continúa desde el mejor genoma PR38 ya sellado, con **12 generaciones
locales × 128 modelos = 1.536 por seed**, total **4.608 evaluaciones
genéticas adicionales**.

Fitness de exploración:

`PFgeo = sqrt(min(PF_Q1,3) × min(PF_Q2,3))`

`producto = PFgeo × (frecuencia_ejecutada_H1 / 0.50)`

`fitness = producto - 0.55×max(0, (DD_H1-12)/12) - 0.12×[max(0,1.2-PF_Q1)+max(0,1.2-PF_Q2)]`

Con 0 contratos la propuesta puntúa como no apta; los genomas con
menos de 501 propuestas son descartados en el motor. PF se limita a 3
**solo en el fitness de selección**, no en el reporte financiero ni
en la aprobación. La multiplicación simétrica hace que +10% relativo en
PFgeo o +10% relativo en frecuencia aporten igual factor al producto;
la penalización de DD y el requisito duro de aceptación impiden calificar
automáticamente una variante de PF muy bajo. No se modifica el contrato
M5 de +80%/−100% ni la gestión de riesgo.

Los genes se optimizan con el mismo operador de población genética del
repositorio; esto *no* es backprop. Las generaciones B se numeran G1–G12
**localmente**, no como continuación Gen498 en el linaje histórico.

### C — Sensibilidad causal del gate dentro de la misma banda

Se congela el **campeón B por seed** y la misma banda GARCH +15%, y se
comparan cuatro gates sin más evolución: p85, p80, p75 y uno causal
`regime_dynamic`. Este último aplica p75 en el núcleo central
25%–75% de la banda, p80 en los hombros 10%–25% y 75%–90%, y p85
en las bandas periféricas. La convicción se estima con ventana de 500
propuestas anteriores, sin scores futuros ni datos OOS. También se
mantiene el tratamiento del historial de liquidaciones de la política
original.

**Hipótesis C:** exigir más convicción en bordes y menos en el núcleo
rentable puede añadir actividad sin degradar PF.

## Comparabilidad y decisión

Los tres experimentos comparten los mismos IS, payoff, account stop,
límite horario, datos M5/M15 y la misma historia GARCH congelada.
El A compara un único genoma fijo entre bandas; el C compara un genoma
congelado entre gates. Solo B cambia los pesos. El resultado de B no
puede atribuirse exclusivamente a la banda. Se publican por seed y
variante PF Q1/Q2, PnL, ejecución real H1, DD semestral, número de
propuestas y gate admitidas, cotas absolutas y aceptación.

La comparación final **solo** declara aceptación si la **misma variante**
(A con la misma ampliación, B +15%/p85, o C con el mismo gate) califica
en semillas **42,43 y 44**. Una mejora aislada no autoriza promoción.
El criterio es IS y debe tratarse como exploración afectada por
selección múltiple; no garantiza rentabilidad fuera del semestre.

## Ejecución y fuentes

- [PR39](https://github.com/Asriel550/Forex/pull/39) abierta en borrador.
- [Workflow A→B→C](https://github.com/Asriel550/Forex/actions/workflows/m5-garch-sweetspot-pr39.yml).
- Código: `src/garch_sweetspot.py`,
  `scripts/run_m5_garch_sweetspot.py`,
  `tests/test_m5_garch_sweetspot.py`.
- Padres: [PR38 Fase3](https://github.com/Asriel550/Forex/actions/runs/42551086458);
  artefactos `pr38-phase3-garch_regime-seed-42/43/44` descargados por su
  ID de run, hashes de genoma verificados.
- La salida `reports/m5-garch-sweetspot/seed-*.json` se consolida en
  `reports/pr39-summary.json` con **15 variantes de A, 3 de B y 12 de C**
  (10 variantes únicas × 3 seeds), 4.608 evaluaciones evolutivas nuevas
  más las replays de sensibilidad.
- Los artefactos de GitHub Actions son temporales; los datasets de
  las Releases privadas son persistentes mientras no los borre un admin.

**Estado de conclusión:** pendiente de confirmar a partir del
resumen automatizado del mismo run. Q3 no se consulta para decidir el
punto dulce. No hay promoción ni despliegue automático.
