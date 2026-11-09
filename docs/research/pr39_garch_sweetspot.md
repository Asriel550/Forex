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

**Estado de conclusión:** cerrado sin aceptación financiera; los resultados íntegros constan a continuación. Q3 no se consultó y no se promovió ningún modelo.

## Resultados experimentales auditados: A→B→C completos

**Ejecución final:** [GitHub Actions #42560483604](https://github.com/Asriel550/Forex/actions/runs/42560483604), 3 semillas exitosas y [JSON consolidado](https://github.com/Asriel550/Forex/actions/runs/42560483604/artifacts/13649182171). 12 generaciones ×128×3=**4.608 genomas evaluados** en B; **30 replays reportados** en el grid A/B/C (3 de los 30 repiten intencionalmente el campeón B bajo p85 como control C). CI técnico `SUCCESS`, aceptación financiera `any_fully_qualified=false`, `qualified_variants=[]`, `q3_accessed=false`.

En las tablas «G» significa G10/G15 del campeón original PR38 en A, pero **generación local 1–12** en B y C. «Banda» es el cociente **volatilidad GARCH M5 actual / mediana histórica desplazada** aceptado, con extremos absolutos específicos del genoma de cada semilla. La frecuencia es **realmente ejecutada**, tras gate, cupo y stop, dividida por horas del semestre continuo. El DD es **semestral**.

### Fase A: cinco tamaños de banda, el mismo genoma original y p85

| Variante | Seed | G | Banda GARCH efectiva (σ/mediana) | PF Q1 | PF Q2 | Ejecutadas/h | DD H1 % | Apta |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `A_band_+0pct_p85` | 42 | 10 | 1.007–2.383 | 1.272 | 1.224 | 0.2083 | 6.40 | No |
| `A_band_+7.5pct_p85` | 42 | 10 | 0.932–2.562 | 1.041 | 0.988 | 0.2712 | 12.96 | No |
| `A_band_+15pct_p85` | 42 | 10 | 0.856–2.740 | 1.050 | 0.907 | 0.3226 | 20.70 | No |
| `A_band_+22.5pct_p85` | 42 | 10 | 0.781–2.919 | 1.037 | 0.747 | 0.2461 | 20.11 | No |
| `A_band_+30pct_p85` | 42 | 10 | 0.705–3.098 | 0.897 | 0.808 | 0.1150 | 20.04 | No |
| `A_band_+0pct_p85` | 43 | 15 | 1.011–2.605 | 1.231 | 1.173 | 0.2011 | 7.47 | No |
| `A_band_+7.5pct_p85` | 43 | 15 | 0.935–2.800 | 1.137 | 1.045 | 0.2614 | 11.13 | No |
| `A_band_+15pct_p85` | 43 | 15 | 0.859–2.996 | 1.060 | 0.877 | 0.3076 | 18.76 | No |
| `A_band_+22.5pct_p85` | 43 | 15 | 0.783–3.191 | 0.863 | 0.891 | 0.1001 | 20.47 | No |
| `A_band_+30pct_p85` | 43 | 15 | 0.707–3.386 | 0.865 | 0.820 | 0.1058 | 20.63 | No |
| `A_band_+0pct_p85` | 44 | 15 | 1.011–2.144 | 1.255 | 1.210 | 0.2209 | 8.27 | No |
| `A_band_+7.5pct_p85` | 44 | 15 | 0.935–2.305 | 1.274 | 1.085 | 0.2876 | 12.41 | No |
| `A_band_+15pct_p85` | 44 | 15 | 0.859–2.465 | 1.119 | 1.087 | 0.3390 | 14.85 | No |
| `A_band_+22.5pct_p85` | 44 | 15 | 0.783–2.626 | 0.764 | 1.037 | 0.0596 | 20.80 | No |
| `A_band_+30pct_p85` | 44 | 15 | 0.707–2.787 | 0.800 | 0.855 | 0.0650 | 20.08 | No |

**Conclusión A:** +15% elevó la actividad frente al control a **0,3226/h**, **0,3076/h** y **0,3390/h** en semillas 42/43/44, pero el PF Q2 cayó a **0,907/0,877/1,087**; DD subió a **20,70/18,76/14,85%**, todos por encima de 12%. La ampliación de +7,5% tampoco conserva PF>1,20 en ambos trimestres ni consigue 0,50/h. En +22,5% y +30%, el stop produce incluso **menos frecuencia realmente ejecutada**, aunque la banda upstream sea más ancha. Abrir sin discriminar regímenes adyacentes **no es una forma válida de escalar**.

### Fase B: optimizar PF geométrico × frecuencia, banda +15%, gate p85

| Variante | Seed | G | Banda GARCH efectiva (σ/mediana) | PF Q1 | PF Q2 | Ejecutadas/h | DD H1 % | Apta |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `B_genetic_PF_x_hourly_+15pct_p85` | 42 | 9 | 0.875–2.272 | 1.171 | 0.963 | 0.3111 | 12.39 | No |
| `B_genetic_PF_x_hourly_+15pct_p85` | 43 | 12 | 0.883–3.013 | 1.328 | 1.037 | 0.3219 | 8.75 | No |
| `B_genetic_PF_x_hourly_+15pct_p85` | 44 | 11 | 0.844–2.297 | 1.226 | 1.051 | 0.3720 | 11.22 | No |

**Conclusión B:** los campeones fueron **G9 (seed42)**, **G12 (seed43)** y **G11 (seed44)**, tras 12 generaciones y 1.536 candidatos por semilla. Obtuvieron **0,3111/h, 0,3219/h y 0,3720/h** respectivamente, sin llegar a 0,50/h. Las semillas 43 y 44 redujeron el DD a **8,75% y 11,22%**, pero sus PF Q2 quedaron en **1,037 y 1,051**. La seed42 quedó en DD **12,39%** y PF Q2 **0,963**. El fitness nuevo mejora el compromiso IS respecto a abrir simplemente un 15%, **pero no soluciona el requisito conjunto**.

### Fase C: misma cabeza ganadora B, cuatro gates, banda +15%

| Variante | Seed | G | Banda GARCH efectiva (σ/mediana) | PF Q1 | PF Q2 | Ejecutadas/h | DD H1 % | Apta |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `C_band_+15pct_p75` | 42 | 9 | 0.875–2.272 | 0.926 | 0.740 | 0.1276 | 20.00 | No |
| `C_band_+15pct_p80` | 42 | 9 | 0.875–2.272 | 1.144 | 0.757 | 0.2512 | 20.67 | No |
| `C_band_+15pct_p85` | 42 | 9 | 0.875–2.272 | 1.171 | 0.963 | 0.3111 | 12.39 | No |
| `C_band_+15pct_regime_dynamic` | 42 | 9 | 0.875–2.272 | 1.088 | 0.800 | 0.2617 | 20.42 | No |
| `C_band_+15pct_p75` | 43 | 12 | 0.883–3.013 | 0.945 | 1.014 | 0.1239 | 20.18 | No |
| `C_band_+15pct_p80` | 43 | 12 | 0.883–3.013 | 1.125 | 0.961 | 0.3955 | 15.77 | No |
| `C_band_+15pct_p85` | 43 | 12 | 0.883–3.013 | 1.328 | 1.037 | 0.3219 | 8.75 | No |
| `C_band_+15pct_regime_dynamic` | 43 | 12 | 0.883–3.013 | 1.113 | 1.003 | 0.3761 | 13.84 | No |
| `C_band_+15pct_p75` | 44 | 11 | 0.844–2.297 | 0.981 | 0.770 | 0.1569 | 20.76 | No |
| `C_band_+15pct_p80` | 44 | 11 | 0.844–2.297 | 1.117 | 0.860 | 0.4200 | 20.08 | No |
| `C_band_+15pct_p85` | 44 | 11 | 0.844–2.297 | 1.226 | 1.051 | 0.3720 | 11.22 | No |
| `C_band_+15pct_regime_dynamic` | 44 | 11 | 0.844–2.297 | 1.082 | 0.848 | 0.3325 | 20.33 | No |

**Conclusión C:** cambiar p85 por p80/p75 o por p75/p80/p85 según núcleo/bordes de régimen **no mejoró simultáneamente frecuencia, PF y DD**. La mayor frecuencia del barrido completo ocurrió en la **semilla 44**, variante `C_band_+15pct_p80`, **0.4200/h**, pero con PF Q1 **1.117**, PF Q2 **0.860** y DD **20.08%**, también inválida. En particular, el gate dinámico local no supera de forma robusta al p85 dentro de la banda ampliada.

### Juicio conjunto y limitaciones

**Criterios simultáneos:** PF>1,20 por separado Q1/Q2, **≥0,50 contratos/h ejecutados en H1**, DD H1 **<12%**, sin stops y mismas reglas en **seeds42/43/44**. **Ninguna de las 10 configuraciones nominales** (5 A, 1 B, 4 C; B y C-p85 son control duplicado) consigue aprobar los tres modelos. No se selecciona el máximo PF de una semilla y la máxima frecuencia de otra para fabricar un ganador.

La pérdida de PF al ampliar sugiere que el alfa **está localizado en la banda original**, no uniformemente distribuido en sus regímenes vecinos. Esto es evidencia retrospectiva IS, no una afirmación de estabilidad futura ni prueba OOS. El cupo horario y el stop continúan activos. Una banda más ancha puede aumentar las oportunidades **pre-stop** y, paradójicamente, bajar la actividad **post-stop** por secuencias de pérdidas que agotan la cuenta.

**Incidencia de CI:** la [primera corrida #42560265962](https://github.com/Asriel550/Forex/actions/runs/42560265962) se detuvo antes de evaluar modelos por un *test* que comparaba falsamente diferentes PF/frecuencias al analizar la penalización del DD. Se corrigió la prueba para mantener los demás factores constantes, **sin cambiar ninguna fórmula de entrenamiento ni datos**. La reejecución #42560483604 completó todas las fases y la consolidación.

**Decisión:** PR39 permanece **draft**, sin promocionar genomas, fusionar un cambio de trading ni abrir Q3. El siguiente diseño, si se autoriza, debería examinar los extremos **superior e inferior por separado** y evaluar incertidumbre/estabilidad por bloques dentro de IS antes de aumentar más generaciones. No alterar los objetivos para declarar éxito.
