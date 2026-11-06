# Escala de drift y umbral operativo de investigación

Se añade drift_scale explícito, con default **1.5**. El log-drift pasa a ser:

```text
excess = sign(mean) * max(abs(mean) - drift_noise_z*SE, 0)
log_mu = clip(drift_scale * drift_shrinkage * excess,
              -drift_cap_sigma*sigma, +drift_cap_sigma*sigma)
mu = log_mu + sigma²/2
```

Se conserva la banda de ruido 1.96*SE, shrinkage 0.75, ventana de ocho retornos,
clip de retornos ±3 sigma y tope de drift ±0.75 sigma. El multiplicador amplifica
una señal ya filtrada, sin cambiar la semilla, las 10.000 trayectorias, los 32
pasos, la causalidad ni la política de reinicio ante huecos. Su rango permitido
es (0,3]. Más escala modifica las probabilidades del modelo; no aumenta por sí
misma la frecuencia real de aciertos.

El default del filtro/entrenamiento cambia de 70% a **60%**. Se pueden comparar
58%, 60%, 62% y 65%, siempre con desigualdad estricta y la ventaja conservadora
por payout. Para payout 0.80, el acierto de equilibrio entre contratos resueltos
sin reembolso es **1/(1+0.80) = 55.56%**. Un porcentaje MC por encima de esa cifra
no demuestra que el acierto observado vaya a superarla. Wilson sigue descontando
ruido de muestreo MC; min_expected_edge sigue siendo 0.02. El drawdown de stop
sigue siendo 20%, liquidado al expirar; el último contrato puede sobrepasarlo.
No se fuerza a PPO a operar ni se premia una pérdida como ganancia.

## Ejecución controlada desde GitHub Actions

En Run workflow elegir main, mode=train, confidence_threshold (0.58/0.60/0.62/0.65)
y drift_scale (1.0/1.5/2.0). Defaults: 0.60 y 1.5. Para push/cron se admiten las
variables FOREX_MIN_CONFIDENCE y FOREX_DRIFT_SCALE; sin ellas se usan los defaults.
El paso quant recibe la escala y train recibe el umbral. Ambos quedan congelados
en el Parquet/manifiesto del bundle. mode=alerts conserva los parámetros de su
bundle; los controles del formulario no reinterpretan pesos existentes.

Bundles anteriores que no registraban drift_scale usan **1.0**, aunque el nuevo
default sea 1.5. Sus RiskConfig archivados conservan el umbral original. Para usar
la nueva configuración es necesario regenerar features, normalizador y PPO.

```bash
python -m src.quant_engine --drop-warmup --drift-scale 1.5
python -m src.sensitivity --scales 1 1.5 2 --thresholds .58 .60 .62 .65
python -m src.train --total-timesteps 100000 --min-confidence .60
python -m src.backtest
```

## Sensibilidad sin leer resultados del holdout

src/sensitivity.py compara las 12 combinaciones en **train y validation** por
separado, con los mismos límites cronológicos/purga de train.py. Utiliza las
estadísticas causales de momentum archivadas para reproyectar cada escala;
no desplaza ventanas ni reajusta GARCH con futuro. No evalúa resultados de test
ni elige automáticamente un ganador. El script admite otros parámetros de split
y preparación mediante su función Python; el workflow usa los defaults comunes.

La política de diagnóstico propone la dirección MC más probable. Es un
**baseline MC, NO el PPO**. Su caja vectorizada se compara en pruebas con el
entorno real: payout, empates, huecos, saldo/pico y stop de riesgo coinciden.
Esto permite obtener una muestra diagnóstica sin afirmar que el agente entrenado
va a imitarla. PPO puede seguir absteniéndose si no aprende una ventaja suficiente.

Se generan reports/sensitivity.json, sensitivity.csv y sensitivity.md. Contienen
contratos, PUT/CALL, neutros, candidatos anteriores al stop, PnL, PF, drawdown,
stop, acierto observado, probabilidad MC media y Brier de contratos evaluados.
También informan el equilibrio por payout y un límite Wilson nominal del acierto
observado; esa cota binomial no corrige pruebas múltiples ni dependencia temporal.
El job de investigación ejecuta este diagnóstico antes de PPO y publica los tres
formatos junto a los reportes habituales.

Los reportes de evaluación/PPO/backtest incorporan segmentos estrictamente >58%,
>60%, >62%, >65%, >70% y >80%. Incluyen PF, PnL, cantidad, acierto sin reembolsos,
edge observado por stake resuelto, equilibrio por payout y cota nominal del acierto.
Son subconjuntos de contratos ejecutados; no una reejecución de PPO con cada filtro.
Sin contratos: null/no_trades; sin pérdidas: null/infinite_no_losses, nunca infinito
serializado ni un supuesto éxito estadístico.

## Resultado de investigación sobre el snapshot existente

Archivos completos: [JSON](research/sensitivity.json), [CSV](research/sensitivity.csv)
y [Markdown](research/sensitivity.md). Estos resultados no seleccionaron los
defaults: escala 1.5/umbral 0.60 se fijaron antes de correr la matriz.

Baseline MC, validación, escala 1.5:

| Umbral | Contratos | PnL USD | PF | Acierto sin reembolsos |
| --- | ---: | ---: | ---: | ---: |
| >58% | 72 | 90 | 1.333 | 62.50% |
| >60% | 48 | 96 | 1.600 | 66.67% |
| >62% | 31 | 50 | 1.455 | 64.52% |
| >65% | 15 | 30 | 1.600 | 66.67% |

La misma escala 1.5 y umbral 60% en train produjo 82 contratos, -US$64 y PF 0.84.
Por tanto no se afirma estabilidad ni rentabilidad fuera de muestra. Para los
48 contratos de validación, el límite Wilson nominal inferior del acierto es
52.54%, por debajo del equilibrio de 55.56%; el PF puntual 1.60 no confirma ventaja.
Las muestras pequeñas de la matriz no deben usarse para elegir un ganador por PF.

El holdout histórico ya fue examinado en versiones anteriores y no se usa aquí
para afinar escala/umbral. Estas comparaciones son investigación; una mejora
requiere confirmación en datos futuros sin utilizar o un protocolo walk-forward
predefinido. El snapshot tampoco proporciona cotizaciones actuales para Telegram.


## PPO experimental separado del baseline

Se entrenó localmente un PPO de 100.352 pasos con escala 1.5 y umbral 60%, usando
solo train y validación para aprender/seleccionar. No se evaluó de nuevo el holdout.
El checkpoint elegido fue el paso 25.000: un CALL, +US$8, 26.394 NEUTRO en
validación. Los checkpoints 50.000, 75.000 y finales ejecutaron cero contratos.
Es evidencia de que la ejecución ya está permitida, no de una estrategia rentable:
un acierto aislado no permite estimar PF ni calibración. La política sigue siendo
muy defensiva; no se alteraron las penalizaciones para forzar una muestra mayor.
Resumen reproducible: [PPO de investigación](research/ppo_validation_summary.json).
Los pesos locales no se publican como bundle de producción en esta entrega.
