# PR43 — 100 entrenamientos genéticos independientes por divisa (H1 IS, Q3 intacto)

## Alcance y protocolo fijo

**100 entrenamientos NUEVOS por cada divisa**, no 100 generaciones de una
única evolución. Cada entrenamiento realiza 20 generaciones × 128 genomas
(2.560 evaluaciones), con cromosomas de 22 genes inicializados desde cero
y semilla distinta; EURUSD, GBPUSD, AUDUSD y USDCAD: **400 entrenamientos
y 1.024.000 evaluaciones programadas**.

Semillas: enteros 1000 a 1099 en cada divisa, 20 shards × 5 semillas por
par, 80 trabajos aislados. El RNG lleva desplazamientos no superpuestos
por activo. Los 12 campeones PR41 originales (semillas 42, 43, 44) **no se
sustituyen, modifican, reentrenan ni usan como fundadores**; PR42 se conserva.

Cada trabajo lee UNA divisa y su GARCH M5 nativo + señales M5 y M15
cerradas. Conserva el mismo motor genético de PR41, función `evaluate_local`,
p85 y criterios duros: PF >1,20 y PnL positivo en Q1/Q2, DD trimestral
y semestral <15%, sin stops. Mantiene cuenta simulada inicial 1000 USD,
stake 10 USD, payout +0,8/−1 y 2 contratos por hora **por divisa**.
No existe cuenta ni cupo compartido. Cada entrenamiento reanuda cero genes,
no la mejor semilla vecina.

## Datos y protección Q3

**Intervalo permitido cerrado-abierto**: [2026-01-01, 2026-07-01), con
Q1/Q2 2026 únicamente. Q3 de julio, agosto y septiembre sigue en
cuarentena. El método `native_problem` de PR42 verifica estructura, hashes
nativos específicos de activo, meses de H1 y ninguna fila igual o posterior
a 2026-07-01. Las fuentes selladas y SHA se mantienen idénticos a PR41;
no hay descargas de Q3, fuentes M1 como señal, optimización sobre julio,
ni comparación retrospectiva con OOS.

**Importante**: 100 búsquedas del mismo IS aumentan el riesgo de
sobreajuste y de encontrar PF altos por azar. Las tres semillas PR41 no
son validación OOS, ni lo son las nuevas 100. No se selecciona
automáticamente una ganadora para operar y no se abre Q3 para escogerla.
Los resultados se presentan como distribución de la búsqueda IS, no
aprobación de rentabilidad ni prueba de alfa.

## Ejecución reproducible

- Script: `scripts/run_pr43_100_native_trainings.py`.
- Test: `tests/test_pr43_100_native_trainings.py`.
- Workflow: `.github/workflows/pr43-100-native-trainings.yml`.
- 4 activos × 20 shards × 5 seeds; **80 trabajos GitHub Actions**;
  ejecución paralela limitada a 8 shards.
- Cada shard publica sus cinco reportes individuales
  `{symbol}/seed-{1000..1099}.json` y un `shard-NN.json`.
- Consolidador requiere exactamente 400 registros, 20 generaciones y
  128 candidatos/semilla con firmas de genoma y fuente local,
  y genera métricas por divisa (fracción aprobada, frecuencias, PF,
  DD sin mezclar saldos).
- Si falla cualquier job, el consolidado **no declara éxito**;
  ningún resultado parcial se interpreta como 400 terminados.
- Ningún job envía órdenes a brokers, usa Telegram, altera modelos
  actuales o toca Q3. La PR permanece en estado draft.

Ejemplo de un shard local, que requiere los archivos H1 originales de esa
divisa:

```bash
python -m pytest -q tests/test_pr43_100_native_trainings.py tests/test_pr42_isolated_silos.py
python -m scripts.run_pr43_100_native_trainings \
  --symbol GBPUSD --shard 0 \
  --universe reports/pr43-local-source/universe.npz \
  --manifest reports/pr43-local-source/manifest.json
```

Los 100 entrenamientos no garantizan un PF superior ni mayor densidad
operativa. El objetivo actual es medir estabilidad de la optimización
entre inicializaciones independientemente del OOS.
