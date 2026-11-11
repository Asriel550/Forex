# Forex — laboratorio evolutivo M5 + M15

Investigación cuantitativa de opciones binarias sobre cuatro pares FX
(EURUSD, GBPUSD, AUDUSD, USDCAD). El proyecto activo es un laboratorio
evolutivo de genomas de 22 genes que combina señal M5, contexto M15 y
volatilidad GARCH causal. No coloca órdenes ni se conecta a un broker.

## Alcance actual

- Señal en M5 y contexto M15; H1/H4 quedaron descartados por ruido.
- Genomas de 22 genes: 8 inputs por timeframe (mercado + contexto).
- Cuatro pares y semillas 42/43/44.
- 24 genomas congelados para la auditoría OOS Q3 2026: 12 controles PR42
  y 12 evolucionados PR43.
- Fundador embebido en `config/evolution_m5_seed_gen20.json`.
- Sin Telegram, sin PPO y sin entrenamiento con PyTorch.

## Estructura

```text
src/            núcleo: evolución, GARCH/MC, entorno, decisión, replay, métricas
scripts/        linaje PR41–PR45 y experimentos M15 Control/Feature/Anchor
tests/          pruebas deterministas y sintéticas
docs/           protocolos PR44/PR45 y contrato de drift
config/         fundador evolutivo y cierres de datos
datasets/       manifiesto HistData y procedencia
```

El núcleo no depende de `stable-baselines3`, `torch` ni `tensorboard`.
`gymnasium` se usa solo para el contrato del entorno.

## Entorno local

Python 3.11/3.12. El entorno validado se congela con `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

En esta máquina se validó el stack sin PPO: la suite local completa pasa
con 277 pruebas.

## Auditoría OOS Q3 2026 (PR44)

El protocolo está pre-registrado en
`docs/research/pr44_q3_single_pass_protocol.md` y se ejecuta con:

```bash
python -m scripts.run_pr44_q3_single_pass_audit \
  --control-root <artefactos PR41> \
  --evolved-root <artefactos PR43> \
  --q3-root <universos Q3>
```

Evalúa una sola vez los 24 genomas congelados sobre Q3
(2026-07-01T00:00Z a 2026-10-01T00:00Z, fin exclusivo), sin evolucionar,
seleccionar, reajustar ni cambiar pesos. Criterios simultáneos:
PF Q3 > 1.15, drawdown < 15 %, degradación < 40 % frente al PF H1
reconstruido y al menos una operación ejecutada. Si falta cualquier
universo Q3, el script falla; no inventa métricas.

Los cuatro universos Q3 nativos (M5/M15/GARCH) están publicados como
Releases `data-pr44-q3-*`. La auditoría todavía no se ejecutó en GitHub
Actions: los runs fallaron por límite de facturación de la cuenta. Puede
correrse localmente con los assets de esas Releases.

## Experimento M15

La dirección M15 no es un feed independiente: se agrega causalmente desde M5.
El experimento compara tres brazos sobre los mismos datos:

```bash
python -m scripts.run_m5_direction_ablation \
  --seed-bundle <bundle> \
  --m15-arm control \
  --m15-arm feature \
  --m15-arm anchor
```

## Datos

- `datasets/eurusd/manifest.json`: procedencia y hashes de las fuentes
  HistData usadas por la preparación local.
- `config/closures.json`: huecos observados del feed; no son afirmaciones
  de mercado cerrado.
- Los ZIP M15 históricos fueron retirados. Los universos de investigación
  se publican como Releases y no se versionan en Git.

## Documentación

- [Contrato de drift causal](docs/dynamic_drift.md)
- [Protocolo PR44 Q3](docs/research/pr44_q3_single_pass_protocol.md)
- [Fuentes nativas PR45](docs/research/pr45_q3_histdata_native_sources.md)
- [PR41: 20 generaciones por activo](docs/research/pr41_independent_20_generations.md)
- [PR42: silos independientes](docs/research/pr42_isolated_silos.md)
- [PR43: 100 entrenamientos](docs/research/pr43_100_independent_trainings.md)

Los resultados son investigación reproducible, no una promesa de rendimiento.
