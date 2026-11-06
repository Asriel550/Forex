# Entrega 3: opciones binarias M15 y entrenamiento PPO

## Contrato de simulación

La estrategia elige la dirección al cierre de una vela y liquida al cierre de
la siguiente, exactamente 15 minutos después. El timestamp de cada fila del
Parquet es apertura de vela. Por eso, OHLC e indicadores de t se conocen en
timestamp[t]+15 minutos; no se usan como información disponible en su apertura.
Esta es una simulación de entrada al close, sin latencia ni cotización real de
un broker de opciones binarias.

| Acción | Condición ganadora | Recompensa | Cambio de saldo |
| --- | --- | --- | --- |
| 0 PUT | close[t+1] < close[t] | +payout | +stake*payout |
| 1 NEUTRO | No abre contrato | 0 por defecto | 0 |
| 2 CALL | close[t+1] > close[t] | +payout | +stake*payout |
| PUT/CALL perdedor | Dirección contraria | -1 | -stake |
| Empate | close[t+1] == close[t] | 0 por defecto | 0 |

Payout=0.8 significa beneficio neto del 80%, no devolución bruta del 80%.
Una apuesta de USD 10 ganadora aumenta la cuenta USD 8; una perdedora resta
USD 10. El stake se supone reservado al abrir y devuelto al liquidar, sin
observaciones intermedias. No se resta dos veces. Nunca queda una posición
abierta después de step. El stake es fijo y no hay martingala.

`--tie-policy loss` cambia los empates a pérdida de stake. Los empates se
registran separadamente de victorias/derrotas direccionales. La tasa de acierto
de validación usa wins/trades, incluyendo empates en el denominador.
`--inactivity-penalty 0.001` penaliza solo la recompensa de esperar; no genera
un costo monetario ficticio. El valor predeterminado es cero.

## Observación y causalidad

Vector float32: RSI de Wilder/100, ATR de Wilder dividido entre close,
spread_pips/10 (cero si la columna no existe), cond_volatility, dist_p95,
dist_p05, últimos N log-retornos, y seis estados de cuenta:

- balance/initial_balance-1;
- drawdown desde el pico de saldo;
- dirección anterior (-1/0/1), no posición abierta;
- recompensa anterior;
- stake/saldo actual;
- payout conocido.

El vector predeterminado tiene 28 elementos: 6 features base/cuantitativas,
16 retornos y 6 estados de cuenta. La selección de columnas es explícita:
no se incorporan accidentalmente columnas de etiquetas, resultados futuros o
percentiles auxiliares del Parquet. Las features externas deben provenir de
un cálculo causal: el entorno no puede detectar leakage dentro de una columna
que alguien haya preparado incorrectamente.

RSI, ATR y lags usan solo pasado. Se elimina únicamente el warmup inicial;
NaN internos e infinitos se rechazan. No se hace backfill. En entrenamiento,
la media/desviación de las features se estima exclusivamente en train y se
congela para validación/inferencia. Se aplica clipping de mercado a ±10 tras
normalizar. Los estados de cuenta tienen sus propias escalas y no se ajustan
con estadísticos futuros. El normalizador guardado es parte obligatoria de
la política desplegada.

## Huecos y fin de episodio

Solo se ejecutan contratos entre filas separadas exactamente por 15 minutos.
Los huecos o fines de semana dividen el conjunto en segmentos. El close final
de cada segmento liquida el último contrato; no se permite abrir otro allí.
Segmentos de una sola fila no generan decisiones.

- `truncated=True`: fin de segmento/datos o longitud máxima de episodio.
- `terminated=True`: saldo menor al stake o drawdown igual/superior al límite.
- step después de terminar, o antes de reset, lanza RuntimeError.

El entrenamiento empieza episodios aleatorios reproducibles con una cuenta
reiniciada; es el muestreo episódico de entrenamiento, no recapitalización en
la evaluación. La validación recorre los segmentos cronológicamente y transporta
saldo y pico entre ellos. Si se alcanza el límite de riesgo, termina toda la
evaluación, sin volver a fondear una cuenta insolvente.

## Entrenamiento en Linux

```bash
# Runner Linux sin GPU: instalar primero el wheel CPU de Torch.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m src.train --input data/eurusd_m15_features.parquet \
  --total-timesteps 100000 --payout 0.8 --stake 10 --initial-balance 1000
```

CPU PyTorch con un hilo, MlpPolicy de SB3, actor/crítico 64x64 Tanh, PPO,
gamma=0.99, gae_lambda=0.95, clip_range=0.2 y ent_coef=0.01. No se usa LSTM
ni API de terceros. La memoria reciente se aporta con lags. Se fija seed y
se activan algoritmos deterministas; esto no garantiza igualdad bit a bit
entre distintas versiones de Torch/hardware.

Los subconjuntos se forman por orden temporal tras el warmup: 70% train,
20% validación y 10% test reservado. Por defecto se descarta una fila adicional
al comienzo de validación y test. Ningún contrato tiene su entrada y expiración
en subconjuntos distintos. Los indicadores de validación pueden usar el pasado
de entrenamiento porque ese pasado ya era conocido.

PPO usa solo train para actualizar pesos. La validación determina el checkpoint
con mayor PnL neto, con stake fijo y sin usar su reward de entrenamiento para
la selección. No se evalúa el test reservado ni se utiliza para elegir pesos.
Los resultados de validación son de selección, no un backtest final independiente.

Hay evaluación inicial, periódica y final tras la última actualización PPO.
Si ninguna política supera a la inicial, la inicial puede quedar como mejor;
si todas pierden, se guarda la de menor pérdida. No se obliga a operar ni se
promete rentabilidad. Un total-timesteps pequeño puede redondearse hacia arriba
para completar un rollout de n_steps; el total real se registra.

## Archivos generados en models/

| Archivo | Propósito |
| --- | --- |
| best_ppo_agent.zip | Pesos seleccionados por validación |
| final_ppo_agent.zip | Últimos pesos entrenados |
| normalizer.json | Media, desviación, orden de columnas y clipping de train |
| training_manifest.json | Configuración, hash del input, splits, versiones y total de pasos |
| best_validation.json | Resultado que seleccionó el mejor checkpoint |
| validation_history.json | Todas las evaluaciones del proceso |

Cada archivo se escribe atómicamente. La atomicidad es individual, no una
transacción sobre los seis archivos. Usar carpetas separadas con --output-dir
para experimentos simultáneos; no ejecutar dos entrenamientos escribiendo sobre
el mismo directorio. No hay reanudación automática de un entrenamiento previo.

Inferencia consistente:

```python
import json
import pandas as pd
from src.agent import load_agent
from src.environment import AccountConfig, BinaryOptionsEnv, prepare_market_data

manifest = json.load(open('models/training_manifest.json'))
normalizer = json.load(open('models/normalizer.json'))
# Debe incluir suficiente pasado anterior al instante que quieres observar.
raw = pd.read_parquet('data/eurusd_m15_features.parquet')
ready = prepare_market_data(raw, manifest['lookback'], manifest['indicator_period'])
env = BinaryOptionsEnv(ready, prepared=True, lookback=manifest['lookback'],
                       normalizer=normalizer, account=AccountConfig(**manifest['account']))
model = load_agent('models/best_ppo_agent.zip', env=env)
observation, info = env.reset()
action, _ = model.predict(observation, deterministic=True)
```

Ese ejemplo demuestra carga/reproducción histórica desde el primer segmento;
no es un emisor conectado a un broker ni usa automáticamente la última fila.

## Pruebas

```bash
python -m pytest -q tests/test_environment.py tests/test_train.py
python -m pytest -q
```

Las pruebas de entorno comprueban PUT/CALL, neutralidad, empate configurable,
beneficio/capital, expiración de una vela, ausencia de contratos simultáneos,
insolvencia, drawdown, huecos, límites, seeds y contrato Gymnasium/SB3.
También comparan observaciones con futuros distintos y prefijos de indicadores.
Las pruebas de entrenamiento verifican split, normalización train-only,
continuidad de saldo por segmentos y entrenamiento PPO real de 32 pasos sobre
datos sintéticos, guardando/cargando los pesos seleccionados. Esa prueba no
entrena una política de mercado ni acredita beneficio fuera de muestra.
