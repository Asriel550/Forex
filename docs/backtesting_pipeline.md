# Backtest reservado

Ejecutar `python -m src.backtest` después de `python -m src.train`.
El manifiesto v2 vincula pesos, normalizador y Parquet por SHA256, registra límites
y exige entrenamiento completado. Datos modificados o un manifiesto incompleto
se rechazan. No se acepta elegir fechas arbitrarias que incluyan train/validation.

La normalización permanece congelada. El backtest recibe solo el subconjunto test,
conserva el capital entre segmentos y no recapitaliza tras insolvencia. Si hay una
parada de riesgo, se mantiene el efectivo restante hasta el final del holdout.

Win rate incluye todos los contratos ejecutados, con empates en el denominador;
se ofrece otra tasa excluyendo reembolsos. Neutros no son contratos. PF es beneficio
bruto/pérdida bruta. Sin pérdidas, se devuelve null con estado infinite_no_losses.
Drawdown mide caída desde el pico, en USD y %. Sharpe y Sortino usan retornos de
equity por barra e incluyen inactividad; anualizan con la convención 252*96.
El PnL monetario no incluye la penalización artificial de reward por inactividad.

Se exportan JSON estricto, trades.csv, signals.csv, equity.csv y dos PNG. Las horas
son de cierre UTC (timestamp de apertura + quince minutos). La muestra de test no
debe reutilizarse para afinar hiperparámetros una vez conocidos sus resultados.
