# Sensibilidad de investigación (baseline MC, no PPO)

Payout 0.80; acierto de equilibrio sin reembolsos 55.56%.
Solo entrenamiento/validación. No se selecciona un ganador automáticamente.

| Fase | Escala | Umbral | Contratos | PnL USD | PF | Acierto | DD | Stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 1.00 | 58% | 45 | -36.00 | 0.836 | 51.11% | 10.40% | False |
| train | 1.00 | 60% | 13 | 14.00 | 1.280 | 61.54% | 2.20% | False |
| train | 1.00 | 62% | 5 | 22.00 | 3.200 | 80.00% | 0.97% | False |
| train | 1.00 | 65% | 1 | 8.00 | infinite_no_losses | 100.00% | 0.00% | False |
| train | 1.50 | 58% | 92 | -180.00 | 0.640 | 44.44% | 20.39% | True |
| train | 1.50 | 60% | 82 | -64.00 | 0.840 | 51.22% | 10.67% | False |
| train | 1.50 | 62% | 43 | -34.00 | 0.838 | 51.16% | 9.40% | False |
| train | 1.50 | 65% | 12 | 24.00 | 1.600 | 66.67% | 1.20% | False |
| train | 2.00 | 58% | 129 | -208.00 | 0.694 | 46.46% | 20.80% | True |
| train | 2.00 | 60% | 99 | -196.00 | 0.637 | 44.33% | 20.87% | True |
| train | 2.00 | 62% | 116 | -170.00 | 0.721 | 47.41% | 19.57% | False |
| train | 2.00 | 65% | 55 | -28.00 | 0.892 | 52.73% | 8.40% | False |
| validation | 1.00 | 58% | 34 | 56.00 | 1.467 | 64.71% | 2.23% | False |
| validation | 1.00 | 60% | 15 | 30.00 | 1.600 | 66.67% | 2.13% | False |
| validation | 1.00 | 62% | 5 | 4.00 | 1.200 | 60.00% | 1.97% | False |
| validation | 1.00 | 65% | 2 | -20.00 | 0.000 | 0.00% | 2.00% | False |
| validation | 1.50 | 58% | 72 | 90.00 | 1.333 | 62.50% | 3.94% | False |
| validation | 1.50 | 60% | 48 | 96.00 | 1.600 | 66.67% | 3.00% | False |
| validation | 1.50 | 62% | 31 | 50.00 | 1.455 | 64.52% | 2.25% | False |
| validation | 1.50 | 65% | 15 | 30.00 | 1.600 | 66.67% | 2.13% | False |
| validation | 2.00 | 58% | 107 | 28.00 | 1.061 | 57.01% | 7.82% | False |
| validation | 2.00 | 60% | 79 | 74.00 | 1.239 | 60.76% | 5.19% | False |
| validation | 2.00 | 62% | 59 | 76.00 | 1.345 | 62.71% | 3.98% | False |
| validation | 2.00 | 65% | 36 | 72.00 | 1.600 | 66.67% | 2.20% | False |

No son resultados PPO ni una calibración de probabilidad. No garantizan rendimiento futuro. El stop se aplica al liquidar y puede exceder 20% con el último contrato.
