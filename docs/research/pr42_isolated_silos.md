# PR42 — silos M5+M15/GARCH independientes, PR41 congelada (solo H1 IS)

## Decisión implementada

**Cuatro instancias autónomas por semilla**: EURUSD, GBPUSD, AUDUSD, USDCAD.
Las semillas 42, 43, 44 son **tres escenarios alternativos de cuatro silos**,
no una licencia para operar simultáneamente tres estrategias sobre el mismo par.
La implementación no elige una semilla ex post. Los doce genomas se mantienen
congelados, cada uno con 22 genes y SHA256 extraídos de los artefactos **finales**
de PR41 (GitHub Actions run `42576852134`). Las 30.720 evaluaciones ya
concluyeron: este cambio efectúa **cero evaluaciones genéticas nuevas**.

No se utiliza `replay_portfolio` ni se añade un gestor de cartera
compartida. `scripts.run_pr42_isolated_silos` carga exactamente **un**
archivo de genoma, **un** universo de datos de la divisa, una ventana de
GARCH nativa, una configuración de cuenta instanciada localmente y un
cupo local de 2 contratos/h. No lee universos de otras divisas; GBP, AUD
y CAD no tienen ninguna dependencia del universo ni modelo EUR en
tiempo de replay. Solo se conservan las mismas condiciones del
contrato M5 del experimento original: cuenta USD simulada con 1000 USD,
stake 10 USD, payout 0,8, stop DD 20% y p85 causal.

El Gate de Convicción, sus 500 observaciones de warmup, los
resultados de liquidaciones previas, el límite horario y el saldo se
gestionan en **el proceso de ese par**. Ningún stop ni pérdida de
otro activo modifica sus admisiones. Cada par mantiene su
`input_universe_sha256` propio, su propio umbral de margen, bandas
GARCH y 22 pesos locales del campeón.

## Qué se ejecuta ahora y qué no

1. El workflow `.github/workflows/pr42-independent-silos.yml` obtiene
   del CI final PR41 el informe **del símbolo y semilla exactos**.
2. Descarga el release H1 sellado correspondiente **solo a ese símbolo**
   y verifica SHA256, fechas, contrato, orden temporal y GARCH ya nativa.
3. Sin mutar genes ni reoptimizar, reproduce únicamente las decisiones
   históricas de Q1/Q2 y comprueba PF Q1, PF Q2, frecuencia, DD, PnL
   y umbrales contra el campeón congelado. Una divergencia falla.
4. Exporta `silo.json` y `decisions.csv` dentro del espacio
   `reports/pr42-isolated-silos/{symbol}/seed-{seed}/`. Los balances
   de los replay de cada símbolo son propios.
5. El job final agrega **solo la frecuencia informativa** de las cuatro
   cuentas por semilla. Nunca suma equity para elegir un genoma ni crea
   un PF o DD artificial de cartera con cuenta compartida.

Este es un **despliegue de arquitectura de investigación histórica**, sin
conexión de broker, señales vigentes, ejecución en vivo, órdenes, ni
promoción de producción. Los artefactos congelados de Q1+Q2 ya han
sido usados para selección, por lo que reproducirlos no añade evidencia
de alfa fuera de muestra. Solo después de elegir un protocolo OOS
pre-registrado y autorizar expresamente la apertura de Q3 podrá
evaluarse su extrapolación.

## Cuarentena Q3

Período permitido: 2026-01-01 00:00Z a 2026-07-01 00:00Z, **exclusivo**.
Se rechaza toda serie con marca temporal a partir del 1 de julio,
universos distintos de los cuatro SHA256 certificados en PR41,
material extra M1 usado como señal o genomas modificados. La prueba
Q3 2026 julio–septiembre sigue blindada. Ninguna etapa del
workflow accede a sus archivos, usa el resultado para selección,
reentrena o abre el OOS.

## Interpretación operativa

Cada modelo PR41 ejecutó entre aproximadamente 0,016 y 0,086
contratos/h IS. Su suma descriptiva por semilla puede alcanzar
aproximadamente 0,17–0,20/h, pero **no es la misma cifra** que
el replay de cartera compartida PR41 porque aquel introducía
un límite global, colisiones y capital compartido. Para obtener
la cifra exacta de las cuentas separadas se requiere el reporte
de PR42, sin inventar métricas ni declarar éxito hasta que pase CI.

El objetivo original de 1 operación/h deja de ser una condición
para permitir el estudio de silos; NO se relajan los filtros
financieros, GARCH ni el p85. La selección de semilla por divisa
para uso futuro sigue sin definición y **no podrá fundamentarse
en el resultado de Q3**.

## Cómo repetir la verificación sin evolucionar

```bash
python -m pytest -q tests/test_pr42_isolated_silos.py
# Ejemplo GBPUSD seed 42; se necesitan los dos archivos H1 sellados:
python -m scripts.run_pr42_isolated_silos \
  --symbol GBPUSD --seed 42 \
  --report reports/pr42-frozen-champion/gbpusd-seed-42.json \
  --universe reports/pr42-symbol-data/universe.npz \
  --manifest reports/pr42-symbol-data/manifest.json
```

Para EURUSD, añadir `--golden` y `--golden-manifest` del
release canónico utilizado en PR41.

Los resultados son pruebas de **reproducción de IS y aislamiento
de software**, no un ensayo financiero independiente ni una
autorización para operar dinero real.
