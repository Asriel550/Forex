# Ingesta estática EUR/USD M15

El pipeline de entrenamiento lee el snapshot HistData ya incluido en el
repositorio: no llama a Dukascopy, Yahoo, OANDA ni HistData y no requiere cargas
manuales. Se adquirieron los años 2021–2025 y enero–octubre de 2026 directamente
del sitio oficial, con su formulario normal de descarga. El sitio publica M1;
se convirtieron a M15 UTC sin interpolación. Ver datasets/eurusd/manifest.json.

## Snapshot ya instalado

Hay seis ZIP anuales con 135,706 velas M15. config/closures.json documenta solo
los intervalos ausentes comprobados en estas fuentes. Algunos corresponden a
minutos incompletos o contradictorios y otros a ausencia total de cotizaciones.
No se han clasificado como festivos ni como cierres del mercado. Son exclusiones
del dataset y el entorno corta los episodios en los saltos temporales. El motor
GARCH usa retornos entre observaciones disponibles, como en el contrato original;
la ausencia de cotizaciones limita la representatividad de las métricas.

El CLI usa config/closures.json por defecto si existe. Desde Python, pasar
closures=read_closures("config/closures.json") explícitamente. Los huecos que no
figuran en esa lista siguen fallando.

## Reproducir la adquisición (opcional, fuera del pipeline)

```bash
python scripts/acquire_histdata.py --start-year 2021 --end-year 2025 --through-month 2026-10
python scripts/build_histdata_dataset.py
python -m src.data_loader --closures config/closures.json
```

El primer script descarga una vez los ZIP originales al directorio ignorado
data/histdata_raw; valida formato ZIP/CRC y registra SHA256. Usa el formulario
público normal de HistData, con sesión y referer, de forma secuencial. No sortea
CAPTCHAs ni bloqueos y falla si el proveedor no publica el archivo.
El segundo comprueba esos hashes, elimina duplicados exactos, excluye todos los
minutos contradictorios y conserva solo grupos de 15 minutos completos. Produce
los ZIP deterministas, el manifiesto y las exclusiones. Los originales permanecen
fuera de Git; los archivos compactos M15 necesarios para Actions quedan en Git.

Los scripts son para renovar el snapshot de forma controlada; no se ejecutan
en cron ni en cada entrenamiento y no implican disponibilidad futura del sitio.

## Añadir o sustituir archivos desde Android (opcional)

Estos pasos son solo para renovar o ampliar el snapshot; el histórico requerido
ya está incluido.

1. En GitHub web abrir datasets/eurusd y elegir Add file → Upload files. Subir
   CSV o ZIP; no es necesario descomprimirlos en Android. Si los controles no
   aparecen, activar Sitio para ordenadores en el navegador. Para la carga web,
   dividir archivos grandes en partes cronológicas menores de 25 MiB cada una.
2. Mantener archivos originales y su nombre cuando sean HistData M1. El cargador
   encuentra archivos también en subcarpetas y ordena todas las filas. Los
   duplicados idénticos entre partes se eliminan antes de sumar volumen.
3. CSV OHLCV M15 con timestamps UTC y sufijo Z no requiere variables. Si sus
   timestamps no tienen zona, en Settings → Secrets and variables → Actions →
   Variables crear FOREX_SOURCE_TIMEZONE=UTC. Para HistData original dejar EST.
4. Para M1 renombrado crear FOREX_INPUT_TIMEFRAME=M1. Para M15 usar M15 o auto.
   No combinar M1 y M15 en el mismo directorio de ingesta. Una zona común se
   aplica a todas las fechas sin offset; no mezclar zonas ambiguas.
5. Guardar los archivos en la rama por defecto. Ejecutar Actions → Forex M15
   research and Telegram → Run workflow → mode=train. Una carga en main también
   activa las pruebas e investigación según los triggers existentes.
6. Revisar Read static EURUSD CSV or ZIP files. Debe mostrar el número de
   velas guardadas y su rango UTC. Después siguen quant, PPO y holdout. Los
   secretos existentes de Telegram no se modifican.

## Formatos

CSV con cabecera, separado por coma, punto y coma o tabulador:

```text
timestamp,open,high,low,close,volume
2024-01-02T05:00:00Z,1.1000,1.1010,1.0990,1.1005,0
```

La fila anterior ilustra el formato; no es un dato de mercado verificado.
También se admiten seis campos sin cabecera, o siete con fecha y hora separadas.
El formato HistData Generic ASCII es YYYYMMDD HHMMSS;open;high;low;close;volume.
HistData lo publica en EST fijo UTC-5 sin DST, no en America/New_York. Su
formato nativo de velas Generic ASCII es M1. Nombres DAT_ASCII_EURUSD_M1_* y
DAT_MT_EURUSD_M1_* se reconocen automáticamente y se agregan a M15. Si un
archivo declarado M15 contiene M1 se rechaza; no se adivina la resolución.

Documentación del proveedor:
https://www.histdata.com/f-a-q/data-files-detailed-specification/
https://www.histdata.com/f-a-q/

Los timestamps son apertura de vela. En M1 se exigen los 15 minutos distintos
por grupo cerrado: open primero, high máximo, low mínimo, close último y volume
suma. Los grupos aún en formación se excluyen. Minutos ausentes no se rellenan;
si el proveedor no cotiza un minuto, debe corregirse/documentarse el dataset
antes de usarlo con esta política estricta. Los ticks Bid/Ask no son OHLCV.

## Contrato y validación

Salida: data/eurusd_m15.parquet, Zstandard, columnas exactamente timestamp,
open, high, low, close, volume. Timestamp timezone-aware UTC; números float64.
Se rechazan nulos/infinitos, OHLC incoherente, precios no positivos, volumen
negativo, timestamps fuera de rejilla y duplicados contradictorios.

El calendario esperado es domingo 17:00 a viernes 17:00 Nueva York, con DST.
Los huecos internos y límites explícitos start/end se validan contra él.
Festivos o pausas verificadas se pueden documentar en config/closures.json:

```json
[{"start":"2024-12-25T00:00:00Z","end":"2024-12-26T00:00:00Z","reason":"Ejemplo: sustituir por cierre comprobado del proveedor"}]
```

No declarar cierres ficticios. Las exclusiones incluidas identifican observaciones
ausentes/descartadas verificadas del feed, sin afirmar un cierre de mercado. La ingesta sin
start/end carga todo el rango cerrado disponible; no exige llegar a hoy.
Se conserva el último Parquet válido si la actualización falla. La caché
compara SHA256 del contenido y parámetros, y se publica con reemplazo atómico
bajo bloqueo. No vuelve a parsear fuentes idénticas; sí lee sus bytes para
verificar hashes. El límite predeterminado es 512 MiB de entrada total/CSV
descomprimidos y 2000 miembros por ZIP. No se extraen archivos al disco; se
rechazan rutas inseguras, symlinks, cifrado y ZIP con compresión excesiva.

## Uso Python/CLI

```bash
python -m src.data_loader --input-dir datasets/eurusd
python -m src.data_loader --inputs datasets/eurusd/part1.zip datasets/eurusd/part2.zip
python -m src.data_loader --inputs datasets/eurusd/renombrado.csv --input-timeframe M1
python -m src.data_loader --input-dir datasets/eurusd --timezone UTC --start 2020-01-01 --end 2025-01-01
```

```python
from src.data_loader import load_data
from src.quant_engine import enrich_features
from src.environment import prepare_market_data

from src.data_loader import read_closures
raw = load_data(input_dir="datasets/eurusd", closures=read_closures("config/closures.json"))
features = enrich_features(raw)
prepared = prepare_market_data(features)
```

start/end son UTC y end es exclusivo. Timestamps con offset se respetan;
--timezone solo interpreta fechas sin zona. --timestamp-format permite un
formato explícito y --epoch-unit permite s/ms/us/ns. También se admite Parquet
local sin características enriquecidas, con el mismo contrato OHLCV.

## Límite de las alertas

El cron no actualiza los archivos. Un dataset histórico permite investigar y
entrenar durante años sin API en vivo. Para emitir una alerta actual es necesario
aportar el último cierre y continuidad con el bundle. Archivos viejos se omiten
por frescura; un bundle de más de 14 días requiere nuevo entrenamiento. Se
conservan los hashes, la outbox y los reintentos exclusivos de Telegram.
