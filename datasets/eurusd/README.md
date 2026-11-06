# Archivos reales de EUR/USD

Subir aquí CSV o ZIP que contengan CSV OHLCV de EUR/USD. No hay descarga desde
Dukascopy ni datos sintéticos de entrenamiento incluidos en el repositorio.

Desde Android, abrir esta carpeta en GitHub web → Add file → Upload files →
seleccionar los CSV/ZIP del móvil → Commit changes. Si no aparecen los controles,
activar “Sitio para ordenadores” en el navegador. No pegar ZIP en el editor.

Conservar el nombre HistData DAT_ASCII_EURUSD_M1_*.csv dentro del ZIP para
reconocer M1 automáticamente. Un CSV M15 con zona usa timestamp UTC con Z;
un timestamp HistData sin zona se interpreta EST fijo (UTC-5).

Consultar [formatos, variables y validaciones](../../docs/static_ingestion.md).
Los históricos antiguos permiten entrenar; no producen alertas vigentes sin
actualización con el último cierre M15.
