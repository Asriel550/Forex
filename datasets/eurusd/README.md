# Histórico EUR/USD incluido: HistData → M15 UTC

**No es necesario subir archivos manualmente.** Se descargaron los ZIP oficiales
de HistData de 2021, 2022, 2023, 2024 y 2025, más los diez meses publicados de
2026 hasta octubre. El snapshot está incluido en seis ZIP anuales M15.

- 135,706 velas, del 2021-01-03 22:15 UTC al 2026-10-30 20:30 UTC.
- Fuente M1: EST fijo UTC-5; salida: apertura de vela UTC con sufijo Z.
- Solo grupos de 15 minutos únicos y coherentes; no hay precios inventados.
- Se eliminaron duplicados exactos y se excluyeron seis timestamps M1 con
  cotizaciones contradictorias. También se descartaron 6,234 grupos M15 incompletos.
- Los 10,853 slots M15 ausentes durante el calendario esperado se registran en
  [config/closures.json](../../config/closures.json), agrupados en 4,767 intervalos.
  Son exclusiones documentadas del feed, **no afirmaciones de mercado cerrado**.
- [manifest.json](manifest.json) contiene URLs originales, hashes de los ZIP M1,
  auditoría por fuente y hashes de ZIP/CSV finales. OHLC proviene de cotizaciones
  Bid del proveedor; el volumen se conserva tal como se publica (cero).

El workflow lee este snapshot con sus exclusiones. No descarga HistData ni
Dukascopy en cada entrenamiento. Los huecos nuevos o no documentados fallan;
el entorno no abre contratos que expiren cruzando un hueco.

Ver [proceso reproducible y uso](../../docs/static_ingestion.md).
El histórico permite investigación/entrenamiento, pero no es un feed de alertas
en tiempo real: el bot mantiene su control de frescura.
