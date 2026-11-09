# PR38 — Investigación de alfa M5+M15: gate, targets y estructura

## Invariantes obligatorios

- Universo de señales: **solo M5 y M15 cerradas**. El sello utilizable es
  `data-m5-m15-is-2026h1-full-universe-v2` (`05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9`).
  La Release v1 contiene antiguas señales M1 y NO es admisible para la entrada
  de esta arquitectura. M1 solo se usa como formato histórico de agregación OHLCV.
- GARCH canónico: `7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504`.
- IS exclusivamente **enero–junio de 2026** en Q1 y Q2; Q3 julio–septiembre
  fuera de todos los scripts, features, etiquetas y archivos.
- Dirección congelada Gen375/43 como fundador en Fases 1–2; se permite un
  residuo entrenable solo al iniciar la Fase 2.
- Contrato operativo sin cambios: vencimiento M5, +80%/−100%, cupo y stop
  originales. No se usa el payoff hipotético de un target R:R como si fuera
  profit factor de contratos realmente ejecutados.
- Neuroevolución de población **128**. El motor no realiza backprop; la loss
  entra como penalización del fitness genético.
- **Criterio conjunto por variante**: PF>1.20 en Q1 y Q2, frecuencia de
  contratos ejecutados del semestre ≥1.0/h, DD semestral <15%, ninguna parada
  de cuenta y resultado reproducido en seeds 42,43,44. CI exitoso significa
  infraestructura correcta, **no** éxito financiero.

## Fase 1: percentiles causales

Se probaron p85 (control), p80, p75 y un percentil condicional M5:
p75 con volatilidad actual inferior a 0.80 de la mediana de 48 muestras
previas, p85 superior a 1.20 y p80 en el intervalo. No es un filtro de
nuevas señales sino modulación del umbral convicción con la misma ventana
histórica de 500, warmup y liquidaciones maduras.

[Ejecutado satisfactoriamente en Actions](https://github.com/Asriel550/Forex/actions/runs/42549591449)
([resumen JSON](https://github.com/Asriel550/Forex/actions/runs/42549591449/artifacts/13644712136)).

Smoke p75 de 2 generaciones ×128 candidatos ×3 semillas:

- Seed42/G2: PF Q1 **0.760**, Q2 **0.939**, frecuencia operativa **0.0534/h**,
  DD semestral **20.00%**.
- Seed43/G2: PF Q1 **0.780**, Q2 **0.927**, frecuencia operativa **0.0534/h**,
  DD **20.42%**.
- Seed44/G2: PF Q1 **0.789**, Q2 **0.932**, frecuencia operativa **0.0483/h**,
  DD **20.56%**.

El p75 aumentó las propuestas admitidas sin recuperar el resultado tras
stop. En Q1 de las tres semillas el PF sigue por debajo de 0.85, en Q2
por debajo de 1.0. El gate dinámico también fracasa bajo los umbrales.
**Decisión: Fase 1 FAIL → Fase 2**.

## Fase 2: etiquetas de entrenamiento diferentes

Para separar **la pérdida de entrenamiento** del contrato M5 real, se
construyeron targets a partir de OHLC M5 y **ATR14 de M15 ya cerrada**.
Solamente se calcula una etiqueta si los 30 minutos de futuro están
íntegros antes de 2026-07-01. Se deshabilitan etiquetas si faltan velas
contiguas o ATR conocida. Ningún futuro entra en los inputs de inferencia.

[Release privada con etiquetas IS selladas](https://github.com/Asriel550/Forex/releases/tag/data-m5-m15-is-2026h1-alpha-labels-v1):
SHA256 `5163af1212217e4a05ad2cdd35928720eafb6031c8fba79c12b934975a2a469d`.

De 35.261 propuestas elegibles, **33.683** tienen seis M5 consecutivas y
ATR M15 válido. Las clases de entrenamiento son:

- `atr_expansion`: dirección cuyo máximo favorable de 30 minutos supere
  ATR14 M15, sea dominante frente a la excursión opuesta (1.2×), o neutra
  si no existe ventaja. 7.961 arriba, 8.248 abajo, 19.052 neutras.
- `rr15`: tomar 0.75×ATR antes de stop 0.50×ATR (R:R=1.5)
  durante seis velas M5. El toque simultáneo en una vela se resuelve
  conservadoramente como **stop antes del take-profit**, porque no hay
  información tick-by-tick. 10.117 arriba, 10.448 abajo, 14.696 neutras.
  **Es una etiqueta de entrenamiento, NO un nuevo contrato liquidado R:R.**
- `asymmetric_fp`: conserva el signo de la vela de vencimiento M5, pero
  la loss penaliza errores direccionales de alta convicción y operaciones
  de bajo momentum contemporáneo M5/M15 con multiplicador adicional 2.5.

Cada variante entrena una **cabeza direccional residual** de 13 genes
sobre el score Gen375, junto con 11 genes upstream, mediante 16
generaciones de 128 por cada semilla. Se mide PF real, DD, PnL y frecuencia
después del mismo gate p85 y stop de cuenta; no se da por conseguido un
objetivo porque una etiqueta R:R tenga más aciertos.

[CI fase 2](https://github.com/Asriel550/Forex/actions/runs/42551086458).

## Fase 3 condicional

Si Fase 2 no obtiene PF>1.10 en ambos trimestres para las tres semillas
de una misma variante y ningún modelo satisface el criterio final,
se ejecutan 15 generaciones de 128 × tres semillas para:

- `m15_attention`: atención causal sencilla sobre el cuerpo de la última
  M15 cerrada y de la M15 anterior, sin datos M1 directos.
- `garch_regime`: limita propuestas a bandas de volatilidad GARCH M5
  relativas a medianas de muestras anteriores; el gate p85 sigue intacto.

El informe final incluye por hipótesis, generación y seed, PF Q1/Q2,
frecuencia operativa semestral, DD semestral y evidencia de que el modelo
no fue promovido. Si los requisitos no se alcanzan en las variantes
acotadas, se declarará fracaso de este barrido IS, no un éxito inventado.
