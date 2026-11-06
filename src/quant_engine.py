"""Features GARCH(1,1) + Monte Carlo GBM causales para EUR/USD M15.

Una fila t se conoce AL CIERRE de su vela; ejecutar a partir de la siguiente.
cond_volatility es sigma[t+1|t], en retorno decimal POR VELA, sin anualizar.
ARCH ajusta log-retornos en puntos básicos (x10,000) para estabilidad numérica.
Se reestima cada refit_every velas con una ventana de pasado; entre ajustes
se actualiza la recursión de varianza, nunca se ajusta toda la muestra futura.

GBM: S[k] = S[0] exp((mu - sigma²/2) k + sigma W[k]), dt=1 vela.
mu es drift aritmético POR VELA (cero por defecto), sigma queda congelada
32 pasos. Los percentiles son TERMINALES a 32 barras, no máximos/mínimos de
las rutas ni niveles garantizados de soporte/resistencia. Un fin de semana
hace que 32 barras de negociación no equivalgan a 8 horas de reloj.

Monte Carlo usa common random numbers: las mismas 10,000 rutas brownianas
independientes entre sí se reutilizan para cada origen temporal. Sus shocks
terminales ordenados permiten percentiles empíricos EXACTOS (interpolación
lineal) en O(1) por vela, sin generar un tensor N x 10,000 x 32. Se puede
materializar la matriz de precios completa con MonteCarloGBM.paths().
No sustituye Monte Carlo por percentiles analíticos de una lognormal.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import logging
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning

LOG = logging.getLogger(__name__)
SCALE = 10_000.0
FEATURES = ["cond_volatility", "dist_p95", "dist_p05"]
ROOT = Path(__file__).resolve().parents[1]


class QuantError(ValueError):
    """Entrada inválida o ajuste GARCH no fiable."""


@dataclass(frozen=True)
class QuantConfig:
    window: int = 2016
    min_observations: int = 512
    refit_every: int = 96
    n_paths: int = 10_000
    horizon: int = 32
    seed: int = 42
    drift_per_bar: float = 0.0
    max_iterations: int = 500

    def __post_init__(self):
        for name in ("window", "min_observations", "refit_every", "n_paths",
                     "horizon", "max_iterations", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise QuantError(f"{name} debe ser entero")
            if value < (0 if name == "seed" else 1):
                raise QuantError(f"{name} fuera de rango")
        if self.min_observations < 100 or self.window < self.min_observations:
            raise QuantError("Requiere window >= min_observations >= 100")
        if self.n_paths < 100 or not np.isfinite(self.drift_per_bar):
            raise QuantError("Requiere n_paths >= 100 y drift finito")


class MonteCarloGBM:
    """Banco determinista de innovaciones; no modifica el RNG global.

    Memoria por defecto: ~2.56 MB para 10,000 x 32 float64. La evaluación
    terminal vectorizada usa memoria O(N_velas), independiente de n_paths.
    """

    def __init__(self, n_paths=10_000, horizon=32, seed=42):
        if (not isinstance(n_paths, (int, np.integer)) or n_paths < 2
                or not isinstance(horizon, (int, np.integer)) or horizon < 1):
            raise QuantError("n_paths >= 2 y horizon >= 1 deben ser enteros")
        self.n_paths, self.horizon = int(n_paths), int(horizon)
        rng = np.random.Generator(np.random.PCG64(seed))
        self._brownian = rng.standard_normal((n_paths, horizon)).cumsum(axis=1)
        self._terminal = np.sort(self._brownian[:, -1])

    @staticmethod
    def _inputs(spot, sigma, drift):
        spot, sigma, drift = np.broadcast_arrays(
            np.asarray(spot, dtype=np.float64), np.asarray(sigma, dtype=np.float64),
            np.asarray(drift, dtype=np.float64))
        if (not np.isfinite(spot).all() or not np.isfinite(sigma).all()
                or not np.isfinite(drift).all() or (spot <= 0).any() or (sigma < 0).any()):
            raise QuantError("spot > 0, sigma >= 0 y drift deben ser finitos")
        return spot, sigma, drift

    def paths(self, spot: float, sigma: float, drift=0.0) -> np.ndarray:
        """Matriz (n_paths, horizon), pasos 1..H; no incluye el precio inicial."""
        spot, sigma, drift = self._inputs(spot, sigma, drift)
        if spot.ndim:
            raise QuantError("paths requiere escalares")
        k = np.arange(1, self.horizon + 1)
        with np.errstate(over="raise", invalid="raise"):
            return spot * np.exp((drift - 0.5 * sigma**2) * k + sigma * self._brownian)

    def terminal_quantiles(self, spot, sigma, drift=0.0):
        """p05, p95 empíricos vectorizados; iguales a np.quantile(paths[:,-1]).

        Interpola PRECIOS, no log-precios: exp(quantile(log S)) no es en
        general igual a quantile(S) con la interpolación lineal de NumPy.
        """
        spot, sigma, drift = self._inputs(spot, sigma, drift)
        offset = (drift - 0.5 * sigma**2) * self.horizon
        quantiles = []
        for q in (0.05, 0.95):
            rank = q * (self.n_paths - 1)
            lower, upper = int(np.floor(rank)), int(np.ceil(rank))
            weight = rank - lower
            with np.errstate(over="raise", invalid="raise"):
                left = spot * np.exp(offset + sigma * self._terminal[lower])
                right = spot * np.exp(offset + sigma * self._terminal[upper])
            quantiles.append(left + weight * (right - left))
        return tuple(quantiles)


def _fit_garch(sample: np.ndarray, max_iterations: int):
    """Máxima verosimilitud normal de ARCH; no usa inicialización aleatoria."""
    if np.var(sample) < 1e-12:
        raise QuantError("Retornos sin variación suficiente para ajustar GARCH")
    # Normaliza solo la ventana histórica de este ajuste. La verosimilitud
    # equivalente tiene mejor escala; después restaura omega/h a puntos básicos².
    rms = float(np.sqrt(np.mean(sample**2)))
    model = arch_model(sample / rms, mean="Zero", vol="GARCH", p=1, o=0, q=1,
                       dist="normal", rescale=False)
    result = None
    # Inicios deterministas alternativos para soluciones en la frontera alpha=0.
    starts = (None, np.array([0.01, 0.05, 0.94]), np.array([0.1, 0.1, 0.8]))
    for starting_values in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                candidate = model.fit(disp="off", update_freq=0, show_warning=False,
                                      starting_values=starting_values,
                                      options={"maxiter": max_iterations, "ftol": 1e-8})
            except ConvergenceWarning:
                continue
        if candidate.convergence_flag == 0:
            result = candidate
            break
    if result is None:
        raise QuantError("El ajuste GARCH no convergió tras 3 inicios deterministas")
    omega, alpha, beta = (float(result.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
    omega *= rms**2
    if (not np.isfinite([omega, alpha, beta]).all() or omega <= 0
            or min(alpha, beta) < 0 or alpha + beta > 1 + 1e-6):
        raise QuantError("Parámetros GARCH inválidos")
    h_t = float(np.asarray(result.conditional_volatility)[-1]) ** 2 * rms**2
    return omega, alpha, beta, h_t


def causal_garch_volatility(close, config: QuantConfig):
    """sigma[t+1|t], historial de ajustes. NaN explícito durante warmup.

    El primer ajuste utiliza min_observations retornos, por lo que necesita
    min_observations+1 cierres. Un fallo aborta: no se oculta con EWMA ni
    parámetros obtenidos usando la muestra futura.
    """
    prices = np.asarray(close, dtype=np.float64)
    if prices.ndim != 1 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise QuantError("close debe ser un vector de precios positivos finitos")
    if len(prices) <= config.min_observations:
        raise QuantError("Histórico insuficiente para el warmup GARCH")
    returns = np.diff(np.log(prices)) * SCALE
    volatility = np.full(len(prices), np.nan)
    fits = []
    next_h = None
    for t in range(config.min_observations, len(prices)):
        if (t - config.min_observations) % config.refit_every == 0:
            begin = max(0, t - config.window)
            # returns[t-1] es el retorno de la vela t: límite derecho exclusivo t.
            omega, alpha, beta, h_t = _fit_garch(returns[begin:t], config.max_iterations)
            fits.append(dict(row=t, first_return=begin, omega=omega,
                             alpha=alpha, beta=beta))
        else:
            h_t = next_h
        next_h = omega + alpha * returns[t - 1]**2 + beta * h_t
        if not np.isfinite(next_h) or next_h <= 0:
            raise QuantError(f"Varianza no positiva/finita en fila {t}")
        volatility[t] = np.sqrt(next_h) / SCALE
    return volatility, fits


def enrich_features(df: pd.DataFrame, config: QuantConfig | None = None) -> pd.DataFrame:
    """Copia enriquecida: conserva orden, índice, OHLCV y warmup sin rellenar.

    dist_p95=(p95-close)/close; dist_p05=(p05-close)/close. Son fracciones,
    por ejemplo 0.01 significa 1%; dist_p05 es normalmente negativa.
    Se incluyen mc_p05 y mc_p95 en unidades de precio para auditoría.
    El warmup debe excluirse antes de alimentar PPO, nunca rellenarse hacia
    atrás con información futura.
    """
    config = config or QuantConfig()
    if not df.columns.is_unique or "timestamp" not in df or "close" not in df:
        raise QuantError("Se requieren columnas únicas timestamp y close")
    timestamps = pd.to_datetime(df.timestamp, errors="raise", utc=True)
    if (timestamps.isna().any() or timestamps.duplicated().any()
            or not timestamps.is_monotonic_increasing
            or (timestamps != timestamps.dt.floor("15min")).any()):
        raise QuantError("timestamp debe estar ordenado, ser único y estar alineado a M15")
    # Los huecos de mercado se verifican en data_loader; aquí las barras son tiempo de trading.
    out = df.copy()
    out["timestamp"] = timestamps
    volatility, fits = causal_garch_volatility(out.close, config)
    out["cond_volatility"] = volatility
    ready = np.isfinite(volatility)
    mc = MonteCarloGBM(config.n_paths, config.horizon, config.seed)
    p05, p95 = mc.terminal_quantiles(out.close.to_numpy()[ready], volatility[ready],
                                   config.drift_per_bar)
    out["mc_p05"], out["mc_p95"] = np.nan, np.nan
    out.loc[ready, "mc_p05"] = p05
    out.loc[ready, "mc_p95"] = p95
    out["dist_p95"] = (out.mc_p95 - out.close) / out.close
    out["dist_p05"] = (out.mc_p05 - out.close) / out.close
    out.attrs = {"quant_config": asdict(config), "garch_fits": fits,
                 "source_rows": len(df), "origin_timestamp": timestamps.iloc[0].isoformat(),
                 "volatility_units": "decimal per M15 bar, forecast sigma[t+1|t]",
                 "available_at": "candle close; execute no earlier than next bar",
                 "monte_carlo": "terminal empirical quantiles; common random numbers"}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/eurusd_m15.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/eurusd_m15_features.parquet")
    for name in ("window", "min_observations", "refit_every", "n_paths", "horizon", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            default=getattr(QuantConfig(), name))
    parser.add_argument("--drift-per-bar", type=float, default=0.0)
    parser.add_argument("--drop-warmup", action="store_true")
    args = vars(parser.parse_args())
    source, output, drop = args.pop("input"), args.pop("output"), args.pop("drop_warmup")
    if source.resolve() == output.resolve():
        parser.error("input y output deben ser diferentes")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = enrich_features(pd.read_parquet(source), QuantConfig(**args))
    if drop:
        result = result.dropna(subset=FEATURES).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        result.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    LOG.info("Guardado %s: %d filas, %d ajustes GARCH", output, len(result),
             len(result.attrs["garch_fits"]))


if __name__ == "__main__":
    main()
