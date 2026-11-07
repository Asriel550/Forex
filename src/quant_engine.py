"""Features GARCH(1,1) + Monte Carlo GBM causales para EUR/USD M15.

Una fila t se conoce AL CIERRE de su vela; ejecutar a partir de la siguiente.
cond_volatility es sigma[t+1|t], en retorno decimal POR VELA, sin anualizar.
ARCH ajusta log-retornos en puntos básicos (x10,000) para estabilidad numérica.
Se reestima cada refit_every velas con una ventana de pasado; entre ajustes
se actualiza la recursión de varianza, nunca se ajusta toda la muestra futura.

GBM: S[k] = S[0] exp((mu - sigma²/2) k + sigma W[k]), dt=1 vela.
mu es drift aritmético POR VELA, derivado de momentum causal por defecto; sigma queda congelada
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
from .market import MarketConfig
from dataclasses import asdict, dataclass, replace
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
PROBABILITY_FEATURES = ['mc_prob_up', 'mc_prob_down', 'mc_prob_tie']
DRIFT_AUDIT_FEATURES = ['mc_drift','mc_log_drift','momentum_mean','momentum_se','momentum_z']
ROOT = Path(__file__).resolve().parents[1]


class QuantError(ValueError):
    """Entrada inválida o ajuste GARCH no fiable."""


class GarchConvergenceError(QuantError):
    """El optimizador no encontró un ajuste convergente."""


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
    drift_mode: str = 'momentum'
    drift_window: int = 8
    drift_shrinkage: float = .75
    drift_scale: float = 1.5
    drift_noise_z: float = 1.96
    drift_cap_sigma: float = .75
    drift_clip_sigma: float = 3.

    @classmethod
    def from_metadata(cls, metadata):
        # Un bundle anterior no declara modo: representa drift CONSTANTE.
        # Nunca reinterpretar pesos viejos con los nuevos defaults de momentum.
        values = dict(metadata)
        values.setdefault('drift_mode','constant')
        values.setdefault('drift_scale',1.)
        return cls(**values)

    def __post_init__(self):
        for name in ("window", "min_observations", "refit_every", "n_paths",
                     "horizon", "max_iterations", "seed", "drift_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise QuantError(f"{name} debe ser entero")
            if value < (0 if name == "seed" else 1):
                raise QuantError(f"{name} fuera de rango")
        if self.min_observations < 100 or self.window < self.min_observations:
            raise QuantError("Requiere window >= min_observations >= 100")
        if self.n_paths < 100 or not np.isfinite(self.drift_per_bar):
            raise QuantError("Requiere n_paths >= 100 y drift finito")
        if self.drift_mode not in ('constant','momentum') or self.drift_window<4:
            raise QuantError('drift_mode constant/momentum; drift_window >= 4')
        params=[self.drift_shrinkage,self.drift_noise_z,self.drift_cap_sigma,self.drift_clip_sigma,self.drift_scale]
        if (not np.isfinite(params).all() or not 0<=self.drift_shrinkage<=1
                or self.drift_noise_z<0 or self.drift_cap_sigma<=0 or self.drift_clip_sigma<=0
                or not 0<self.drift_scale<=3):
            raise QuantError('Parámetros de momentum inválidos')
        if self.drift_mode=='momentum' and self.drift_per_bar!=0:
            raise QuantError('drift_per_bar no nulo requiere drift_mode=constant')


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
        self._one_step = np.sort(self._brownian[:, 0])

    def direction_probabilities(self, sigma, drift=0.0, expiry_candles=1):
        """Fracciones empíricas de las mismas rutas en expiry_candles (default 1).

        P05/P95 siguen siendo a 32 pasos. Estas probabilidades no son una
        confianza PPO ni probabilidades calibradas contra resultados reales.
        searchsorted evita una matriz n_velas x 10,000; no usa cierres futuros.
        """
        if isinstance(expiry_candles,bool) or not isinstance(expiry_candles,int) or not 1<=expiry_candles<=self.horizon:
            raise QuantError('Expiración fuera de las rutas Monte Carlo')
        samples = self._one_step if expiry_candles==1 else np.sort(self._brownian[:,expiry_candles-1])
        _, sigma, drift = self._inputs(1., sigma, drift)
        threshold = np.divide(.5*sigma**2-drift, sigma,
                              out=np.zeros_like(sigma), where=sigma>0)*expiry_candles
        down = np.searchsorted(samples, threshold, side='left')/self.n_paths
        up = 1.-np.searchsorted(samples, threshold, side='right')/self.n_paths
        up = np.where(sigma>0, up, (drift>0).astype(float))
        down = np.where(sigma>0, down, (drift<0).astype(float))
        return up, down, np.maximum(0., 1.-up-down)

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



def project_monte_carlo(mc, spot, sigma, drift, expiry_candles=1):
    """Una misma mu alimenta bandas H y probabilidad a expiry_candles.

    Las distancias conservan el signo: ambas bandas pueden quedar arriba o
    abajo del spot. Posición sin clip=(spot-p05)/(p95-p05); bias=+1 si spot
    está debajo de p05, -1 si supera p95. Son geometría a H, no probabilidades
    binarias: el filtro de 70% sigue usando mc_prob_up/down del horizonte de expiración.
    """
    spot=np.asarray(spot,dtype=float)
    p05,p95=mc.terminal_quantiles(spot,sigma,drift)
    up,down,tie=mc.direction_probabilities(sigma,drift,expiry_candles)
    width=p95-p05
    fallback=np.where(spot<p05,0.,np.where(spot>p95,1.,.5))
    position=np.divide(spot-p05,width,out=np.broadcast_to(fallback,np.shape(width)).copy(),where=width>0)
    return dict(mc_p05=p05,mc_p95=p95,dist_p05=(p05-spot)/spot,dist_p95=(p95-spot)/spot,
                mc_prob_up=up,mc_prob_down=down,mc_prob_tie=tie,
                mc_band_position=position,mc_band_bias=np.where(spot<p05,1.,np.where(spot>p95,-1.,0.)))


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
        raise GarchConvergenceError("El ajuste GARCH no convergió tras 3 inicios deterministas")
    omega, alpha, beta = (float(result.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
    omega *= rms**2
    if (not np.isfinite([omega, alpha, beta]).all() or omega <= 0
            or min(alpha, beta) < 0 or alpha + beta > 1 + 1e-6):
        raise QuantError("Parámetros GARCH inválidos")
    h_t = float(np.asarray(result.conditional_volatility)[-1]) ** 2 * rms**2
    return omega, alpha, beta, h_t


def causal_garch_volatility(close, config: QuantConfig, *, retain_on_nonconvergence=False):
    """sigma[t+1|t], historial de ajustes. NaN explícito durante warmup.

    El primer ajuste utiliza min_observations retornos, por lo que necesita
    min_observations+1 cierres. Solo con retención explícita se permite un
    refit fallido aislado usando los últimos parámetros causales y su estado.
    El primer fallo sin ajuste previo o dos consecutivos abortan.
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
    retained_last_refit = False
    for t in range(config.min_observations, len(prices)):
        if (t - config.min_observations) % config.refit_every == 0:
            begin = max(0, t - config.window)
            # returns[t-1] es el retorno de la vela t: límite derecho exclusivo t.
            try:
                omega, alpha, beta, h_t = _fit_garch(returns[begin:t], config.max_iterations)
            except GarchConvergenceError:
                if not retain_on_nonconvergence or next_h is None or retained_last_refit:
                    raise
                retained_last_refit = True
                h_t = next_h
                fits.append(dict(row=t, retained_from_row=fits[-1]['row']))
                LOG.warning('GARCH sin convergencia en fila %d; se conserva el último ajuste por un intervalo', t)
            else:
                retained_last_refit = False
                fits.append(dict(row=t, first_return=begin, omega=omega, alpha=alpha, beta=beta))
        else:
            h_t = next_h
        next_h = omega + alpha * returns[t - 1]**2 + beta * h_t
        if not np.isfinite(next_h) or next_h <= 0:
            raise QuantError(f"Varianza no positiva/finita en fila {t}")
        volatility[t] = np.sqrt(next_h) / SCALE
    return volatility, fits



def momentum_log_drift(mean,se,sigma,config):
    excess=np.sign(mean)*np.maximum(np.abs(mean)-config.drift_noise_z*se,0.)
    return np.clip(config.drift_scale*config.drift_shrinkage*excess,
                   -config.drift_cap_sigma*sigma,config.drift_cap_sigma*sigma)


def reproject_drift_scale(frame,scale):
    """Reproyecta mu/MC con estadísticas causales archivadas; no reajusta GARCH.

    Es fila a fila: no lee retornos futuros ni cambia el origen de ventanas.
    Permite evaluar subconjuntos train/validation sin tocar el holdout.
    """
    cfg=replace(QuantConfig.from_metadata(frame.attrs.get('quant_config',{})),drift_scale=scale)
    if cfg.drift_mode!='momentum' or not {'momentum_mean','momentum_se','cond_volatility','close'}.issubset(frame):
        raise QuantError('La sensibilidad requiere features de momentum archivadas')
    out=frame.copy();sigma=out.cond_volatility.to_numpy(dtype=float);ready=np.isfinite(sigma)
    mean=out.momentum_mean.to_numpy(dtype=float)[ready];se=out.momentum_se.to_numpy(dtype=float)[ready]
    if not np.isfinite(mean).all() or not np.isfinite(se).all() or (se<0).any():
        raise QuantError('Estadísticas de momentum inválidas')
    log_mu=momentum_log_drift(mean,se,sigma[ready],cfg);mu=log_mu+.5*sigma[ready]**2
    out['mc_log_drift']=np.nan;out.loc[ready,'mc_log_drift']=log_mu
    out['mc_drift']=np.nan;out.loc[ready,'mc_drift']=mu
    mc=MonteCarloGBM(cfg.n_paths,cfg.horizon,cfg.seed)
    for name,values in project_monte_carlo(mc,out.close.to_numpy()[ready],sigma[ready],mu,int(frame.attrs.get("direction_probability_horizon",1))).items():
        out[name]=np.nan;out.loc[ready,name]=values
    out.attrs=dict(frame.attrs,quant_config=asdict(cfg))
    return out


def causal_drift(close, volatility, config: QuantConfig, timestamps=None, bar_minutes=15):
    """mu[t] GBM a partir de L retornos conocidos al CIERRE t.

    En momentum se estima la MEDIA LOGARÍTMICA (no el drift aritmético GBM):
      r = clip(log(P[j]/P[j-1]), +/- clip_sigma*sigma[t])
      se = max(std(r,ddof=1),sigma[t])/sqrt(L)
      log_mu = clip(scale*shrink*sign(mean)*max(abs(mean)-z*se,0), +/- cap*sigma[t])
      mu = log_mu + sigma[t]**2/2
    Así el exponente GBM es log_mu, sin confundir retornos log y aritméticos.
    La banda de ruido es una heurística de contracción, NO calibración predictiva.
    Sin L retornos contiguos del timeframe, log_mu=0; sigma NaN conserva warmup NaN.
    Si timestamps se omite, se asume que cada fila representa una vela contigua.
    """
    prices=np.asarray(close,dtype=float)
    sigma=np.asarray(volatility,dtype=float)
    if (prices.ndim!=1 or sigma.shape!=prices.shape or not np.isfinite(prices).all()
            or (prices<=0).any() or np.isinf(sigma).any() or (sigma[np.isfinite(sigma)]<0).any()):
        raise QuantError('Precios/volatilidad inválidos para drift causal')
    n=len(prices);ready=np.isfinite(sigma)
    ts=None
    if timestamps is not None:
        ts=pd.DatetimeIndex(pd.to_datetime(timestamps,utc=True,errors='raise')).as_unit('ns')
        if len(ts)!=n or ts.hasnans or not ts.is_unique or not ts.is_monotonic_increasing:
            raise QuantError('Timestamps inválidos para momentum')
    out={name:np.full(n,np.nan) for name in DRIFT_AUDIT_FEATURES}
    for name in DRIFT_AUDIT_FEATURES:out[name][ready]=0.
    out['mc_drift'][ready]=config.drift_per_bar
    out['mc_log_drift'][ready]=config.drift_per_bar-.5*sigma[ready]**2
    if config.drift_mode=='constant':return out
    # Sin señal se proyecta mediana plana, no una tendencia artificial.
    out['mc_drift'][ready]=.5*sigma[ready]**2
    out['mc_log_drift'][ready]=0.
    length=config.drift_window
    if n<=length:return out
    candidates=np.arange(length,n)
    valid=ready[candidates]
    if ts is not None:
        bad=np.r_[0,np.diff(ts.asi8)!=pd.Timedelta(minutes=bar_minutes).value]
        cumulative=np.r_[0,np.cumsum(bad)]
        valid &= (cumulative[candidates+1]-cumulative[candidates-length+1])==0
    candidates=candidates[valid]
    if not len(candidates):return out
    returns=np.diff(np.log(prices))
    windows=np.lib.stride_tricks.sliding_window_view(returns,length)[candidates-length]
    scales=sigma[candidates]
    clipped=np.clip(windows,-config.drift_clip_sigma*scales[:,None],config.drift_clip_sigma*scales[:,None])
    mean=clipped.mean(axis=1)
    se=np.maximum(clipped.std(axis=1,ddof=1),scales)/np.sqrt(length)
    z=np.divide(mean,se,out=np.zeros_like(mean),where=se>0)
    excess=np.sign(mean)*np.maximum(np.abs(mean)-config.drift_noise_z*se,0.)
    log_mu=momentum_log_drift(mean,se,scales,config)
    out['momentum_mean'][candidates]=mean;out['momentum_se'][candidates]=se
    out['momentum_z'][candidates]=z;out['mc_log_drift'][candidates]=log_mu
    out['mc_drift'][candidates]=log_mu+.5*scales**2
    return out


def enrich_features(df: pd.DataFrame, config: QuantConfig | None = None, *, timeframe=None, expiry_candles=1) -> pd.DataFrame:
    """Copia enriquecida: conserva orden, índice, OHLCV y warmup sin rellenar.

    dist_p95=(p95-close)/close; dist_p05=(p05-close)/close. Son fracciones,
    por ejemplo 0.01 significa 1%; ambos signos se conservan con drift.
    Se incluyen mc_p05 y mc_p95 en unidades de precio para auditoría.
    El warmup debe excluirse antes de alimentar PPO, nunca rellenarse hacia
    atrás con información futura.
    """
    config = config or QuantConfig()
    market = MarketConfig(timeframe or df.attrs.get("timeframe","M15"),expiry_candles=expiry_candles)
    if df.attrs.get("timeframe",market.timeframe)!=market.timeframe:
        raise QuantError("Temporalidad solicitada distinta a la archivada en los datos")
    if not df.columns.is_unique or "timestamp" not in df or "close" not in df:
        raise QuantError("Se requieren columnas únicas timestamp y close")
    timestamps = pd.to_datetime(df.timestamp, errors="raise", utc=True)
    if (timestamps.isna().any() or timestamps.duplicated().any()
            or not timestamps.is_monotonic_increasing
            or (timestamps != timestamps.dt.floor(market.frequency)).any()):
        raise QuantError("timestamp debe estar ordenado, ser único y estar alineado a M15")
    # Los huecos de mercado se verifican en data_loader; aquí las barras son tiempo de trading.
    out = df.copy()
    out["timestamp"] = timestamps
    volatility, fits = causal_garch_volatility(
        out.close, config, retain_on_nonconvergence=market.timeframe in ('M1', 'M5'),
    )
    out["cond_volatility"] = volatility
    drift = causal_drift(out.close,volatility,config,timestamps,int(market.step.total_seconds()/60))
    for name,values in drift.items():out[name]=values
    ready = np.isfinite(volatility)
    mc = MonteCarloGBM(config.n_paths, config.horizon, config.seed)
    projected = project_monte_carlo(mc,out.close.to_numpy()[ready],volatility[ready],out.mc_drift.to_numpy()[ready],expiry_candles)
    for name,values in projected.items():
        out[name] = np.nan
        out.loc[ready,name] = values
    out.attrs = {"quant_config": asdict(config), "garch_fits": [fit for fit in fits if "omega" in fit],
                 "source_rows": len(df), "origin_timestamp": timestamps.iloc[0].isoformat(),
                 "volatility_units": f"decimal per {market.timeframe} bar, forecast sigma[t+1|t]",
                 "available_at": "candle close; execute no earlier than next bar",
                 "monte_carlo": "terminal empirical quantiles; common random numbers"}
    out.attrs['drift_contract'] = 'causal_momentum_v1' if config.drift_mode=='momentum' else 'constant_v1'
    out.attrs['direction_probability_horizon'] = expiry_candles
    out.attrs['timeframe'] = market.timeframe
    if market.timeframe in ('M1', 'M5'):
        out.attrs['garch_retained_refits'] = [fit for fit in fits if 'retained_from_row' in fit]
    if 'source_provenance' in df.attrs:
        out.attrs['source_provenance'] = df.attrs['source_provenance']
    out.attrs['probability_interpretation'] = 'GBM empirical model probability, not calibrated certainty'
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/eurusd_m15.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/eurusd_m15_features.parquet")
    for name in ("window", "min_observations", "refit_every", "n_paths", "horizon", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            default=getattr(QuantConfig(), name))
    parser.add_argument('--drift-mode',choices=['constant','momentum'],default=QuantConfig().drift_mode)
    parser.add_argument('--drift-window',type=int,default=QuantConfig().drift_window)
    for name in ('drift_scale','drift_shrinkage','drift_noise_z','drift_cap_sigma','drift_clip_sigma'):
        parser.add_argument('--'+name.replace('_','-'),type=float,default=getattr(QuantConfig(),name))
    parser.add_argument("--drift-per-bar", type=float, default=0.0)
    parser.add_argument("--drop-warmup", action="store_true")
    parser.add_argument("--timeframe",choices=["M1","M5","M15","H1","H4"],default="M15")
    parser.add_argument("--expiry-candles",type=int,default=1)
    args = vars(parser.parse_args())
    source, output, drop = args.pop("input"), args.pop("output"), args.pop("drop_warmup")
    if source.resolve() == output.resolve():
        parser.error("input y output deben ser diferentes")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    timeframe,expiry=args.pop("timeframe"),args.pop("expiry_candles")
    result = enrich_features(pd.read_parquet(source), QuantConfig(**args),timeframe=timeframe,expiry_candles=expiry)
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

