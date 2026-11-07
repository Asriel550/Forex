"""Opciones binarias M15, expiración de una vela y liquidación sin solapamiento.

La observación t utiliza exclusivamente velas hasta t, incluidas sus OHLC.
Entrada hipotética al close[t], liquidación al close[t+1], 15 minutos después.
El timestamp de la fuente es APERTURA: la decisión ocurre timestamp[t]+15min.
No modela latencia, cotización contractual del broker ni ejecución real.
Payout es beneficio neto sobre stake: acertar devuelve stake + payout*stake.
No se añaden spread/comisiones Forex a un contrato cuyo payout ya está fijado.
Los empates devuelven stake por defecto; se puede configurar empate=derrota.

Los huecos separan episodios: jamás se abre un contrato de 15 minutos que
expire el lunes a partir del viernes. El último close de cada segmento solo
sirve para liquidar. Un evaluador puede transportar saldo/pico entre segmentos.
Las features cuantitativas de entrada deben haberse calculado causalmente:
ningún entorno puede detectar por sí solo leakage oculto dentro de un Parquet.
"""
from __future__ import annotations

from dataclasses import dataclass,replace

import gymnasium as gym
from gymnasium import spaces
from .market import MarketConfig
from .decision import RiskConfig, ensure_probabilities, resolve_action, mc_probability_lower
from .quant_engine import PROBABILITY_FEATURES
import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)
QUANT_FEATURES = ("cond_volatility", "dist_p95", "dist_p05")
ACCOUNT_FEATURES = ("balance_return", "drawdown", "previous_direction",
                    "previous_reward", "stake_to_balance", "payout", "trade_activity")


@dataclass(frozen=True)
class AccountConfig:
    initial_balance: float = 1_000.0
    stake: float = 10.0
    payout: float = 0.80
    tie_policy: str = "refund"
    inactivity_penalty: float = 0.0
    max_drawdown: float = 1.0

    def __post_init__(self):
        values = (self.initial_balance, self.stake, self.payout,
                  self.inactivity_penalty, self.max_drawdown)
        if not np.isfinite(values).all():
            raise ValueError("Parámetros de cuenta deben ser finitos")
        if self.stake <= 0 or self.initial_balance < self.stake:
            raise ValueError("Requiere initial_balance >= stake > 0")
        if not 0 < self.payout <= 1 or not 0 < self.max_drawdown <= 1:
            raise ValueError("payout y max_drawdown deben estar en (0, 1]")
        if self.inactivity_penalty < 0 or self.tie_policy not in ("refund", "loss"):
            raise ValueError("inactivity_penalty >= 0; tie_policy=refund/loss")


@dataclass(frozen=True)
class AlignmentReward:
    """Shaping de M5 binario, en unidades de reward y separado del saldo."""

    opposite_penalty: float = 1.25
    aligned_win_reward: float = 1.75

    def __post_init__(self):
        values = (self.opposite_penalty, self.aligned_win_reward)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("La penalización y el premio de alineación deben ser positivos y finitos")


def market_feature_names(lookback=16):
    return ("rsi_scaled", "atr_fraction", "spread_scaled", *QUANT_FEATURES, *PROBABILITY_FEATURES,
            *(f"log_return_lag_{lag}" for lag in range(lookback)))


def prepare_market_data(df: pd.DataFrame, lookback=16, indicator_period=14, *, market=None):
    """Indicadores y lags estrictamente causales; elimina solo warmup inicial.

    RSI y ATR usan medias exponenciales de Wilder (alpha=1/period, adjust=False).
    spread_pips es opcional y solo informativo; sin columna se usa cero.
    No se aceptan nulos internos para no convertir huecos en velas contiguas.
    El constructor ignora columnas ajenas a la lista explícita de features.
    """
    if not isinstance(lookback, int) or not isinstance(indicator_period, int) or min(lookback, indicator_period) < 1:
        raise ValueError("lookback e indicator_period deben ser enteros positivos")
    required = ["timestamp", "open", "high", "low", "close", *QUANT_FEATURES]
    if not df.columns.is_unique or not set(required).issubset(df.columns):
        raise ValueError(f"Se requieren columnas únicas: {required}")
    config = market or MarketConfig()
    config.check_frame(df)
    out = ensure_probabilities(df,config.expiry_candles).reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, errors="raise")
    ts = out.timestamp
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing or (ts != ts.dt.floor(config.frequency)).any():
        raise ValueError("timestamp debe ser único, ordenado, UTC y alineado al timeframe")
    prices = out[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("OHLC debe ser positivo y finito")
    if (prices.high < prices.max(axis=1)).any() or (prices.low > prices.min(axis=1)).any():
        raise ValueError("OHLC incoherente")
    out[prices.columns] = prices.astype(float)
    delta = out.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/indicator_period, adjust=False, min_periods=indicator_period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/indicator_period, adjust=False, min_periods=indicator_period).mean()
    rsi = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    out["rsi_scaled"] = rsi / 100
    true_range = pd.concat([out.high-out.low, (out.high-out.close.shift()).abs(),
                            (out.low-out.close.shift()).abs()], axis=1).max(axis=1)
    out["atr_fraction"] = true_range.ewm(alpha=1/indicator_period, adjust=False,
                                         min_periods=indicator_period).mean()/out.close
    spread = pd.to_numeric(out["spread_pips"], errors="raise") if "spread_pips" in out else pd.Series(0., index=out.index)
    if not np.isfinite(spread).all() or (spread < 0).any():
        raise ValueError("spread_pips debe ser finito y no negativo")
    out["spread_scaled"] = spread / 10
    returns = np.log(out.close).diff()
    for lag in range(lookback):
        out[f"log_return_lag_{lag}"] = returns.shift(lag)
    cols = list(market_feature_names(lookback))
    out[cols] = out[cols].apply(pd.to_numeric, errors="raise")
    if np.isinf(out[cols].to_numpy()).any():
        raise ValueError("Infinitos en features")
    ready = np.isfinite(out[cols].to_numpy()).all(axis=1)
    if not ready.any():
        raise ValueError("Sin filas completas tras warmup")
    first = int(np.flatnonzero(ready)[0])
    if not ready[first:].all():
        raise ValueError("Nulos/infinitos internos en features después del warmup")
    out = out.iloc[first:].reset_index(drop=True)
    if (out.cond_volatility < 0).any():
        raise ValueError("cond_volatility no puede ser negativa")
    return out


def fit_normalizer(prepared: pd.DataFrame, lookback=16):
    """Estadísticos de entrenamiento únicamente; nunca llamar con validación/test."""
    prepared = ensure_probabilities(prepared,int(prepared.attrs.get("direction_probability_horizon",1)))
    names = market_feature_names(lookback)
    matrix = prepared[list(names)].to_numpy(dtype=float)
    if len(matrix) < 2 or not np.isfinite(matrix).all():
        raise ValueError("Normalización requiere al menos dos filas finitas")
    std = matrix.std(axis=0)
    return dict(features=list(names), mean=matrix.mean(axis=0).tolist(),
                std=np.where(std < 1e-12, 1.0, std).tolist(), clip=10.0)


class BinaryOptionsEnv(gym.Env):
    """0=PUT, 1=NEUTRO, 2=CALL. step devuelve la API Gymnasium de cinco valores.

    Caja: win=+payout*stake, loss=-stake, tie=0 (refund).
    Por defecto RiskConfig aplica filtro causal y shaping de reward separado
    del saldo: PnL/riesgo ex ante, coste de actividad y drawdown incremental.
    NEUTRO sin setup elegible es gratis. Si el setup supera los filtros puede
    tener un coste pequeño y configurable; nunca depende de su resultado futuro.
    Intentos bloqueados tienen coste de reward reducido, sin cambiar el saldo.
    M5 binario admite shaping adicional para la dirección de un setup válido.
    RiskConfig(enabled=False) conserva explícitamente la liquidación/reward v2.
    Al liquidar no queda ninguna posición abierta ni PnL flotante.
    Fin de datos/segmento o episode_length => truncated; insolvencia/DD => terminated.
    reset(options={start_index, balance, peak_balance}) permite evaluación
    continua por segmentos; random_start usa exclusivamente el RNG de Gymnasium.
    """
    metadata = {"render_modes": []}

    def __init__(self, df, *, account=None, lookback=16, indicator_period=14,
                 prepared=False, normalizer=None, random_start=False, episode_length=None,
                 risk=None, market=None, alignment_reward=None):
        super().__init__()
        if market is not None and market.trading_mode != "binary" and type(self) is BinaryOptionsEnv:
            raise ValueError("Usar ForexSpotEnv para forex_spot")
        self.modern_contract = market is not None
        self.market = market or MarketConfig()
        self.market.check_frame(df)
        self.bar_step = self.market.step
        self.expiry_candles = self.market.expiry_candles
        self.account = account or AccountConfig()
        self.risk = risk if risk is not None else (replace(RiskConfig(),neutral_opportunity_penalty=.05) if self.modern_contract else RiskConfig())
        if alignment_reward is not None and (
            self.market.timeframe != "M5" or self.market.trading_mode != "binary" or not self.risk.enabled
        ):
            raise ValueError("La recompensa de alineación requiere M5 binario con filtro activo")
        self.alignment_reward = alignment_reward
        self.lookback = lookback
        frame = ensure_probabilities(df,self.expiry_candles).reset_index(drop=True) if prepared else prepare_market_data(df, lookback, indicator_period,market=self.market)
        names = market_feature_names(lookback)
        required = ["timestamp", "close", *names]
        if not set(required).issubset(frame.columns) or len(frame) < 2:
            raise ValueError("Datos preparados insuficientes/incompletos")
        self._times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).as_unit("ns")
        if (self._times.hasnans or not self._times.is_monotonic_increasing or not self._times.is_unique
                or (self._times != self._times.floor(self.market.frequency)).any()):
            raise ValueError("Timestamps preparados inválidos")
        self._probabilities = frame[PROBABILITY_FEATURES].to_numpy(dtype=float,copy=True)
        self._n_paths = int(frame.attrs.get("quant_config",{}).get("n_paths",10000))
        confidence=self._probabilities[:,:2].max(axis=1)
        lower=mc_probability_lower(confidence,self._n_paths,self.risk.mc_z)
        self._setup_mask=(confidence>self.risk.min_confidence)&(self.account.payout*lower-(1-lower)>self.risk.min_expected_edge)
        self._close = frame.close.to_numpy(dtype=float, copy=True)
        self._sigma = frame.cond_volatility.to_numpy(dtype=float,copy=True)
        matrix = frame[list(names)].to_numpy(dtype=float, copy=True)
        if not np.isfinite(matrix).all() or not np.isfinite(self._close).all() or (self._close <= 0).any():
            raise ValueError("Features/precios preparados deben ser finitos y precios positivos")
        if (frame.cond_volatility.to_numpy(dtype=float) < 0).any():
            raise ValueError("cond_volatility no puede ser negativa")
        if normalizer is not None:
            mean, std = np.asarray(normalizer["mean"]), np.asarray(normalizer["std"])
            clip = normalizer["clip"]
            if (list(names) != normalizer["features"] or mean.shape != (len(names),)
                    or std.shape != mean.shape or not np.isfinite(mean).all()
                    or not np.isfinite(std).all() or (std <= 0).any() or not np.isfinite(clip) or clip <= 0):
                raise ValueError("Normalizador incompatible o inválido")
            matrix = np.clip((matrix-mean)/std, -clip, clip)
        self._market = matrix.astype(np.float32)
        if not np.isfinite(self._market).all():
            raise ValueError("Features fuera del rango float32")
        breaks = np.flatnonzero(np.diff(self._times.asi8) != self.bar_step.value) + 1
        boundaries = np.r_[0, breaks, len(frame)]
        self.segments = tuple((int(a), int(b-1)) for a, b in zip(boundaries[:-1], boundaries[1:]) if b-a > self.expiry_candles)
        if not self.segments:
            raise ValueError("No hay ventanas contiguas suficientes para la expiración")
        self._eligible = np.concatenate([np.arange(a, b-self.expiry_candles+1) for a, b in self.segments])
        if episode_length is not None and (not isinstance(episode_length, int) or episode_length < self.expiry_candles):
            raise ValueError("episode_length debe ser entero positivo o None")
        self.random_start, self.episode_length = random_start, episode_length
        self.feature_names = (*names, *ACCOUNT_FEATURES)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.feature_names),), dtype=np.float32)
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        start = options.get("start_index")
        if start is None:
            start = int(self.np_random.choice(self._eligible)) if self.random_start else self.segments[0][0]
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or start not in self._eligible:
            raise ValueError("start_index debe permitir liquidar en una vela M15 consecutiva")
        end = next(b for a, b in self.segments if a <= start < b)
        self._index = int(start)
        self._end = min(end, start+self.episode_length) if self.episode_length else end
        self.balance = float(options.get("balance", self.account.initial_balance))
        self.peak_balance = float(options.get("peak_balance", max(self.balance, self.account.initial_balance)))
        if (not np.isfinite([self.balance, self.peak_balance]).all()
                or self.balance < self.account.stake or self.peak_balance < self.balance):
            raise ValueError("Saldo/pico inicial inválidos")
        self.previous_direction = 0.0
        self.previous_reward = 0.0
        self.trade_activity = float(options.get("trade_activity",0.))
        if not np.isfinite(self.trade_activity) or not 0<=self.trade_activity<=1:raise ValueError("trade_activity inválida")
        self.trades = self.wins = self.losses = self.ties = self.steps = 0
        self.max_drawdown = max(0., 1-self.balance/self.peak_balance)
        self._done = False
        return self._observation(), self._info()

    def _observation(self):
        state = np.array([self.balance/self.account.initial_balance-1,
                          1-self.balance/self.peak_balance, self.previous_direction,
                          self.previous_reward, self.account.stake/max(self.balance, 1e-12),
                          self.account.payout, self.trade_activity], dtype=np.float32)
        return np.concatenate([self._market[self._index], state]).astype(np.float32)

    def _info(self):
        return dict(index=self._index, timestamp=self._times[self._index].isoformat(),
                    decision_time=(self._times[self._index]+self.bar_step).isoformat(),
                    balance=self.balance, peak_balance=self.peak_balance,
                    drawdown=1-self.balance/self.peak_balance, max_drawdown=self.max_drawdown,
                    trades=self.trades, wins=self.wins, losses=self.losses, ties=self.ties,
                    open_positions=0,trade_activity=self.trade_activity)

    def step(self, action):
        if self._done:
            raise RuntimeError("Llamar reset antes de step y después del fin del episodio")
        if not self.action_space.contains(action):
            raise ValueError("Acción inválida: 0=PUT, 1=NEUTRO, 2=CALL")
        requested_action = int(action)
        decision = resolve_action(requested_action,*self._probabilities[self._index],
                                  payout=self.account.payout,tie_policy=self.account.tie_policy,
                                  risk=self.risk,n_paths=self._n_paths)
        action = decision['action']
        # Para NEUTRO, resolve_action ya calcula confianza/edge de la mejor dirección.
        # Todo se determina antes de mirar el cierre de liquidación.
        missed_opportunity = bool(self.risk.enabled and requested_action==1
            and decision['mc_confidence']>self.risk.min_confidence
            and decision['expected_edge_lower']>self.risk.min_expected_edge)
        old_drawdown = 1-self.balance/self.peak_balance
        entry_index = self._index
        setup_action = 1
        if self.alignment_reward is not None and self._setup_mask[entry_index]:
            setup_action = 2 if decision['mc_prob_up'] > decision['mc_prob_down'] else 0
        opposed_setup = setup_action != 1 and requested_action == 2 - setup_action
        settlement_index = entry_index+self.expiry_candles
        entry, settlement = self._close[entry_index],self._close[settlement_index]
        profit, reward, outcome = 0., (0. if self.risk.enabled else -self.account.inactivity_penalty), "neutral"
        if action != 1:
            self.trades += 1
            if settlement == entry:
                self.ties += 1
                outcome = "tie"
                reward = 0. if self.account.tie_policy == "refund" else -1.
            elif (action == 2 and settlement > entry) or (action == 0 and settlement < entry):
                self.wins += 1
                outcome, reward = "win", self.account.payout
            else:
                self.losses += 1
                outcome, reward = "loss", -1.
            profit = self.account.stake*reward
            self.balance += profit
        self.peak_balance = max(self.peak_balance, self.balance)
        self.max_drawdown = max(self.max_drawdown, 1-self.balance/self.peak_balance)
        self._index += self.expiry_candles if action!=1 else 1  # sin solapamientos; esperar avanza una barra.
        self.steps += 1
        self.previous_direction = float(action-1)
        self.trade_activity = self.risk.activity_decay*self.trade_activity+(1-self.risk.activity_decay)*float(action!=1)
        limit = min(self.account.max_drawdown,self.risk.drawdown_limit) if self.risk.enabled else self.account.max_drawdown
        if self.risk.enabled:
            if action!=1:
                win = decision['mc_prob_up'] if action==2 else decision['mc_prob_down']
                loss = decision['mc_prob_down'] if action==2 else decision['mc_prob_up']
                if self.account.tie_policy=='loss':loss += decision['mc_prob_tie']
                mean = self.account.payout*win-loss
                scale = np.sqrt(max(self.account.payout**2*win+loss-mean**2,self.risk.risk_std_floor**2))
                reward = (profit/self.account.stake)/scale
                if self.alignment_reward is not None and action == setup_action and outcome == "win":
                    reward = self.alignment_reward.aligned_win_reward
                reward -= self.risk.trade_penalty
                reward -= self.risk.overtrade_penalty*max(0.,self.trade_activity-self.risk.target_activity)
            elif decision['blocked']:
                if opposed_setup:
                    reward = -self.alignment_reward.opposite_penalty
                elif not self.modern_contract or self._setup_mask[entry_index]:
                    reward = -self.risk.blocked_penalty
            elif missed_opportunity:
                reward = -self.risk.neutral_opportunity_penalty
            reward -= self.risk.drawdown_penalty*max(0.,1-self.balance/self.peak_balance-old_drawdown)
            if self.max_drawdown+1e-12>=limit:reward -= self.risk.stop_penalty
        self.previous_reward = float(reward)
        terminated = self.balance < self.account.stake or self.max_drawdown + 1e-12 >= limit
        truncated = not terminated and self._index+self.expiry_candles > self._end
        self._done = bool(terminated or truncated)
        info = self._info()
        info.update(**decision)
        if self.alignment_reward is not None:
            info['mc_setup_action'] = setup_action
        info.update(action=action,missed_opportunity=missed_opportunity, entry_index=entry_index, entry_price=float(entry),
                    settlement_price=float(settlement), profit=float(profit), outcome=outcome,
                    expiry_time=(self._times[self._index]+self.bar_step).isoformat(),
                    end_reason=("insufficient_balance" if self.balance < self.account.stake else
                                "max_drawdown" if terminated else "data_or_time_limit" if truncated else None))
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

