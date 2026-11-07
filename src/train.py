"""Entrena PPO para opciones binarias M15 y selecciona pesos por validación.

Pipeline: Parquet causal -> indicadores/lags causales -> split cronológico
70/20/10 (train/validation/test reservado) -> normalizador SOLO train -> PPO.
La validación compara un score de retorno neto ajustado por drawdown y actividad, con stake fijo, transportando saldo y
pico entre segmentos continuos. Nunca se ejecutan contratos sobre un hueco.
Se valida al comenzar, periódicamente y al terminar; incluso un entrenamiento
corto produce best_ppo_agent.zip. Si la política inicial gana la comparación,
puede ser el checkpoint elegido: no se promete rentabilidad ni aprendizaje.
El holdout test se registra pero NO se ejecuta ni selecciona pesos con él.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from .market import MarketConfig,PROTECTED_HOLDOUT_START
from .agent import build_agent
from .exploration import ExplorationAuditCallback,policy_snapshot
from .decision import RiskConfig
from .metrics import decision_metrics
from .artifacts import file_sha256
from .environment import (AccountConfig, BinaryOptionsEnv, fit_normalizer,
                          market_feature_names, prepare_market_data)

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def write_json(path, payload):
    """Metadatos atómicos, sin permitir NaN/Infinity silenciosos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def atomic_save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".zip")
    os.close(fd)
    try:
        model.save(temp)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def chronological_split(prepared, *, train_fraction=.70, validation_fraction=.20, purge_bars=1):
    """Subconjuntos disjuntos. Ningún step accede al close de otro subconjunto.

    purge_bars descarta filas iniciales de validation y test como separación
    adicional. El forecast MC de 32 barras es una feature, no una etiqueta
    futura; no requiere purgar 32 barras para una expiración binaria de una.
    Los indicadores de validación pueden usar pasado de train legítimamente.
    """
    if (not np.isfinite([train_fraction, validation_fraction]).all()
            or min(train_fraction, validation_fraction) <= 0
            or train_fraction+validation_fraction >= 1):
        raise ValueError("Fracciones positivas cuya suma sea < 1; el resto se reserva para test")
    if not isinstance(purge_bars, int) or purge_bars < 0:
        raise ValueError("purge_bars debe ser entero no negativo")
    n = len(prepared)
    train_end = int(n*train_fraction)
    validation_end = int(n*(train_fraction+validation_fraction))
    pieces = dict(train=prepared.iloc[:train_end].copy().reset_index(drop=True),
                  validation=prepared.iloc[train_end+purge_bars:validation_end].copy().reset_index(drop=True),
                  test=prepared.iloc[validation_end+purge_bars:].copy().reset_index(drop=True))
    if any(len(frame) < 2 for frame in pieces.values()):
        raise ValueError("Histórico insuficiente para train/validation/test y purga")
    return pieces


def chronological_market_split(prepared,market,*,train_fraction=.70,validation_fraction=.20,purge_bars=1):
    """Protege también el holdout histórico ABSOLUTO al cambiar timeframe.

    La agrupación/warmup pueden mover el percentil 90 de fechas. Si lo mueven
    hacia adelante se particiona solo el prefijo anterior a la frontera PR9.
    Ninguna OHLC cuya barra cierre después de esa frontera entra a train/val.
    """
    pieces=chronological_split(prepared,train_fraction=train_fraction,
        validation_fraction=validation_fraction,purge_bars=purge_bars)
    if pieces['test'].timestamp.iloc[0]>PROTECTED_HOLDOUT_START:
        prefix=prepared.loc[prepared.timestamp+market.step<=PROTECTED_HOLDOUT_START]
        end=int(len(prefix)*train_fraction/(train_fraction+validation_fraction))
        pieces=dict(train=prefix.iloc[:end].reset_index(drop=True),
            validation=prefix.iloc[end+purge_bars:].reset_index(drop=True),
            test=prepared.loc[prepared.timestamp>=PROTECTED_HOLDOUT_START].reset_index(drop=True))
    for phase in ('train','validation'):
        pieces[phase]=pieces[phase].loc[pieces[phase].timestamp+market.step<=PROTECTED_HOLDOUT_START].reset_index(drop=True)
    if any(len(f)<2 for f in pieces.values()):
        raise ValueError('Datos insuficientes antes de la frontera protegida del holdout PR9')
    return pieces


def evaluate_policy(model, env: BinaryOptionsEnv):
    """Recorre TODA validación en orden, sin recapitalizar tras fines de semana."""
    if getattr(env,"modern_contract",False):
        from .market_evaluation import evaluate_market_policy
        return evaluate_market_policy(model,env)
    balance, peak = env.account.initial_balance, env.account.initial_balance
    total_reward = 0.
    counts = {"put":0, "neutral":0, "call":0}
    wins = losses = ties = trades = steps = 0
    max_drawdown = 0.
    stopped = False
    decisions = []
    policy_observations=[]
    activity = 0.
    for start, _ in env.segments:
        obs, _ = env.reset(options=dict(start_index=start, balance=balance, peak_balance=peak,trade_activity=activity))
        while True:
            if steps%128==0 and len(policy_observations)<256 and hasattr(model,'policy'):
                policy_observations.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action)
            counts[("put", "neutral", "call")[info["action"]]] += 1
            decisions.append(info.copy())
            activity = info["trade_activity"]
            total_reward += reward
            steps += 1
            balance, peak = info["balance"], info["peak_balance"]
            max_drawdown = max(max_drawdown, info["max_drawdown"])
            if terminated or truncated:
                wins += info["wins"]
                losses += info["losses"]
                ties += info["ties"]
                trades += info["trades"]
                stopped = terminated
                break
        if stopped:
            break
    analysis = decision_metrics(pd.DataFrame(decisions),payout=env.account.payout)
    score = (balance/env.account.initial_balance-1-env.risk.drawdown_penalty*max_drawdown
             -env.risk.trade_penalty*trades/max(steps,1)
             -env.risk.blocked_penalty*analysis["blocked_attempts"]/max(steps,1)
             -env.risk.neutral_opportunity_penalty*analysis["missed_opportunities"]/max(steps,1))
    if not env.risk.enabled:score = balance-env.account.initial_balance
    return dict(policy_snapshot=policy_snapshot(model,policy_observations) if policy_observations else None,selection_score=score,decision_analysis=analysis,net_profit=balance-env.account.initial_balance, final_balance=balance,
                total_reward=total_reward, mean_reward=total_reward/steps if steps else 0.,
                max_drawdown=max_drawdown, trades=trades, wins=wins, losses=losses,
                ties=ties, win_rate=wins/trades if trades else 0., steps=steps,
                signals=counts, risk_stop=bool(stopped))


class BestValidationCallback(BaseCallback):
    """Selecciona por score de riesgo de validación, sin mirar el holdout."""
    def __init__(self, validation_env, output_dir, eval_freq=25_000, verbose=1):
        super().__init__(verbose)
        if eval_freq < 1:
            raise ValueError("eval_freq debe ser positivo")
        self.validation_env, self.output_dir = validation_env, Path(output_dir)
        self.eval_freq = int(eval_freq)
        self.best_score = -np.inf
        self.best_profit = -np.inf
        self.history = []
        self.next_eval = self.eval_freq
        self.last_eval = -1

    def _evaluate(self):
        metrics = evaluate_policy(self.model, self.validation_env)
        record = dict(timesteps=int(self.num_timesteps), **metrics)
        self.history.append(record)
        self.last_eval = self.num_timesteps
        if metrics["selection_score"] > self.best_score:
            self.best_score = metrics["selection_score"]
            self.best_profit = metrics["net_profit"]
            atomic_save_model(self.model, self.output_dir/"best_ppo_agent.zip")
            write_json(self.output_dir/"best_validation.json", record)
        write_json(self.output_dir/"validation_history.json", self.history)
        self.logger.record("validation/net_profit", metrics["net_profit"])
        self.logger.record("validation/trades", metrics["trades"])
        LOG.info("Decisiones/confianza validación: %s",json.dumps(metrics["decision_analysis"],allow_nan=False))
        LOG.info("Validación step=%d PnL=%.2f trades=%d DD=%.4f",
                 self.num_timesteps, metrics["net_profit"], metrics["trades"], metrics["max_drawdown"])

    def _on_training_start(self):
        self._evaluate()

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
            self.next_eval = self.num_timesteps+self.eval_freq
        return True

    def _on_training_end(self):
        # Evalúa otra vez aun si coincide el timestep: la última actualización
        # PPO sucede después del callback de recolección de ese mismo timestep.
        self._evaluate()


def train(input_path, *, output_dir=ROOT/"models", total_timesteps=100_000,
          train_fraction=.70, validation_fraction=.20, purge_bars=1,
          lookback=16, indicator_period=14, episode_length=512,
          account=None, risk=None, seed=42, n_steps=2048, batch_size=64,
          n_epochs=10, eval_freq=25_000, learning_rate=3e-4, ent_coef=.02, verbose=1, market=None, opportunity_training=True, trading_profile=None):
    """Devuelve manifiesto; guarda mejores pesos, pesos finales y normalización."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps debe ser positivo")
    account = account or AccountConfig()
    risk = risk if risk is not None else (replace(RiskConfig(),neutral_opportunity_penalty=.05) if market is not None else RiskConfig())
    input_path, output_dir = Path(input_path), Path(output_dir)
    source_hash = file_sha256(input_path)
    raw = pd.read_parquet(input_path)
    if file_sha256(input_path) != source_hash:
        raise ValueError("El Parquet cambió durante la lectura")
    prepared = prepare_market_data(raw, lookback, indicator_period,market=market)
    splitter = chronological_split if market is None else lambda frame,**kw:chronological_market_split(frame,market,**kw)
    splits = splitter(prepared, train_fraction=train_fraction,
                                  validation_fraction=validation_fraction, purge_bars=purge_bars)
    normalizer = fit_normalizer(splits["train"], lookback)
    if market is not None: market.check_frame(raw)
    common = dict(account=account, risk=risk, lookback=lookback, prepared=True, normalizer=normalizer)
    if market is not None: common['market']=market
    env_class = BinaryOptionsEnv
    if market is not None and market.trading_mode=='forex_spot':
        from .forex_spot import ForexSpotEnv
        env_class=ForexSpotEnv
    training_env = env_class(splits["train"], random_start=True,
                                    episode_length=episode_length, **common)
    validation_env = env_class(splits["validation"], **common)
    if market is not None and market.trading_mode=='binary' and opportunity_training and risk.enabled:
        from .opportunities import OpportunityTraining
        training_env=OpportunityTraining(training_env)
    check_env(training_env, warn=True, skip_render_check=True)
    training_env.reset(seed=seed)
    validation_env.reset(seed=seed+1)
    manifest = dict(manifest_version=3, input_file=str(input_path.resolve()), input_sha256=source_hash,
                    account=asdict(account), risk=asdict(risk), reward_version="risk_adjusted_v2", seed=seed, lookback=lookback,
                    indicator_period=indicator_period, episode_length=episode_length,
                    requested_timesteps=total_timesteps, n_steps=n_steps, batch_size=batch_size,
                    n_epochs=n_epochs, learning_rate=learning_rate, ent_coef=ent_coef, eval_freq=eval_freq,
                    train_fraction=train_fraction, validation_fraction=validation_fraction,
                    purge_bars=purge_bars, normalizer_fit="train only", selection="validation risk_adjusted_score",
                    features=list(market_feature_names(lookback)), holdout_test_evaluated=False, holdout_use="research; previously examined historical snapshot, requires fresh data for confirmation",
                    exploration_evaluation="stochastic rollout audit; deterministic validation/holdout",
                    source_quant_metadata=raw.attrs,
                    splits={name:dict(rows=len(frame), start=frame.timestamp.iloc[0].isoformat(),
                                      end=frame.timestamp.iloc[-1].isoformat())
                            for name, frame in splits.items()},
                    versions={name:importlib.metadata.version(name)
                              for name in ("numpy", "pandas", "torch", "gymnasium", "stable-baselines3")})
    if market is not None:
        manifest.update(manifest_version=4,market_config=asdict(market),
            protected_holdout_start=PROTECTED_HOLDOUT_START.isoformat(),
            reward_version='opportunity_v3' if market.trading_mode=='binary' else 'spot_equity_v1',
            opportunity_training=bool(opportunity_training and market.trading_mode=='binary' and risk.enabled),
            discount_clock='eligible decisions' if opportunity_training and market.trading_mode=='binary' and risk.enabled else 'bars',
            observation_features=list(validation_env.feature_names),
            selection='validation opportunity risk score',artifact_status='research only; no automatic promotion')
        if market.trading_mode=='forex_spot': manifest['spot_config']=asdict(validation_env.spot)
        if trading_profile is not None:
            manifest['trading_profile'] = trading_profile
            if trading_profile == 'scalping' and market.trading_mode == 'binary':
                from .trading_profiles import SCALPING_TARGET_PER_HOUR
                manifest['frequency_target_per_hour'] = list(SCALPING_TARGET_PER_HOUR)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir/"normalizer.json", normalizer)
    write_json(output_dir/"training_manifest.json", manifest)
    normalizer_hash = file_sha256(output_dir/"normalizer.json")
    monitored = Monitor(training_env)
    callback = BestValidationCallback(validation_env, output_dir, eval_freq, verbose)
    try:
        model = build_agent(monitored, seed=seed, n_steps=n_steps, batch_size=batch_size,
                            n_epochs=n_epochs, learning_rate=learning_rate, ent_coef=ent_coef, verbose=verbose)
        model.learn(total_timesteps=total_timesteps, callback=[ExplorationAuditCallback(output_dir),callback], progress_bar=False)
        atomic_save_model(model, output_dir/"final_ppo_agent.zip")
        if file_sha256(input_path) != source_hash or file_sha256(output_dir/"normalizer.json") != normalizer_hash:
            raise ValueError("Datos o normalizador cambiaron durante entrenamiento")
        manifest["normalizer_sha256"] = normalizer_hash
        manifest["best_model_sha256"] = file_sha256(output_dir/"best_ppo_agent.zip")
        manifest["actual_timesteps"] = int(model.num_timesteps)
        if market is not None:
            selected=json.loads((output_dir/'best_validation.json').read_text())
            manifest.update(selected_checkpoint_timesteps=selected['timesteps'],
                selected_checkpoint_trained=bool(selected['timesteps']>0),
                initial_validation_score=callback.history[0]['selection_score'],
                candidate_beats_initial_score=bool(callback.best_score>callback.history[0]['selection_score']))
        manifest["best_validation_net_profit"] = float(callback.best_profit)
        manifest["best_validation_score"] = float(callback.best_score)
        manifest["best_model"] = str((output_dir/"best_ppo_agent.zip").resolve())
        write_json(output_dir/"training_manifest.json", manifest)
        if market is not None:
            from .agent import load_agent
            from .market_evaluation import evaluate_market_policy
            evaluate_market_policy(load_agent(output_dir/'best_ppo_agent.zip'), validation_env,
                                   output_dir/'validation', trading_profile=trading_profile)
    finally:
        monitored.close()
        validation_env.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"data/eurusd_m15_features.parquet")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--train-fraction", type=float, default=.70)
    parser.add_argument("--validation-fraction", type=float, default=.20)
    parser.add_argument("--purge-bars", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--indicator-period", type=int, default=14)
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--initial-balance", type=float, default=1000.)
    parser.add_argument("--stake", type=float, default=10.)
    parser.add_argument("--payout", type=float, default=.8)
    parser.add_argument("--tie-policy", choices=["refund", "loss"], default="refund")
    parser.add_argument("--inactivity-penalty", type=float, default=0.)
    parser.add_argument("--max-drawdown", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument('--min-confidence',type=float)
    parser.add_argument('--min-expected-edge',type=float,default=.02)
    parser.add_argument('--risk-drawdown-limit',type=float,default=.20)
    parser.add_argument('--trade-penalty',type=float,default=RiskConfig().trade_penalty)
    parser.add_argument('--blocked-penalty',type=float,default=RiskConfig().blocked_penalty)
    parser.add_argument('--neutral-opportunity-penalty',type=float)
    parser.add_argument('--ent-coef',type=float)
    parser.add_argument('--overtrade-penalty',type=float,default=2.)
    parser.add_argument('--drawdown-penalty',type=float,default=10.)
    from .trading_profiles import PROFILES, resolve_profile
    parser.add_argument('--trading-profile', choices=['auto', *PROFILES])
    parser.add_argument('--timeframe',choices=['M1','M5','M15','H1','H4'])
    parser.add_argument('--trading-mode',choices=['binary','forex_spot'],default='binary')
    parser.add_argument('--expiry-candles',type=int)
    parser.add_argument('--no-opportunity-training',action='store_true')
    args = vars(parser.parse_args())
    timeframe = args.pop('timeframe')
    trading_mode = args.pop('trading_mode')
    expiry = args.pop('expiry_candles')
    profile = args.pop('trading_profile')
    confidence = args.pop('min_confidence')
    neutral_cost = args.pop('neutral_opportunity_penalty')
    entropy = args.pop('ent_coef')
    target_activity = RiskConfig().target_activity
    if profile is None:
        market = MarketConfig(timeframe or 'M15', trading_mode, 1 if expiry is None else expiry)
        confidence = RiskConfig().min_confidence if confidence is None else confidence
        neutral_cost = .05 if neutral_cost is None else neutral_cost
        entropy = .02 if entropy is None else entropy
    else:
        resolved = resolve_profile(
            profile=profile, timeframe=timeframe or 'auto', trading_mode=trading_mode,
            expiry_candles=0 if expiry is None else expiry,
            confidence='auto' if confidence is None else confidence,
            entropy='auto' if entropy is None else entropy,
            neutral_cost='auto' if neutral_cost is None else neutral_cost,
        )
        profile = resolved['FOREX_TRADING_PROFILE']
        market = MarketConfig(resolved['FOREX_TIMEFRAME'], trading_mode, resolved['FOREX_EXPIRY_CANDLES'])
        confidence = resolved['FOREX_MIN_CONFIDENCE']
        neutral_cost = resolved['FOREX_NEUTRAL_OPPORTUNITY_PENALTY']
        entropy = resolved['FOREX_ENT_COEF']
        target_activity = resolved['FOREX_TARGET_ACTIVITY']
    opportunity_training=not args.pop('no_opportunity_training')
    risk = RiskConfig(min_confidence=confidence,target_activity=target_activity,min_expected_edge=args.pop('min_expected_edge'),
                      drawdown_limit=args.pop('risk_drawdown_limit'),trade_penalty=args.pop('trade_penalty'),
                      blocked_penalty=args.pop('blocked_penalty'),overtrade_penalty=args.pop('overtrade_penalty'),
                      drawdown_penalty=args.pop('drawdown_penalty'),neutral_opportunity_penalty=neutral_cost)
    account = AccountConfig(**{key:args.pop(key) for key in
                              ("initial_balance", "stake", "payout", "tie_policy",
                               "inactivity_penalty", "max_drawdown")})
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    input_path = args.pop("input")
    if args['output_dir'] is None:
        args['output_dir'] = ROOT/'models' if profile is None else (
            ROOT/'models/research_v4'/profile/market.timeframe/market.trading_mode/f'e{market.expiry_candles}'
        )
    result = train(input_path, account=account, risk=risk, market=market,opportunity_training=opportunity_training,
                   trading_profile=profile,ent_coef=entropy, **args)
    LOG.info("Mejor modelo: %s; test reservado sin evaluar", result["best_model"])


if __name__ == "__main__":
    main()

