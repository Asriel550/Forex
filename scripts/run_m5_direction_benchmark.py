"""Compara probabilidades de dirección M5 sobre admisiones fijas y sin stop."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_m5_direction_ablation import (
    FROZEN_MARCH_ADMISSIONS_SHA256, FROZEN_SEED_SHA256, founder_observations, load_features,
)
from src.decision import RiskConfig
from src.environment import BinaryOptionsEnv, prepare_market_data
from src.evolution_context import CONTEXT_FEATURES, m15_direction_bias
from src.evolution_direction import FrozenDirectionSimulation, frozen_simulation, unit_payoff
from src.evolution_engine import M5Simulation
from src.evolution_lineage import MARKET, MARKET_FEATURES, load_founder
from src.market import PROTECTED_HOLDOUT_START
from src.quant_engine import QuantConfig
from src.train import write_json


SEEDS = Path('config/m5-direction-benchmark')
ANCHOR_SHA256 = 'ca376c5775fb1b3ec2f72259b5e8fc05afb17596664bacd6008c9d31a26c36e5'
FIT_START = pd.Timestamp('2025-01-01T00:00:00Z')
CALIBRATION_START = pd.Timestamp('2025-11-26T00:00:00Z')
EVALUATION_START = pd.Timestamp('2026-01-01T00:00:00Z')
LABELS = np.array([-1, 0, 1])
MC_COLUMNS = ('mc_prob_down', 'mc_prob_tie', 'mc_prob_up')
MODEL_FEATURES = (*MARKET_FEATURES, *CONTEXT_FEATURES, 'm15_bias')
MIN_DAYS_FOR_INTERVAL = 10
BOOTSTRAP_SAMPLES = 4000
SEED = 42
MIN_PROBABILITY = 1e-8


@dataclass(frozen=True)
class AdmissionBlock:
    rows: pd.DataFrame
    eligible_bars: int
    observed_hours: float
    mask_sha256: str


def admission_blocks(raw, founder, windows, *, anchor=None, fixed_opens=None):
    """Reproduce la admisión original y liquida solo con el close M5 siguiente."""
    if raw.empty or (raw.timestamp + MARKET.step > PROTECTED_HOLDOUT_START).any():
        raise ValueError('El benchmark requiere velas anteriores al holdout protegido')
    prepared = prepare_market_data(raw, market=MARKET)
    context, normalized, market_normalizer = founder_observations(prepared, founder)
    bias = m15_direction_bias(prepared)
    result = {}
    for name, start, end in windows:
        if not start < end <= PROTECTED_HOLDOUT_START:
            raise ValueError(f'Ventana inválida para {name}')
        frame = prepared.loc[(prepared.timestamp >= start) & (prepared.timestamp + MARKET.step <= end)]
        env = BinaryOptionsEnv(frame, market=MARKET, prepared=True, account=founder.account,
                               normalizer=market_normalizer,
                               risk=RiskConfig(enabled=False, drawdown_limit=founder.drawdown_limit))
        sim = M5Simulation(env, normalized.loc[env._times])
        frozen = FrozenDirectionSimulation(sim, founder.genome)
        proposed = frozen.proposed
        if fixed_opens is not None:
            proposed = env._times[sim.entries].get_indexer(fixed_opens[name])
            if (proposed < 0).any() or (np.diff(proposed) <= 0).any():
                raise ValueError(f'La máscara archivada de {name} requiere aperturas elegibles y ordenadas')
        admitted = frame.iloc[sim.entries[proposed]]
        opens = env._times[sim.entries[proposed]]
        rows = pd.concat((admitted.set_index('timestamp').loc[:, list(MARKET_FEATURES)],
                          context.loc[opens], bias.loc[opens].rename('m15_bias')), axis=1)
        rows.index = opens + MARKET.step
        rows['outcome'] = sim.change[proposed].astype(np.int8)
        rows['mc_side'] = frozen.mc_side[proposed].astype(np.int8)
        if anchor is not None:
            reference, m15_config, m15_weight = anchor
            trial = frozen_simulation(sim, founder.genome, frame, bias, m15_config)
            _, direction = trial.direction_scores(reference.genome, m15_weight=m15_weight)
            rows['gen306_side'] = np.sign(direction[proposed]).astype(np.int8)
        result[name] = AdmissionBlock(rows, len(sim.entries), sim.observed_hours,
                                      sha256(opens.asi8.tobytes()).hexdigest())
    return result, prepared


def fit_logistic(features, outcomes):
    """Mantiene el orden PUT, empate, CALL en todas las probabilidades."""
    if not np.array_equal(np.unique(outcomes), LABELS):
        raise ValueError('Se requieren las tres clases en el ajuste y la calibración')
    return make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=SEED)).fit(
        features, outcomes,
    )


def log_probabilities(probabilities):
    return np.log(np.clip(probabilities, MIN_PROBABILITY, 1.))


def fit_models(fit, calibration):
    if (fit.empty or calibration.empty or fit.index[-1] + MARKET.step > calibration.index[0]):
        raise ValueError('El ajuste y la calibración deben ser cronológicos y disjuntos')
    logistic = fit_logistic(fit.loc[:, list(MODEL_FEATURES)], fit.outcome)
    calibration_probs = logistic.predict_proba(calibration.loc[:, list(MODEL_FEATURES)])
    calibrate_logistic = fit_logistic(log_probabilities(calibration_probs), calibration.outcome)
    calibrate_mc = fit_logistic(log_probabilities(calibration.loc[:, list(MC_COLUMNS)].to_numpy()),
                                calibration.outcome)
    return logistic, calibrate_logistic, calibrate_mc


def bootstrap_days(values, timestamps):
    """Reagrupa el diferencial pareado por día UTC antes de remuestrear."""
    indices, days = pd.factorize(timestamps.normalize(), sort=True)
    if len(days) < MIN_DAYS_FOR_INTERVAL:
        return None
    totals = np.bincount(indices, weights=values)
    counts = np.bincount(indices)
    samples = np.random.default_rng(SEED).integers(len(days), size=(BOOTSTRAP_SAMPLES, len(days)))
    distribution = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return np.quantile(distribution, [.025, .975]).tolist()


def summarize(rows, probabilities, account):
    """Puntúa los mismos contratos, con denominador fijo y sin truncar la secuencia."""
    if rows.empty:
        return dict(admissions=0, days=0, policies={})
    actual = rows.outcome.to_numpy()
    mc_side = rows.mc_side.to_numpy()
    mc_payoff = unit_payoff(mc_side, actual, account)
    sides = {'mc': mc_side, 'gen306': rows.gen306_side.to_numpy()}
    for name in ('mc_calibrated', 'logistic_calibrated'):
        margin = probabilities[name][:, 2] - probabilities[name][:, 0]
        sides[name] = np.where(margin != 0., np.sign(margin), mc_side)

    policies = {}
    truth = actual[:, None] == LABELS
    for name, side in sides.items():
        payoff = unit_payoff(side, actual, account)
        delta = payoff - mc_payoff
        wins, losses = int((payoff > 0.).sum()), int((payoff < 0.).sum())
        stats = dict(
            wins=wins, losses=losses, refunds=len(actual) - wins - losses,
            win_rate_excluding_refunds=wins / (wins + losses) if wins + losses else None,
            profit_factor=account.payout * wins / losses if losses else None,
            pnl=float(payoff.sum() * account.stake),
            return_per_admission=float(payoff.mean()),
            return_ci95=bootstrap_days(payoff, rows.index),
            delta_vs_mc=float(delta.mean()),
            delta_vs_mc_ci95=bootstrap_days(delta, rows.index),
            direction_disagreements_vs_mc=int((side != mc_side).sum()),
        )
        if name in probabilities:
            probs = probabilities[name]
            stats['brier'] = float(np.sum((probs - truth) ** 2, axis=1).mean())
            stats['log_loss'] = float(-log_probabilities(probs[truth]).mean())
        policies[name] = stats
    return dict(admissions=len(actual), days=int(rows.index.normalize().nunique()), policies=policies)


def evaluate_block(block, models, account, cutoff):
    rows = block.rows
    logistic, calibrate_logistic, calibrate_mc = models
    probabilities = {}
    if not rows.empty:
        raw_mc = rows.loc[:, list(MC_COLUMNS)].to_numpy()
        raw_logistic = logistic.predict_proba(rows.loc[:, list(MODEL_FEATURES)])
        probabilities = dict(
            mc=raw_mc,
            mc_calibrated=calibrate_mc.predict_proba(log_probabilities(raw_mc)),
            logistic_calibrated=calibrate_logistic.predict_proba(log_probabilities(raw_logistic)),
        )
    high = rows.cond_volatility.to_numpy() > cutoff
    regimes = {}
    for name, mask in (('all', np.ones(len(rows), dtype=bool)), ('moderate', ~high), ('high', high)):
        regimes[name] = summarize(rows.loc[mask], {k: v[mask] for k, v in probabilities.items()}, account)
    return dict(eligible_bars=block.eligible_bars, observed_hours=block.observed_hours,
                admissions_per_observed_hour=len(rows) / block.observed_hours,
                mask_sha256=block.mask_sha256, regimes=regimes,
                first_admission=rows.index[0].isoformat() if len(rows) else None,
                last_expiry=(rows.index[-1] + MARKET.step).isoformat() if len(rows) else None)


def load_references():
    bundles = [(SEEDS/'gen268.zip', FROZEN_SEED_SHA256), (SEEDS/'gen306.zip', ANCHOR_SHA256)]
    for bundle, digest in bundles:
        if sha256(bundle.read_bytes()).hexdigest() != digest:
            raise ValueError(f'SHA256 incorrecto para {bundle.name}')
    founder_path, anchor_path = (bundle for bundle, _ in bundles)
    with ZipFile(founder_path) as archive:
        founder_manifest = json.loads(archive.read('manifest.json'))
    with ZipFile(anchor_path) as archive:
        anchor_manifest = json.loads(archive.read('manifest.json'))
    for manifest in (founder_manifest, anchor_manifest):
        if any(pd.Timestamp(split['end']) + MARKET.step > CALIBRATION_START for split in manifest['splits'].values()):
            raise ValueError('Un checkpoint usó datos posteriores al inicio de calibración')
    founder = load_founder(founder_path, None, CALIBRATION_START, CALIBRATION_START)
    reference = load_founder(anchor_path, None, CALIBRATION_START, CALIBRATION_START, allow_direction_ablation=True)
    ablation = anchor_manifest['direction_ablation']
    if ablation['frozen_entry_bundle_sha256'] != FROZEN_SEED_SHA256 or ablation['m15']['arm'] != 'anchor':
        raise ValueError('La referencia direccional requiere las entradas de la Gen 268')
    anchor = (reference, ablation['m15'], float(anchor_manifest['selection']['m15_weight']))
    return founder, anchor, QuantConfig.from_metadata(founder_manifest['quant_config'])


def run_benchmark(raw_dir, output):
    founder, anchor, quant = load_references()
    history = load_features(raw_dir, ['2025'], quant)
    development, prepared = admission_blocks(history, founder, (
        ('fit', FIT_START, CALIBRATION_START), ('calibration', CALIBRATION_START, EVALUATION_START),
    ))
    fit, calibration = (development[name].rows for name in ('fit', 'calibration'))
    cutoff = float(prepared.loc[prepared.timestamp + MARKET.step <= CALIBRATION_START,
                                'cond_volatility'].quantile(2/3))
    models = fit_models(fit, calibration)

    # Se cargan los datos de evaluación una vez fijados los pesos, calibradores y régimen.
    months = pd.date_range(EVALUATION_START, PROTECTED_HOLDOUT_START, freq='MS')
    ends = [*months[1:], PROTECTED_HOLDOUT_START]
    windows = [(start.strftime('%Y-%m'), start, end) for start, end in zip(months, ends)]
    fresh = load_features(raw_dir, [start.strftime('%Y%m') for start in months], quant, end=PROTECTED_HOLDOUT_START)
    blocks, _ = admission_blocks(fresh, founder, windows, anchor=anchor)
    evaluation = {name: evaluate_block(block, models, founder.account, cutoff) for name, block in blocks.items()}
    for name, start, end in windows:
        evaluation[name].update(start=start.isoformat(), end=end.isoformat(),
                                retrospective=start.month <= 3)
    march = blocks['2026-03']
    result = dict(
        market=asdict(MARKET), account=asdict(founder.account), stop_applied=False,
        protected_holdout_start=PROTECTED_HOLDOUT_START.isoformat(), protected_holdout_evaluated=False,
        entry_checkpoint=founder.provenance, anchor_checkpoint=anchor[0].provenance,
        logistic_features=MODEL_FEATURES, regularization_c=1.,
        fit=dict(start=FIT_START.isoformat(), end=CALIBRATION_START.isoformat(), admissions=len(fit)),
        calibration=dict(start=CALIBRATION_START.isoformat(), end=EVALUATION_START.isoformat(),
                         admissions=len(calibration)),
        high_volatility_cutoff=cutoff, high_volatility_quantile=2/3,
        bootstrap=dict(min_utc_days=MIN_DAYS_FOR_INTERVAL, samples=BOOTSTRAP_SAMPLES, seed=SEED),
        march_matches_reference=(march.eligible_bars, len(march.rows), march.mask_sha256)
        == (6258, 2859, FROZEN_MARCH_ADMISSIONS_SHA256),
        evaluation=evaluation,
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/histdata_scalping'))
    parser.add_argument('--output', type=Path, default=Path('reports/direction-benchmark/benchmark.json'))
    args = parser.parse_args()
    result = run_benchmark(args.raw_dir, args.output)
    print('DIRECTION_BENCHMARK ' + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
