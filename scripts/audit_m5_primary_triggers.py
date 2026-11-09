"""Audita triggers primarios M5 ANTES del Gate de Convicción Pura.

Solo ablation de cobertura (no PnL/PF ni entrenamiento), con fundador
congelado y seis meses IS. Una recomputación GARCH distinta de la Release
queda marcada como NO COMPARABLE, nunca se adopta como nueva verdad.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_direction_ablation import load_minutes, m5_features, founder_observations
from scripts.run_m5_conviction_evolution import setup
from scripts.run_m5_semester_replay import END, MONTHS, PERIODS, SOURCE_MANIFEST, assert_isolated, verify_source_periods
from scripts.run_m5_adaptive_gate import RAW
from scripts.verify_m5_durable_snapshot import verify
from src.decision import RiskConfig
from src.environment import BinaryOptionsEnv, prepare_market_data
from src.evolution_engine import M5Simulation
from src.evolution_lineage import MARKET
from src.replay_engine import apply_hourly_cap
from src.reproducibility import market_frame_fingerprint


VARIANTS = (
    'current', 'margin_half', 'margin_zero',
    'entry_relaxed', 'entry_zero', 'both_relaxed',
)


def primary_masks(entry, margin, activation_threshold):
    """Mascaras estrictamente inclusivas respecto del filtro original.

    entry_relaxed elimina un cuarto de desviación de barrera del entry logit.
    Las variaciones son *diagnósticas*: ningún nuevo trade se promueve.
    """
    entry, margin = np.asarray(entry, dtype=float), np.asarray(margin, dtype=float)
    if (entry.shape != margin.shape or entry.ndim != 1 or not np.isfinite(entry).all()
            or not np.isfinite(margin).all()
            or not np.isfinite(activation_threshold)):
        raise ValueError('Los scores del trigger deben ser finitos')
    original = (entry > 0.) & (margin > 0.) & (margin > activation_threshold)
    relax_entry = min(0., -0.25 * float(entry.std()))
    relax_margin = min(activation_threshold, max(0., activation_threshold * .5))
    masks = dict(
        current=original,
        margin_half=(entry > 0.) & (margin > 0.) & (margin > relax_margin),
        margin_zero=(entry > 0.) & (margin > 0.),
        entry_relaxed=(entry > relax_entry) & (margin > 0.) & (margin > activation_threshold),
        entry_zero=(margin > 0.) & (margin > activation_threshold),
        both_relaxed=(margin > 0.),
    )
    if any(np.any(original & ~mask) for mask in masks.values()):
        raise ValueError('Los contrafactuales no pueden eliminar oportunidades originales')
    return masks


def hourly_capacity(times, cap):
    if len(times) == 0:
        return 0
    stamps = pd.DatetimeIndex(times).sort_values()
    decisions = apply_hourly_cap(stamps, np.full(len(stamps), 'accepted'), cap)
    return int(np.count_nonzero(decisions == 'accepted'))


def analyze(*, snapshot, manifest, output):
    golden = verify(snapshot, manifest)
    report = json.loads(Path(manifest).read_text(encoding='utf-8'))
    assert_isolated(PERIODS)
    sources = json.loads(SOURCE_MANIFEST.read_text(encoding='utf-8'))
    verify_source_periods(sources)
    reference, founder, quant, _, _, _, _, _, _ = setup()

    minutes = load_minutes(RAW, PERIODS, end=END)
    features = m5_features(minutes, quant, end=END)
    fingerprint = market_frame_fingerprint(
        features, columns=('cond_volatility', 'mc_prob_up', 'mc_prob_down', 'mc_prob_tie')
    )
    canonical_quant = (fingerprint == report['fingerprints']['garch_and_mc'])
    prepared = prepare_market_data(features, market=MARKET)
    context, normalized, norm = founder_observations(prepared, founder)
    del context
    period_variants, observed_hours = {}, {}
    periods = {}
    for name, start, end in MONTHS:
        frame = prepared.loc[
            (prepared.timestamp >= start) & (prepared.timestamp + MARKET.step <= end)
        ]
        env = BinaryOptionsEnv(
            frame, market=MARKET, prepared=True, account=founder.account, normalizer=norm,
            risk=RiskConfig(enabled=False, drawdown_limit=founder.drawdown_limit),
        )
        sim = M5Simulation(env, normalized.loc[env._times])
        entry, direction = sim.decision_scores(founder.genome)
        margin = np.abs(direction)
        masks = primary_masks(entry, margin, float(founder.genome[-1]))
        opens = env._times[sim.entries]
        times = opens + MARKET.step
        baseline = opens[masks['current']]
        opens_hash = sha256(baseline.as_unit('ns').asi8.tobytes()).hexdigest()
        frozen_month = report['fingerprints']['month_admissions'][name]
        period_variants[name] = {key: times[masks[key]] for key in VARIANTS}
        observed_hours[name] = float(sim.observed_hours)
        periods[name] = dict(
            eligible_bars=len(opens),
            raw_entry_positive=int(np.count_nonzero(entry > 0.)),
            raw_margin_positive=int(np.count_nonzero(margin > 0.)),
            passes_entry_and_nonzero=int(np.count_nonzero((entry > 0.) & (margin > 0.))),
            passes_baseline=len(baseline),
            baseline_mask_sha256=opens_hash,
            matches_frozen_admissions=(
                len(baseline) == frozen_month['proposals']
                and opens_hash == frozen_month['opens_sha256']
            ),
            observed_hours=float(sim.observed_hours),
            activation_threshold=float(founder.genome[-1]),
            entry_score_p10_p50_p90=np.quantile(entry, [.1, .5, .9]).tolist(),
            margin_p10_p50_p90=np.quantile(margin, [.1, .5, .9]).tolist(),
            variants={
                key: dict(
                    pre_gate_proposals=int(masks[key].sum()),
                    after_hourly_cap=hourly_capacity(times[masks[key]],
                                                     reference['gate']['hourly_cap']),
                ) for key in VARIANTS
            },
        )
    quarters = {}
    for label, indices in (('Q1', PERIODS[:3]), ('Q2', PERIODS[3:]), ('semester', PERIODS)):
        hours = sum(observed_hours[p] for p in indices)
        variants = {}
        for kind in VARIANTS:
            times = pd.to_datetime(np.concatenate([
                period_variants[p][kind].as_unit('ns').asi8 for p in indices
            ]), utc=True, unit='ns')
            count = hourly_capacity(times, reference['gate']['hourly_cap'])
            variants[kind] = dict(proposals=len(times), after_hourly_cap=count,
                                  max_possible_contracts_per_hour=count / hours)
        quarters[label] = dict(observed_hours=hours,
                               eligible_bars=sum(periods[p]['eligible_bars'] for p in indices),
                               variants=variants)
    comparable = canonical_quant and all(periods[p]['matches_frozen_admissions'] for p in PERIODS)
    audit = dict(
        protocol='PR35_primary_trigger_IS_v1',
        is_months=PERIODS, oos_accessed=False,
        reference_snapshot_sha256=golden['snapshot_sha256'],
        garch_fingerprint_reproduced=canonical_quant,
        all_primary_masks_reproduced=all(periods[p]['matches_frozen_admissions'] for p in PERIODS),
        safe_for_canonical_comparison=comparable,
        model='frozen original entry and activation heads; no changed genomic admission',
        method='counterfactual funnel only: no gate, no PF, no payoff, no real fills',
        trigger_relaxations=list(VARIANTS),
        hourly_cap=reference['gate']['hourly_cap'],
        periods=periods, quarters=quarters,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('PRIMARY_TRIGGER_AUDIT ' + json.dumps({
        'canonical': comparable,
        'q1': quarters['Q1']['variants'], 'q2': quarters['Q2']['variants'],
        'activation_threshold': float(founder.genome[-1]),
    }, allow_nan=False), flush=True)
    if not comparable:
        raise ValueError('Reconstrucción GARCH no canónica: solo diagnóstico, no validar variantes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(snapshot=args.snapshot, manifest=args.manifest, output=args.output)
