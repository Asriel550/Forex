"""PR #34: reproducibilidad M1/M5/MC y fitness sin colapso de densidad."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.m5_is_repro_audit import compare_independent_builds, load_verified_snapshot
from src.evolution_engine import assess_temporal_multiregime_fitness, temporal_fitness_weights
from src.reproducibility import market_frame_fingerprint, canonical_open_diff, array_sha256


def block(pnl=-10., gate_only_pnl=-10., density=.10, gate_density=.50,
          cap=1.1, dd=20., stop=True):
    return dict(
        pnl_usd=pnl, gate_only_pnl_usd=gate_only_pnl,
        max_drawdown_pct=dd, contracts_per_observed_hour=density,
        gate_only_contracts_per_observed_hour=gate_density,
        capacity_contracts_per_observed_hour=cap,
        profit_factor=.9, profit_factor_status='finite',
        observed_hours=100., admissions=1000,
        account_stop={'reason': 'drawdown'} if stop else None,
    )


def test_market_hash_changes_on_one_candle_without_changing_count():
    times = pd.date_range('2026-03-01T00:00Z', periods=5, freq='min')
    frame = pd.DataFrame(dict(
        timestamp=times, open=np.full(5, 1.1), high=np.full(5, 1.2),
        low=np.full(5, 1.), close=np.full(5, 1.1), volume=np.ones(5),
    ))
    a = market_frame_fingerprint(frame, columns=('open', 'high', 'low', 'close', 'volume'))
    frame.loc[2, 'close'] += .00001
    b = market_frame_fingerprint(frame, columns=('open', 'high', 'low', 'close', 'volume'))
    assert a['rows'] == b['rows'] and a['timestamps_sha256'] == b['timestamps_sha256']
    assert a['values_sha256'] != b['values_sha256']
    assert array_sha256([1., 2.]) != array_sha256([2., 1.])


def test_open_diff_reports_exact_divergent_timestamp():
    expected = pd.date_range('2026-03-01T00:00Z', periods=3, freq='5min')
    actual = expected.delete(1).append(pd.DatetimeIndex(['2026-03-01T00:20Z']))
    delta = canonical_open_diff(actual, expected)
    assert delta['missing_count'] == 1 and delta['unexpected_count'] == 1
    assert delta['missing_open_utc'][0] == expected[1].isoformat()


def test_compare_independent_builds_rejects_garch_only_diff(tmp_path):
    one = dict(
        schema='frozen', sources=['202601'], period_start='2026-01-01',
        period_end_exclusive='2026-07-01', hashes={'parent': 'unchanged'},
        quarter_observed_hours={'Q1': 3}, quarter_capacity_contracts_per_observed_hour={'Q1': 1.},
        fingerprints={'m1': 'same', 'm5': 'same', 'garch_and_mc': 'aaa'},
        snapshot_sha256='same',
    )
    a, b = tmp_path / 'one.json', tmp_path / 'two.json'
    a.write_text(json.dumps(one))
    b.write_text(json.dumps(one))
    assert compare_independent_builds(a, b) == 'same'
    other = dict(one, fingerprints=dict(one['fingerprints'], garch_and_mc='bbb'))
    b.write_text(json.dumps(other))
    with pytest.raises(ValueError, match='garch_and_mc'):
        compare_independent_builds(a, b)


def test_snapshot_rejects_tampered_sha_before_accessing_market(tmp_path):
    fake = tmp_path / 'data.npz'
    with fake.open('wb') as f:
        np.savez_compressed(f, row_times_ns=np.array([1], dtype=np.int64))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'schema': 'pr34_m5_is_frozen_v1', 'snapshot_sha256': 'not_the_sha',
        'sources': ['202601', '202602', '202603', '202604', '202605', '202606'],
    }))
    with pytest.raises(ValueError, match='sellado'):
        load_verified_snapshot(fake, manifest)


def test_fitness_density_uses_pre_stop_execution_not_a_stopped_account():
    low = [block(gate_density=.30), block(gate_density=.30)]
    high = [block(gate_density=.80), block(gate_density=.80)]
    a = assess_temporal_multiregime_fitness(low, stake=10., direction_loss=1.)
    b = assess_temporal_multiregime_fitness(high, stake=10., direction_loss=1.)
    assert b['score'] > a['score']
    assert b['quarter_frequencies'] == a['quarter_frequencies'] == [.1, .1]
    assert b['quarter_gate_only_frequencies'] == [.8, .8]
    assert b['qualifies'] is False
    assert temporal_fitness_weights()['return_scale'] == 100.


def test_fitness_respects_q2_capacity_below_global_target():
    low = [block(gate_density=.9, cap=.95), block(gate_density=.9, cap=.95)]
    full = [block(gate_density=.95, cap=.95), block(gate_density=.95, cap=.95)]
    a = assess_temporal_multiregime_fitness(low, stake=10., direction_loss=.5)
    b = assess_temporal_multiregime_fitness(full, stake=10., direction_loss=.5)
    assert b['frequency_shortfall'] == pytest.approx(0.)
    assert b['score'] > a['score']
    with pytest.raises(ValueError, match='Métricas'):
        assess_temporal_multiregime_fitness(
            [block(gate_density=1.5, cap=.95), block()], stake=10., direction_loss=.5
        )


def test_drawdown_still_penalizes_and_qualification_is_strict():
    safe = [block(dd=12., stop=False), block(dd=12., stop=False)]
    stopped = [block(dd=20., stop=True), block(dd=20., stop=True)]
    a = assess_temporal_multiregime_fitness(safe, stake=10., direction_loss=.5)
    b = assess_temporal_multiregime_fitness(stopped, stake=10., direction_loss=.5)
    assert a['score'] > b['score']
    assert not b['qualifies']


def test_calibrated_fitness_all_fields_can_enter_vectorized_population():
    """El pipeline heredado llama .tolist() en cada campo del candidato."""
    stats = assess_temporal_multiregime_fitness(
        [block(), block()], stake=10., direction_loss=.8
    )
    assert not any(isinstance(value, dict) for value in stats.values())
    for value in stats.values():
        arr = np.asarray([value, value])
        assert arr[0].tolist() is not None
    assert stats['penalty_frequency'] >= 0.
    assert stats['penalty_max_dd'] > 0.
