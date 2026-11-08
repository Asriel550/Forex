"""Invariantes causales y de separación OOS de la PR #32."""
import numpy as np
import pandas as pd
import pytest

from scripts.run_m5_adaptive_gate import GEN354_SHA256
from scripts.run_m5_multiscale import genome_hash
from scripts.run_m5_semester_replay import (
    END, OOS_END, PERIODS, START, assert_isolated, frozen_models, tested_metrics as measure_targets,
    verify_source_periods,
)


def test_semestral_calendar_reserves_all_of_q3():
    assert_isolated(PERIODS)
    assert START == pd.Timestamp('2026-01-01T00:00:00Z')
    assert END == pd.Timestamp('2026-07-01T00:00:00Z')
    assert OOS_END == pd.Timestamp('2026-10-01T00:00:00Z')
    with pytest.raises(ValueError, match='OOS'):
        assert_isolated((*PERIODS, '202607'))
    with pytest.raises(ValueError, match='OOS'):
        assert_isolated(PERIODS, end=pd.Timestamp('2026-10-01T00:00:00Z'))


def test_semester_source_manifest_is_pinned_and_excludes_oos():
    manifest = dict(sources=[
        dict(period=p, sha256='a' * 64, zip_file=f'{p}.zip') for p in PERIODS
    ])
    verify_source_periods(manifest)
    with pytest.raises(ValueError, match='SHA256'):
        verify_source_periods(dict(sources=manifest['sources'][:-1]))
    with pytest.raises(ValueError, match='duplicados'):
        verify_source_periods(dict(sources=manifest['sources'] + manifest['sources'][:1]))


def test_selected_champions_cannot_change_after_seal():
    parent = [0., 1.]
    evolution = dict(
        parent_sha256=GEN354_SHA256, protected_holdout_evaluated=False,
        winners={'p80': 43, 'p75': 43},
        replicas=[
            dict(percentile=p, seed=43, genome=[p, 1.],
                 genome_sha256=genome_hash([p, 1.]),
                 champion=dict(generation=375))
            for p in (.80, .75)
        ],
    )
    models = frozen_models(evolution, parent)
    assert models['gen354']['generation'] == 354
    assert models['gen375_p80']['generation'] == 375
    evolution['replicas'][0]['genome_sha256'] = 'tampered'
    with pytest.raises(ValueError, match='incompatible'):
        frozen_models(evolution, parent)


def test_target_checks_are_strict_for_pf_and_drawdown():
    outcome = dict(operational=dict(profit_factor=1.20,
                                   contracts_per_observed_hour=1.2,
                                   max_drawdown_pct=15.0))
    stats = measure_targets(outcome)
    assert not stats['meets_pf'] and stats['meets_density'] and not stats['meets_drawdown']
    outcome['operational'].update(profit_factor=1.201, max_drawdown_pct=14.99)
    stats = measure_targets(outcome)
    assert all(stats[key] for key in ('meets_pf', 'meets_density', 'meets_drawdown'))
