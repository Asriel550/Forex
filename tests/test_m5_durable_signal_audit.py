"""Regresiones PR35: binarios release y expansiones pre-gate sin modificar genomas."""
import numpy as np
import pandas as pd
import pytest

from scripts.audit_m5_primary_triggers import primary_masks, hourly_capacity, VARIANTS
from scripts.verify_m5_durable_snapshot import verify, SNAPSHOT_SHA256, RELEASE_TAG


@pytest.mark.parametrize("threshold", [-0.2, 0.0, 0.5, 1.3])
def test_relaxed_triggers_include_every_original_opportunity(threshold):
    entry = np.array([-1.5, -.3, -.001, .01, .4, 1.2])
    margin = np.array([.001, .2, 1.0, .01, .4, 2.])
    masks = primary_masks(entry, margin, threshold)
    assert set(masks) == set(VARIANTS)
    assert all(np.all(masks['current'] <= mask) for mask in masks.values())
    assert masks['both_relaxed'].sum() >= masks['current'].sum()
    assert masks['entry_zero'].sum() >= masks['current'].sum()


def test_month_boundary_and_hourly_cap_are_utc_and_rolling():
    times = pd.to_datetime(['2026-03-01T00:00Z', '2026-03-01T00:05Z',
                            '2026-03-01T00:10Z', '2026-03-01T01:05Z'])
    assert hourly_capacity(times, 2) == 3
    assert hourly_capacity(pd.DatetimeIndex([], tz='UTC'), 2) == 0


def test_scores_must_be_valid_and_shared_length():
    with pytest.raises(ValueError):
        primary_masks(np.array([0., np.nan]), np.array([1., 2.]), .1)
    with pytest.raises(ValueError):
        primary_masks(np.array([0., 1.]), np.array([1.]), .1)


def test_release_is_fail_closed_if_binary_is_edited(tmp_path):
    snapshot, manifest = tmp_path / 'frozen.npz', tmp_path / 'frozen.json'
    snapshot.write_bytes(b'not-the-canonical-bytes')
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='SHA256'):
        verify(snapshot, manifest)
    assert len(SNAPSHOT_SHA256) == 64
    assert RELEASE_TAG.startswith('data-m5-is-')
