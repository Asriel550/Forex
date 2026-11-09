"""Regresiones PR38 del gate causal (Q1/Q2 IS, sin señales M1)."""
import numpy as np
import pandas as pd
import pytest

from src.conviction_regime_gate import (
    MODES, PERCENTILES, causal_volatility_percentiles, gate_reasons,
)
from scripts.run_m5_temporal_evolution import causal_reasons


def test_gate_modes_and_thresholds_are_controlled():
    assert MODES==("p85","p80","p75","volatility_dynamic")
    assert PERCENTILES=={"p85":.85,"p80":.80,"p75":.75}


def test_local_volatility_reference_is_shifted_before_current_m5():
    first=np.full(80,.01)
    after=causal_volatility_percentiles(first)
    changed=first.copy();changed[55]=.05
    updated=causal_volatility_percentiles(changed)
    np.testing.assert_array_equal(after[:55],updated[:55])
    assert updated[55]==.85
    assert after[55]==.80
    assert updated[56] == .80  # future results are irrelevant


def test_causal_p85_is_identical_to_original_gate():
    times=pd.date_range("2026-03-01T00:05Z",periods=850,freq="5min")
    rows=pd.DataFrame({"outcome":np.tile([-1,1],425)},index=times)
    scores=np.sin(np.arange(850)/6.)
    sigma=np.full(850,.0001)
    np.testing.assert_array_equal(
        gate_reasons(rows,scores,sigma,"p85"),causal_reasons(rows,scores,.85)
    )
    for mode in MODES:
        reasons=gate_reasons(rows,scores,sigma,mode)
        assert len(reasons)==len(rows)
        assert np.all(reasons[:500]=="conviction_warmup")
        assert np.isin(reasons,["accepted","conviction","conviction_warmup"]).all()


def test_volatility_dynamic_cannot_read_future_scores_or_volatility():
    times=pd.date_range("2026-03-01T00:05Z",periods=900,freq="5min")
    rows=pd.DataFrame({"outcome":np.tile([-1,1],450)},index=times)
    scores=np.cos(np.arange(900)/5.)
    sigma=np.full(900,.0001)
    a=gate_reasons(rows,scores,sigma,"volatility_dynamic")
    updated=scores.copy();updated[880:]*=-1.0
    vol=sigma.copy();vol[880:]=.002
    b=gate_reasons(rows,updated,vol,"volatility_dynamic")
    np.testing.assert_array_equal(a[:880],b[:880])
    with pytest.raises(ValueError):
        gate_reasons(rows,scores,sigma,"p60")
