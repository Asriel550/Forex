"""PR40 GARCH source pipeline tests; synthetic inputs NEVER trading results."""
import numpy as np
import pandas as pd
import pytest

from scripts.build_pr40_garch_universe import m15_closed_context
from src.evolution_context import M15_TACTICAL
from src.m5_multiasset_inputs import FIELDS
from src.m5_multiasset_portfolio import END,START


def fake_m15():
    t=pd.date_range("2026-03-02T00:00:00Z",periods=100,freq="15min")
    o=1.10+np.arange(len(t))*.00002
    return pd.DataFrame(dict(
        timestamp=t,open=o,high=o+.0001,low=o-.00006,
        close=o+.000015,volume=np.zeros(len(t)),
    ))


def test_m15_features_only_reflect_last_completed_candle():
    frame=fake_m15()
    decisions=pd.date_range("2026-03-02T10:05Z",periods=20,freq="5min")
    first=m15_closed_context(frame,decisions)
    assert list(first)==list(M15_TACTICAL)
    assert first.index.equals(decisions)
    changed=frame.copy()
    # 11:30 candle not complete until 11:45; earlier decisions unchanged.
    changed.loc[46,"high"]+=.02
    second=m15_closed_context(changed,decisions)
    pd.testing.assert_frame_equal(
        first.loc[:pd.Timestamp("2026-03-02T11:30Z")],
        second.loc[:pd.Timestamp("2026-03-02T11:30Z")],
    )
    assert np.isfinite(first["m15_atr_fraction"].iloc[:5]).all()


def test_no_q3_in_symbol_universe_or_m1_inputs():
    assert START==pd.Timestamp("2026-01-01T00:00Z")
    assert END==pd.Timestamp("2026-07-01T00:00Z")
    assert FIELDS==(
        "row_times_ns","outcome","volatility","entry","margin",
        "timeframe","direction_inputs","legacy_mask"
    )
    assert not any(k.startswith("m1") or k=="micro" for k in FIELDS)
