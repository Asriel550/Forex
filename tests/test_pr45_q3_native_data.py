"""Synthetic acquisition + H1->Q3 GARCH continuity tests; never select a model."""
import numpy as np
import pandas as pd
import pytest

from scripts import build_pr45_q3_native as native
from scripts import run_pr44_q3_single_pass_audit as audit


def test_exactly_four_q3_parquet_names_and_input_windows():
    assert native.SYMBOLS == ("EURUSD","GBPUSD","AUDUSD","USDCAD")
    assert native.Q3_MONTHS == ("202607","202608","202609")
    assert native.ALLOWED["EURUSD"]==tuple(f"2026{m:02d}" for m in range(1,10))
    for name in native.SYMBOLS[1:]:
        assert native.ALLOWED[name]==("202606","202607","202608","202609")
    assert native.Q3_END.isoformat()=="2026-10-01T00:00:00+00:00"
    assert native.Q3_START.isoformat()=="2026-07-01T00:00:00+00:00"
    assert set(native.TYPES)==set(audit.FIELD_DTYPES)


def test_reject_october_download_before_network(tmp_path):
    import requests
    with pytest.raises(ValueError, match="Mes no autorizado"):
        native.get_month("GBPUSD","202610",tmp_path,requests.Session())


def test_reject_modified_unapproved_model_or_h1_bar_history():
    assert len(native.EXPECTED_H1_SHA)==3
    assert all(len(x)==64 for x in native.EXPECTED_H1_SHA.values())
    assert audit.SOURCE_SHA["AUDUSD"]=="8d1f34d5bf38ce1c4beeb5079d374a7af1a9d07a8161d07fc010d56b251b0eee"


def test_q2_seed_continuity_of_ratio_without_future_volatility():
    n=100
    data=dict(timeframe=np.ones((n,8)),volatility=np.full(n,.004))
    genes=np.zeros(22)
    genes[0]=1.
    genes[9]=.25
    q2=np.full(48,.003)
    score,mask,diag=audit.local_asset_head(data,genes,q2_garch_tail=q2)
    with_future=dict(data,volatility=data["volatility"].copy())
    with_future["volatility"][80:]=.01
    score2,mask2,_=audit.local_asset_head(with_future,genes,q2_garch_tail=q2)
    np.testing.assert_array_equal(score[:80],score2[:80])
    np.testing.assert_array_equal(mask[:80],mask2[:80])
    assert set(diag)=={"garch_low","garch_high","margin"}
    with pytest.raises(ValueError,match="Q2"):
        audit.local_asset_head(data,genes,q2_garch_tail=np.zeros(47))


def test_q3_m15_full_period_validation_is_3_months_not_6():
    opens=pd.date_range("2026-07-01T00:00Z","2026-09-30T23:45Z",
                        freq="15min",tz="UTC")
    bars=pd.DataFrame(dict(timestamp=opens,open=1.0,high=1.01,
                           low=.99,close=1.0,volume=1.0))
    native._check_frame(bars,pd.Timedelta(minutes=15),
                        native.Q3_START,native.Q3_END)
    with pytest.raises(ValueError):
        native._check_frame(bars.assign(timestamp=opens+pd.Timedelta(days=93)),
                            pd.Timedelta(minutes=15),native.Q3_START,native.Q3_END)
