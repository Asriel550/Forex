"""PR36: solo señales M5/M15 cerradas, sin entrada directa M1."""
import numpy as np
import pandas as pd
import pytest

from src.upstream_filters import (
    SIGNAL_NAMES, GENE_COUNT, timeframe_features, dynamic_upstream,
    legacy_upstream,
)
from scripts.build_m5_full_is_universe import assert_universe


def candles(freq, count, *, start="2026-03-01T12:00:00Z"):
    step=pd.Timedelta(freq)
    t=pd.date_range(start,periods=count,freq=freq)
    o=1.10+np.arange(count)*.00007
    c=o+np.where(np.arange(count)%2==0,.00004,-.000025)
    return pd.DataFrame(dict(
        timestamp=t,open=o,high=np.maximum(o,c)+.00006,
        low=np.minimum(o,c)-.00006,close=c,volume=np.ones(count),
    ))


def test_m15_is_used_only_after_full_15_minute_close():
    m5=candles("5min",8)
    m15=candles("15min",3)
    prior=timeframe_features(m5,m15)
    changed=m15.copy()
    changed.loc[1,"close"]+=.00009
    changed.loc[1,"high"]=max(changed.loc[1,"high"],changed.loc[1,"close"])
    after=timeframe_features(m5,changed)
    # La vela M15 abierta a las 12:15 no cierra hasta las 12:30.
    np.testing.assert_array_equal(prior.iloc[:5].to_numpy(),after.iloc[:5].to_numpy())
    assert not np.array_equal(prior.iloc[5].to_numpy(),after.iloc[5].to_numpy())
    # A las 12:05/12:10 aún no existe la primera M15 cerrada.
    np.testing.assert_array_equal(prior.iloc[:2,4:].to_numpy(),np.zeros((2,4)))
    assert list(prior.columns)==list(SIGNAL_NAMES)
    assert all(n.startswith(("m5_","m15_")) for n in SIGNAL_NAMES)


def test_m5_future_candle_cannot_change_previous_features():
    m5=candles("5min",8)
    m15=candles("15min",3)
    original=timeframe_features(m5,m15)
    future=m5.copy()
    future.loc[7,"close"]+=.00008
    future.loc[7,"high"]=max(future.loc[7,"high"],future.loc[7,"close"])
    after=timeframe_features(future,m15)
    np.testing.assert_array_equal(original.iloc[:7].to_numpy(),after.iloc[:7].to_numpy())


def test_m5_gap_does_not_reuse_stale_previous_body():
    m5=candles("5min",8).drop(index=2).reset_index(drop=True)
    m15=candles("15min",3)
    features=timeframe_features(m5,m15)
    assert features.iloc[2]["m5_previous_body"]==0.


def test_upstream_neutral_is_exact_legacy():
    entries=np.array([-1.,-.1,0.,.05,1.,2.])
    margins=np.array([2.,5.,5.2207,5.220701,7.,10.])
    signals=np.zeros((len(entries),len(SIGNAL_NAMES)))
    sigma=np.full(len(entries),.0001)
    threshold=5.2207
    passed,cutoff=dynamic_upstream(entries,margins,signals,sigma,np.zeros(GENE_COUNT),threshold)
    np.testing.assert_array_equal(passed,legacy_upstream(entries,margins,threshold))
    np.testing.assert_array_equal(cutoff,np.full(len(entries),threshold))


def test_m5_m15_features_modulate_entry_and_margin():
    n=100
    entries=np.full(n,-.1)
    margins=np.full(n,5.)
    features=np.zeros((n,len(SIGNAL_NAMES)))
    features[40:,0]=1.
    sigma=np.r_[np.full(50,.01),np.full(50,.04)]
    genes=np.zeros(GENE_COUNT);genes[0]=1.2;genes[-2]=.3;genes[-1]=.6
    selected,threshold=dynamic_upstream(entries,margins,features,sigma,genes,5.2207)
    assert not selected[:40].any()
    assert selected[40:].any()
    assert .05*5.2207 <= threshold.min() <= threshold.max() <= 5.2207
    assert threshold[70]<threshold[10]
    with pytest.raises(ValueError):
        dynamic_upstream(entries,margins,features,sigma,np.ones(GENE_COUNT-1),5.22)


def test_original_proposals_are_preserved_exactly_and_m1_field_is_absent():
    n=30120
    base=12
    t=pd.date_range('2026-01-01T00:00Z',periods=n,freq='5min').asi8.astype('<i8')
    archive=dict(row_times_ns=t,outcome=np.ones(n,dtype='i1'),
        direction_inputs=np.zeros((n,32)),timeframe=np.zeros((n,len(SIGNAL_NAMES))),
        legacy_mask=np.arange(n)<base,entry=np.ones(n),margin=np.ones(n),
        volatility=np.ones(n))
    golden=dict(row_times_ns=t[:base],outcome=np.ones(base,dtype='i1'),
                inputs=np.zeros((base,32)),cond_volatility=np.ones(base))
    assert_universe(golden,archive)
    modified=dict(archive,legacy_mask=np.arange(n)<base+1)
    with pytest.raises(ValueError,match='EXACTAMENTE'):
        assert_universe(golden,modified)
