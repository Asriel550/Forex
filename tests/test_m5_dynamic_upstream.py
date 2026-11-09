"""PR36: invariantes de causalidad M1/M5, entrada evolutiva y OOS sellado."""
import numpy as np
import pandas as pd
import pytest

from src.upstream_filters import (
    PROXY_NAMES, GENE_COUNT, microstructure_proxies,
    dynamic_upstream, legacy_upstream, MAX_MARGIN_RELAXATION,
)
from scripts.build_m5_full_is_universe import assert_universe


def tiny_m1():
    t = pd.date_range('2026-03-01T12:00Z', periods=15, freq='min')
    o = 1.10 + np.linspace(0, .00020, 15)
    c = o + .00001 * np.where(np.arange(15)%3==0, -1, 1)
    return pd.DataFrame(dict(timestamp=t,open=o,high=np.maximum(o,c)+.000005,
                            low=np.minimum(o,c)-.000005,close=c,volume=np.ones(15)))


def m5_aggregated(minutes):
    bucket = minutes.timestamp.dt.floor('5min')
    return minutes.groupby(bucket,sort=True).agg(
        timestamp=('timestamp','first'),open=('open','first'),
        high=('high','max'),low=('low','min'),
        close=('close','last'),volume=('volume','sum')
    ).reset_index(drop=True)


def test_microstructure_uses_only_five_closed_m1():
    minute = tiny_m1()
    candle = m5_aggregated(minute)
    prior = microstructure_proxies(minute.iloc[:10],candle.iloc[:2])
    later = minute.copy()
    later.loc[10:,"close"] *= 1.0002
    later.loc[10:,"high"] = np.maximum(later.loc[10:,"open"],later.loc[10:,"close"])+.000005
    after = microstructure_proxies(later,candle)
    np.testing.assert_array_equal(prior.to_numpy(),after.iloc[:2].to_numpy())
    assert list(prior.columns)==list(PROXY_NAMES)
    with pytest.raises(ValueError,match='cinco M1'):
        microstructure_proxies(minute.drop(index=2),candle)


def test_upstream_neutral_is_bytewise_legacy_for_all_bars():
    entries=np.array([-1.0,-.1,0.,.05,1.,2.])
    margins=np.array([2.,5.,5.2207,5.220701,7.,10.])
    features=np.zeros((len(entries),len(PROXY_NAMES)))
    sigma=np.full(len(entries),.0001)
    threshold=5.2207
    predicted,cutoff=dynamic_upstream(entries,margins,features,sigma,
                                      np.zeros(GENE_COUNT),threshold)
    np.testing.assert_array_equal(predicted,legacy_upstream(entries,margins,threshold))
    np.testing.assert_array_equal(cutoff,np.full(len(entries),threshold))


def test_dynamic_proxies_and_volatility_influence_entry_without_future_outcomes():
    n=100
    entries=np.full(n,-.1)
    margins=np.full(n,5.)
    features=np.zeros((n,len(PROXY_NAMES)))
    features[40:,0]=1.
    sigma=np.r_[np.full(50,.01),np.full(50,.04)]
    genes=np.zeros(GENE_COUNT);genes[0]=1.2;genes[-2]=.3;genes[-1]=.6
    selected,threshold=dynamic_upstream(entries,margins,features,sigma,genes,5.2207)
    assert not selected[:40].any()
    assert selected[40:].any()
    assert 0.05*5.2207 <= threshold.min() <= threshold.max() <= 5.2207
    assert threshold[70] < threshold[10]
    with pytest.raises(ValueError):
        dynamic_upstream(entries,margins,features,sigma,np.ones(GENE_COUNT-1),5.22)


def test_original_proposals_cannot_be_approximated_or_dropped():
    n=30120
    base=12
    t=pd.date_range('2026-01-01T00:00Z',periods=n,freq='5min').asi8.astype('<i8')
    archive=dict(row_times_ns=t,outcome=np.ones(n,dtype='i1'),
        direction_inputs=np.zeros((n,32)),micro=np.zeros((n,len(PROXY_NAMES))),
        legacy_mask=np.arange(n)<base,entry=np.ones(n),margin=np.ones(n),
        volatility=np.ones(n))
    golden=dict(row_times_ns=t[:base],outcome=np.ones(base,dtype='i1'),
                inputs=np.zeros((base,32)),cond_volatility=np.ones(base))
    assert_universe(golden,archive)
    altered=dict(archive,legacy_mask=(np.arange(n)<base+1))
    with pytest.raises(ValueError,match='EXACTAMENTE'):
        assert_universe(golden,altered)
