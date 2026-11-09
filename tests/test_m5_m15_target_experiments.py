"""Pruebas de etiquetas ATR/R:R causales y de la loss asimétrica PR38."""
import numpy as np
import pandas as pd
import pytest

from scripts.build_m5_m15_alpha_targets import (
    atr_on_closed_m15, targets_from_m5, FORWARD_STEPS, M5_NS,
)
from scripts.run_m5_m15_target_experiments import (
    DIR_GENE_COUNT, TOTAL_GENE_COUNT, PHASE2, PHASE3,
    asymmetric_target_loss, score_and_mask, accepted_by_pr38,
)
from src.upstream_filters import GENE_COUNT


def candles(freq, n, start="2026-02-02T00:00:00Z", delta=.00003):
    t=pd.date_range(start,periods=n,freq=freq)
    o=1.10+np.arange(n)*delta
    c=o+delta/2
    return pd.DataFrame(dict(
        timestamp=t,open=o,high=np.maximum(o,c)+delta,
        low=np.minimum(o,c)-delta,close=c,volume=np.ones(n),
    ))


def test_atr_uses_only_completed_m15_candle():
    m15=candles("15min",120)
    t=np.array([
        pd.Timestamp("2026-02-02T05:10Z").value,
        pd.Timestamp("2026-02-02T05:15Z").value,
        pd.Timestamp("2026-02-02T05:20Z").value,
    ],dtype="<i8")
    original=atr_on_closed_m15(m15,t)
    new=m15.copy()
    new.loc[21,"high"]+=.01  # M15 opened 05:15, closes 05:30.
    after=atr_on_closed_m15(new,t)
    np.testing.assert_array_equal(original,after)


def test_targets_never_need_m1_features():
    m5=candles("5min",200)
    m15=candles("15min",75)
    times=(pd.DatetimeIndex(m5.timestamp).as_unit("ns").asi8[75:140]+M5_NS)
    values=np.zeros((len(times),8),dtype=float)
    labels=targets_from_m5(m5,m15,times,values)
    assert set(labels)=={
        "atr_m15","atr_expansion_label","rr15_label","valid_label","low_momentum"
    }
    assert labels["valid_label"].sum()>0
    assert len(labels["atr_m15"])==len(times)
    assert np.isin(labels["rr15_label"],[-1,0,1]).all()
    # Changing a candle after the 30-minute evaluation window cannot alter the label.
    newer=m5.copy();newer.loc[199,"high"]+=.08
    other=targets_from_m5(newer,m15,times,values)
    np.testing.assert_array_equal(labels["rr15_label"],other["rr15_label"])


def test_asymmetric_low_momentum_false_positive_costs_more():
    labels=np.array([1,1,1,-1,-1,0,0],dtype=np.int8)
    scores=np.array([2,2,2,2,2,2,2],dtype=float)
    valid=np.ones(len(labels),bool)
    normal=np.zeros(len(labels),bool)
    low=np.ones(len(labels),bool)
    assert asymmetric_target_loss(scores,labels,valid,low)>asymmetric_target_loss(
        scores,labels,valid,normal
    )
    assert asymmetric_target_loss(-scores,labels,valid,normal)>0.
    with pytest.raises(ValueError):
        asymmetric_target_loss(scores,labels,np.ones(1,bool),low)


def test_direction_head_starts_at_frozen_direction_gen375():
    n=120
    problem=dict(
        scores=np.linspace(-2,2,n),
        side=np.where(np.linspace(-2,2,n)>0,1,-1),
        data=dict(timeframe=np.zeros((n,8)),volatility=np.ones(n)*.001),
    )
    genes=np.zeros(TOTAL_GENE_COUNT)
    for variant in PHASE2+PHASE3:
        score,mask=score_and_mask(problem,genes,variant)
        np.testing.assert_array_equal(score,problem["scores"])
        if variant=="garch_regime":
            assert mask.shape==(n,) and mask.dtype==bool
        else:
            assert mask is None
    assert DIR_GENE_COUNT==13
    assert TOTAL_GENE_COUNT==GENE_COUNT+13


def test_strict_semester_acceptance_requires_1_0_hour_and_both_pf():
    candidate=dict(
        quarter_pf=[1.25,1.35],quarter_pnl_usd=[100.,110.],
        quarter_dd=[11.,13.],semester_dd=14.9,semester_rate=1.00,
        quarter_stop=[False,False],semester_stop=False,
    )
    assert accepted_by_pr38(candidate)
    candidate["quarter_pf"][0]=1.20
    assert not accepted_by_pr38(candidate)
    candidate["quarter_pf"][0]=1.3
    candidate["semester_dd"]=15.
    assert not accepted_by_pr38(candidate)
