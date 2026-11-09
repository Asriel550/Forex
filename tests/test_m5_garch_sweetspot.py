"""PR39: barrido GARCH, gates causales y fitness PF×frecuencia.

Garantiza identidad del p85 original y que la expansión de banda no
puede reducir la máscara. Las pruebas no necesitan datos Q3 ni ZIP M1.
"""
import numpy as np
import pandas as pd
import pytest

from src.garch_sweetspot import (
    BAND_RELAXATIONS,GATE_MODES,MAX_SEMESTER_DD,TARGET_MIN_HOURLY,
    normalized_garch_ratio,garch_band,local_regime_percentiles,
    causal_custom_gate,accepted,pf_frequency_objective,
)
from src.conviction_regime_gate import gate_reasons
from src.upstream_filters import GENE_COUNT
from scripts.run_m5_garch_sweetspot import SEEDS,GENERATIONS,POPULATION


def fake_report(pfs=(1.25,1.30),rate=.65,dd=10.,stop=False):
    return dict(
        quarter_pf=list(pfs),quarter_pnl_usd=[100.,120.],
        quarter_stop=[False,False],semester_rate=rate,
        semester_dd=dd,semester_stop=stop,
    )


def test_experiment_dimensions_and_no_oos():
    assert SEEDS==(42,43,44)
    assert GENERATIONS==12 and POPULATION==128
    assert BAND_RELAXATIONS==(0.,.075,.15,.225,.30)
    assert GATE_MODES==("p85","p80","p75","regime_dynamic")
    assert GENERATIONS*POPULATION*len(SEEDS)==4608


def test_band_15pct_is_monotonic_and_uses_original_head_thresholds():
    n=350
    sigma=.005 + .002*np.sin(np.arange(n)/7.)
    genes=np.zeros(GENE_COUNT+13)
    base,bounds,ratio=garch_band(genes,sigma,0.)
    wider,b15,ratio15=garch_band(genes,sigma,.15)
    np.testing.assert_array_equal(ratio,ratio15)
    assert np.all(~base|wider)
    assert b15["applied_low"]==pytest.approx(bounds["applied_low"]*.85)
    assert b15["applied_high"]==pytest.approx(bounds["applied_high"]*1.15)
    assert b15["selected_rows"]>=bounds["selected_rows"]
    with pytest.raises(ValueError):
        garch_band(genes,sigma,-.1)


def test_regime_volatility_never_depends_on_future_m5():
    original=.005+.0009*np.sin(np.arange(160)/5)
    later=original.copy();later[95:]=.1
    a=normalized_garch_ratio(original)
    b=normalized_garch_ratio(later)
    np.testing.assert_array_equal(a[:95],b[:95])
    assert a[95]!=b[95]


def test_local_gate_p75_core_p80_shoulder_p85_edge():
    r=np.array([.05,.25,.50,.75,.95])*2+1
    # band: low 1 high 3 ; relative = [.05,.25,.50,.75,.95]
    ps=local_regime_percentiles(r,1.,3.)
    np.testing.assert_array_equal(ps,[.85,.75,.75,.75,.85])
    with pytest.raises(ValueError):
        local_regime_percentiles(r,3,1)


def test_gate_p85_reproduces_previous_500_window():
    times=pd.date_range("2026-03-01T00:05:00Z",periods=850,freq="5min")
    frame=pd.DataFrame({"outcome":np.tile([-1,1],425)},index=times)
    scores=np.sin(np.arange(850)/6.)
    sigma=np.full(850,.001)
    np.testing.assert_array_equal(
        causal_custom_gate(frame,scores,np.full(850,.85)),
        gate_reasons(frame,scores,sigma,"p85")
    )
    first=causal_custom_gate(frame,scores,np.full(850,.85))
    changed=scores.copy();changed[800:]=10.
    second=causal_custom_gate(frame,changed,np.full(850,.85))
    np.testing.assert_array_equal(first[:800],second[:800])


def test_acceptance_is_hard_and_not_a_fitness_proxy():
    base=fake_report()
    assert accepted(base)
    assert not accepted(fake_report(pfs=(1.2,1.3)))
    assert not accepted(fake_report(rate=.49))
    assert not accepted(fake_report(dd=12.))
    assert not accepted(fake_report(stop=True))


def test_multiplicative_fitness_rewards_frequency_and_pf_symmetrically():
    a=fake_report(rate=.50,pfs=(1.0,1.0),dd=10.)
    b=fake_report(rate=.60,pfs=(1.0,1.0),dd=10.)
    c=fake_report(rate=.50,pfs=(1.2,1.2),dd=10.)
    score_a,product_a=pf_frequency_objective(a)
    score_b,product_b=pf_frequency_objective(b)
    score_c,product_c=pf_frequency_objective(c)
    assert product_a==pytest.approx(1.)
    assert product_b==pytest.approx(1.2)
    assert product_c==pytest.approx(1.2)
    assert score_b>score_a and score_c>score_a
    assert pf_frequency_objective(fake_report(dd=20.))[0] < score_a
