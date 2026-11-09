"""PR41 independent asset training controls; synthetic arrays are NEVER real PF."""
import numpy as np
import pandas as pd
import pytest

from scripts.run_pr41_native_asset_evolution import (
    SEEDS,GENERATIONS,GENE_COUNT_LOCAL,POPULATION,SYMBOLS,
    asset_head,strict_asset_pass,genome_sha,
)


def fake_data(n=500):
    tf=np.zeros((n,8),dtype=float)
    tf[:,0]=np.linspace(-.8,.8,n)
    tf[:,4]=np.sin(np.arange(n)/17.)
    return dict(timeframe=tf,volatility=np.full(n,.001,dtype=float))


def example_report(pf=(1.3,1.31),dd=12.,stop=False):
    return dict(
        quarter_pf=list(pf),
        quarter_pnl_usd=[11.,22.],
        quarter_dd=[10.,11.],
        quarter_stop=[False,False],
        semester_dd=dd,semester_pnl_usd=33.,
        semester_stop=stop,semester_rate=.20,
    )


def test_exact_20_generations_128_candidates_four_pairs_three_seeds():
    assert SEEDS==(42,43,44)
    assert SYMBOLS==("EURUSD","GBPUSD","AUDUSD","USDCAD")
    assert GENERATIONS==20 and POPULATION==128
    assert GENERATIONS*POPULATION*len(SYMBOLS)*len(SEEDS)==30720


def test_zero_genome_is_local_not_copied_from_eurusd():
    zeros=np.zeros(GENE_COUNT_LOCAL)
    s,m,b=asset_head(fake_data(),zeros)
    assert s.shape==(500,) and not s.any()
    assert not m.any()
    assert b["band_lower"]==pytest.approx(.85)
    assert b["band_upper"]==pytest.approx(2.2)
    assert len(genome_sha(zeros))==64


def test_direction_and_entry_weights_evolve_independently():
    original=fake_data()
    g=np.zeros(GENE_COUNT_LOCAL)
    g[9]=1.2;g[0]=2.
    a,mask,details=asset_head(original,g)
    assert np.ptp(a)>1.
    assert mask.any()
    g2=g.copy();g2[19]=2.;g2[20]=-2.
    b,mask2,bounds=asset_head(original,g2)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(mask,mask2)
    assert bounds["band_lower"]>details["band_lower"]
    assert bounds["band_upper"]<details["band_upper"]


def test_current_m5_volatility_cannot_mutate_past_admissions():
    data=fake_data()
    g=np.zeros(GENE_COUNT_LOCAL)
    g[9]=2.;g[0]=2.
    _,base,_=asset_head(data,g)
    altered=dict(data,volatility=data["volatility"].copy())
    altered["volatility"][430:]=.03
    _,changed,_=asset_head(altered,g)
    np.testing.assert_array_equal(base[:430],changed[:430])


def test_standalone_gate_strict_pf_dd_and_stop():
    assert strict_asset_pass(example_report())
    assert not strict_asset_pass(example_report(pf=(1.20,1.3)))
    assert not strict_asset_pass(example_report(dd=15.))
    assert not strict_asset_pass(example_report(stop=True))
    assert not strict_asset_pass(example_report(pf=(None,1.4)))


def test_evolution_history_is_flat_and_never_nested_under_record():
    from src.evolution_multiscale import evolve_population
    from scripts.run_m5_dynamic_upstream import key
    def fitness(pop):
        n=len(pop)
        values=np.full(n,-1.0)
        return dict(
            qualifies=np.zeros(n,dtype=bool),
            score=values,
            worst_quarter_return=values,
            quarter_return_std=np.ones(n),
            min_gate_only_frequency=np.zeros(n),
            return_per_admission=values,
            quarter_pf=np.zeros((n,2)),
            semester_rate=np.zeros(n),
            semester_dd=np.zeros(n),
            regime_lower=np.full(n,.85),
            regime_upper=np.full(n,2.2),
        )
    run=evolve_population(fitness,np.zeros(GENE_COUNT_LOCAL),
        "pr41_test_history",42,first_generation=1,last_generation=2,ranking=key)
    assert run["evaluations"]==256
    assert [r["generation"] for r in run["generations"]]==[1,2]
    assert all("record" not in x and "quarter_pf" in x for x in run["generations"])
