"""PR44 one-pass OOS audit contract tests using SYNTHETIC values only."""
import numpy as np
import pandas as pd
import pytest

from scripts import run_pr44_q3_single_pass_audit as audit


def test_exact_24_pinned_models_and_q3_clock():
    assert len(audit.PINNED_SHA) == 4
    assert {key for sym in audit.PINNED_SHA for key in audit.PINNED_SHA[sym]} == set(audit.SEEDS)
    assert len({(sym, seed, group) for sym in audit.SYMBOLS
                for seed in audit.SEEDS for group in audit.GROUPS}) == 24
    assert audit.START.isoformat() == "2026-07-01T00:00:00+00:00"
    assert audit.END.isoformat() == "2026-10-01T00:00:00+00:00"
    assert audit.STEP == pd.Timedelta(minutes=5)
    for seeds in audit.PINNED_SHA.values():
        for pair in seeds.values():
            assert len(pair) == 2
            assert all(len(digest) == 64 for digest in pair)


def test_pf_h1_reconstruction_is_not_just_cherrypicked_q1():
    # Q1: gross gains 200, gross losses 100, PF 2.
    # Q2: gross gains 150, gross losses 100, PF 1.5.
    # Hence pooled H1 gross gains 350 and losses 200 => PF 1.75.
    assert audit.h1_profit_factor(2., 1.5, 100., 50.) == pytest.approx(1.75)
    with pytest.raises(ValueError):
        audit.h1_profit_factor(None, 1.5, 100., 50.)


def test_strict_q3_success_threshold_and_small_sample_failures():
    a = audit.assess_oos(1.21, 14.9, 2., trade_count=30, stopped=False)
    assert a["degradation_pct"] == pytest.approx(39.5)
    assert a["qualifies"] is True
    assert not audit.assess_oos(1.2, 14.9, 2., trade_count=30, stopped=False)["qualifies"]
    assert not audit.assess_oos(1.15, 14.9, 1.3, trade_count=30, stopped=False)["qualifies"]
    assert not audit.assess_oos(1.3, 15., 1.3, trade_count=30, stopped=False)["qualifies"]
    assert not audit.assess_oos(None, 0., 1.3, trade_count=0, stopped=False)["qualifies"]
    assert not audit.assess_oos(1.3, 10., 1.3, trade_count=30, stopped=True)["qualifies"]


def test_q3_sources_missing_fails_before_loading_any_market(tmp_path,monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Los arrays Q3 nunca deben leerse si falta un activo")
    monkeypatch.setattr(audit,"load_native_q3",forbidden)
    with pytest.raises(FileNotFoundError,match="Falta Q3"):
        audit.require_q3_files(tmp_path)


def test_lookahead_gate_does_not_use_future_q3_outcomes():
    times=pd.date_range("2026-07-01",periods=620,freq="5min",tz="UTC")
    scores=np.sin(np.arange(620)/19.).astype(float)
    base=pd.DataFrame({"outcome":np.ones(620,dtype="i1")},index=times)
    alt=base.copy()
    alt.iloc[400:,0]=-1
    a=audit.strictly_causal_p85(base,scores)
    b=audit.strictly_causal_p85(alt,scores)
    np.testing.assert_array_equal(a[:400],b[:400])
    assert set(a[:500])=={"conviction_warmup"}


def test_garch_ratio_and_admissions_are_past_only():
    n=100
    data=dict(timeframe=np.ones((n,8)),volatility=np.full(n,.001))
    genes=np.zeros(22)
    genes[0]=1.
    genes[9]=1.
    a,mask1,_=audit.local_asset_head(data,genes)
    changed=dict(data,volatility=data["volatility"].copy())
    changed["volatility"][80:]=.002
    b,mask2,_=audit.local_asset_head(changed,genes)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(mask1[:80],mask2[:80])


def test_sha_of_genes_is_bounded_and_deterministic():
    genes=np.arange(22,dtype="<f8")
    assert audit.digest_genome(genes)==audit.digest_genome(genes.copy())
    with pytest.raises(ValueError):
        audit.digest_genome(np.arange(21))


def test_q3_rejects_empty_or_2026h1_legacy_manifest(tmp_path):
    root=tmp_path/"eurusd"
    root.mkdir()
    (root/"universe.npz").write_bytes(b"not a real q3 source")
    (root/"manifest.json").write_text('{"schema":"pr40_m5_m15_symbol_universe_v1"}')
    with pytest.raises(ValueError,match="Q3 sin procedencia"):
        audit.load_native_q3("EURUSD",root/"universe.npz",root/"manifest.json")
