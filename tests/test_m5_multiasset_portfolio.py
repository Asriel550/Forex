"""PR40 unit tests: synthetic fixtures are NEVER empirical FX research."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.m5_multiasset_portfolio import (
    SYMBOLS,START,Q2,END,check_streams,replay_portfolio,
    meets_portfolio_targets,
)
from src.replay_engine import replay_contracts
from scripts.check_pr40_data_readiness import validate_inventory,INVENTORY


ACCOUNT=SimpleNamespace(
    initial_balance=1000.,stake=10.,payout=.8,
    tie_policy="refund",max_drawdown=.20,
)
RISK=SimpleNamespace(drawdown_limit=.20)
HOURS=dict(Q1=1100.,Q2=1100.,semester=2200.)


def stream(symbol, times, sides, outcomes, reasons=None):
    idx=pd.DatetimeIndex(pd.to_datetime(times,utc=True))
    return pd.DataFrame(
        dict(side=np.asarray(sides,dtype="i1"),
             outcome=np.asarray(outcomes,dtype="i1"),
             gate_reason=np.asarray(
                 reasons if reasons is not None else ["accepted"]*len(times)
             )),
        index=idx,
    )


def test_original_single_asset_m5_cap_and_payout_are_identical():
    times=pd.date_range("2026-01-05T00:00:00Z",periods=510,freq="5min")
    reasons=["conviction_warmup"]*500+["accepted"]*10
    sides=np.array([1]*510,dtype="i1")
    outcomes=np.array([1,-1]*255,dtype="i1")
    df=stream("EURUSD",times,sides,outcomes,reasons)
    single=replay_portfolio({"EURUSD":df},ACCOUNT,RISK,hours=HOURS)
    original,_=replay_contracts(
        pd.DataFrame({"outcome":outcomes},index=times),sides,ACCOUNT,RISK,
        observed_hours=HOURS["Q1"],hourly_cap=2,
        candidate_reasons=np.asarray(reasons,dtype="<U24"),
    )
    expected=original["operational"]
    actual=single["Q1"]
    assert actual["executed"]==expected["trades"]
    assert actual["pnl_usd"]==pytest.approx(expected["pnl_usd"])
    assert actual["pf"]==pytest.approx(expected["profit_factor"])
    assert actual["drawdown_pct"]==pytest.approx(expected["max_drawdown_pct"])
    assert actual["executed_hourly"]==pytest.approx(expected["contracts_per_observed_hour"])


def test_global_hourly_cap_applies_across_pairs_not_four_independent_caps():
    t=pd.Timestamp("2026-02-02T12:00:00Z")
    streams={symbol:stream(symbol,[t],[1],[1]) for symbol in SYMBOLS}
    result=replay_portfolio(streams,ACCOUNT,RISK,hours=HOURS)
    q1=result["Q1"]
    assert q1["gate_accepted"]==4
    assert q1["executed"]==2
    assert q1["opened"] if "opened" in q1 else q1["executed"]==2
    assert q1["global_hourly_cap_rejections"]==2
    assert q1["max_simultaneously_open"]==2
    assert len([s for s,c in q1["contributions"].items() if c["executed"]])==2
    assert q1["executed_hourly"]==pytest.approx(2/HOURS["Q1"])
    assert sum(c["pnl_usd"] for c in q1["contributions"].values())==pytest.approx(q1["pnl_usd"])


def test_sharing_drawdown_stop_rejects_all_symbols_after_loss():
    account=SimpleNamespace(initial_balance=100.,stake=10.,payout=.8,
                            tie_policy="refund",max_drawdown=.15)
    risk=SimpleNamespace(drawdown_limit=.15)
    t=pd.Timestamp("2026-02-04T09:00:00Z")
    streams={
        "EURUSD":stream("EURUSD",[t,t+pd.Timedelta(minutes=5)],[1,1],[-1,1]),
        "GBPUSD":stream("GBPUSD",[t,t+pd.Timedelta(minutes=5)],[1,1],[-1,1]),
    }
    result=replay_portfolio(streams,account,risk,hours=HOURS)
    assert result["Q1"]["executed"]==2
    assert result["Q1"]["drawdown_pct"]==pytest.approx(20.)
    assert result["Q1"]["stopped"] is True
    assert result["Q1"]["rejected_after_stop"]==2
    assert result["semester"]["executed"]==2
    assert result["semester"]["stopped"] is True


def test_completed_quarters_and_shared_semester_differ():
    a=pd.Timestamp("2026-02-02T10:00:00Z")
    b=pd.Timestamp("2026-05-03T10:00:00Z")
    streams={
        "EURUSD":stream("EURUSD",[a,b],[1,1],[-1,1]),
        "GBPUSD":stream("GBPUSD",[a,b],[1,1],[-1,1]),
    }
    account=SimpleNamespace(initial_balance=100.,stake=10.,payout=.8,
                            tie_policy="refund",max_drawdown=.15)
    risk=SimpleNamespace(drawdown_limit=.15)
    p=replay_portfolio(streams,account,risk,hours=HOURS)
    assert p["Q1"]["stopped"] is True
    assert p["Q2"]["executed"]==2
    assert p["semester"]["executed"]==2  # stopped in Q1, no reset in H1
    assert p["semester"]["stopped"] is True
    assert p["q3_accessed"] is False


def test_q3_and_m1_features_are_rejected():
    jan=pd.Timestamp("2026-03-03T13:00:00Z")
    q3=pd.Timestamp("2026-07-01T00:00:00Z")
    with pytest.raises(ValueError):
        replay_portfolio({"EURUSD":stream("EURUSD",[q3],[1],[1])},
                         ACCOUNT,RISK,hours=HOURS)
    df=stream("EURUSD",[jan],[1],[1])
    df["m1_body_proxy"]=.7
    with pytest.raises(ValueError,match="incompleto"):
        check_streams({"EURUSD":df})
    bad=stream("EURUSD",[jan,pd.Timestamp("2026-03-03T13:02:00Z")],
               [1,1],[1,1])
    with pytest.raises(ValueError):
        check_streams({"EURUSD":bad})


def test_no_synthetic_fixture_can_claim_financial_acceptance():
    t=pd.Timestamp("2026-02-02T12:00:00Z")
    p=replay_portfolio(
        {symbol:stream(symbol,[t],[1],[1]) for symbol in SYMBOLS},
        ACCOUNT,RISK,hours=HOURS
    )
    assert meets_portfolio_targets(p) is False
    assert p["Q1"]["pf"] is None  # no losses != verified PF
    assert p["semester"]["executed_hourly"]<1.


def test_missing_real_non_eurusd_data_blocks_all_portfolio_claims():
    readiness=validate_inventory(INVENTORY)
    assert readiness["verified_assets"]==["EURUSD"]
    assert readiness["missing_assets"]==["GBPUSD","AUDUSD","USDCAD"]
    assert readiness["ready_for_real_backtest"] is False
    assert readiness["portfolio_pf_q1"] is None
    assert readiness["portfolio_executed_contracts_per_hour"] is None
    assert readiness["portfolio_max_dd_pct"] is None
    assert readiness["q3_accessed"] is False
