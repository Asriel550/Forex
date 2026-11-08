import numpy as np
import pandas as pd
import pytest

from src.adaptive_gate import AdaptiveGate, rolling_m15_regime
from src.decision import RiskConfig
from src.environment import AccountConfig
from src.replay_engine import replay_contracts


def test_cutoff_excludes_current_proposal_and_keeps_rejected_proposals():
    prior = np.linspace(0., 2., 500)
    gate = AdaptiveGate(warmup_scores=prior)
    first = gate.decide(3., volatility=2., ceiling=1.)
    assert first.cutoff == pytest.approx(np.quantile(np.tanh(prior), .85))
    assert first.reason == 'volatility'
    following = gate.decide(3., volatility=0., ceiling=1.)
    assert following.cutoff == pytest.approx(np.quantile(np.tanh(np.r_[prior[1:], 3.]), .85))
    assert following.reason == 'accepted'
    tied = AdaptiveGate(warmup_scores=np.full(500, 3.))
    for i in range(500):
        tied.observe_settlement(i % 4 != 0, i % 2 == 0)
    decision = tied.decide(3., volatility=0., ceiling=1.)
    assert decision.mean_conviction > .80
    assert decision.percentile == .85 and not decision.mastery_evidence
    assert decision.reason == 'conviction'


def test_mastery_needs_both_regimes_and_reopens_in_confirmed_steps():
    gate = AdaptiveGate(warmup_scores=np.full(500, 2.))
    for _ in range(500):
        gate.observe_settlement(True, False)
    assert not gate.decide(3., 2., 1.).mastery_evidence
    for i in range(500):
        gate.observe_settlement(True, i % 2 == 0)
    first = gate.decide(3., 2., 1.)
    assert first.mastery_evidence and first.percentile == .85
    for expected in (.60, .30, 0.):
        for i in range(100):
            gate.observe_settlement(True, i % 2 == 0)
        decision = gate.decide(3., 2., 1.)
        assert decision.percentile == expected
    assert decision.reason == 'accepted' and decision.cutoff == 0.
    for _ in range(12):
        gate.observe_settlement(False, True)
    deteriorated = gate.decide(4., 2., 1.)
    assert deteriorated.percentile == .85 and deteriorated.reason == 'volatility'


def test_m15_regime_uses_prior_closed_candles_and_rejects_stale_data():
    closes = pd.date_range('2026-03-01T00:00Z', periods=195, freq='15min')
    values = pd.Series(np.arange(len(closes), dtype=float), index=closes)
    time = closes[-1]
    decisions = pd.DatetimeIndex([time - pd.Timedelta(minutes=5), time,
                                  time + pd.Timedelta(minutes=10), time + pd.Timedelta(minutes=15)])
    result = rolling_m15_regime(values, decisions)
    assert result.volatility.tolist()[:3] == [193., 194., 194.]
    assert result.ceiling.iloc[1] == pytest.approx(np.quantile(values.iloc[-193:-1], .8))
    assert result.iloc[-1].isna().all()
    changed = values.copy()
    changed.iloc[-1] = 1e6
    alternative = rolling_m15_regime(changed, decisions)
    assert result.ceiling.equals(alternative.ceiling)


def test_adaptive_admission_keeps_risk_cap_and_account_stop():
    times = pd.date_range('2026-03-01T00:00Z', periods=3, freq='5min')
    rows = pd.DataFrame({'outcome': [-1, 1, 1]}, index=times)
    result, trace = replay_contracts(rows, [1, 1, 1], AccountConfig(initial_balance=50.), RiskConfig(),
                                     observed_hours=1., hourly_cap=2, candidate_reasons=['accepted'] * 3)
    assert trace.gate_reason.tolist() == ['accepted', 'accepted', 'hourly_cap']
    assert trace.exit_reason.tolist() == ['expiry', 'account_stopped', 'not_entered']
    assert result['operational']['account_stop']['drawdown_pct'] == pytest.approx(20.)
