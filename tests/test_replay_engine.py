import numpy as np
import pandas as pd
import pytest

from src.decision import RiskConfig
from src.environment import AccountConfig
from src.replay_engine import ContractExits, replay_contracts


def cohort(minutes, probabilities, outcome):
    times = pd.Timestamp('2026-03-02T10:00:00Z') + pd.to_timedelta(minutes, unit='min')
    return pd.DataFrame(dict(mc_prob_up=probabilities, mc_prob_down=1 - np.asarray(probabilities),
                             mc_prob_tie=0., cond_volatility=.0002, outcome=outcome), index=times)


def replay(rows, account, *, exits=None):
    risk = RiskConfig(min_confidence=.57, min_expected_edge=.01, mc_z=2.24, drawdown_limit=.20)
    return replay_contracts(rows, np.ones(len(rows)), account, risk, observed_hours=2., n_paths=10000,
                            max_volatility=.0004, hourly_cap=2, exits=exits)


def test_gate_reasons_and_trailing_stop_have_separate_denominators():
    rows = cohort([0, 5, 10, 15, 20, 25, 75], [.55, .5701, .8, .8, .8, .8, .8], [1, 1, 1, -1, 1, 1, 1])
    rows.iloc[2, rows.columns.get_loc('cond_volatility')] = .0005
    result, trace = replay(rows, AccountConfig(initial_balance=10., stake=2.))
    assert trace.gate_reason.tolist() == [
        'confidence', 'expected_edge', 'volatility', 'accepted', 'accepted', 'hourly_cap', 'accepted',
    ]
    assert result['filters']['rejected_probability_or_volatility_pct'] == pytest.approx(300/7)
    assert result['filters']['rejected_pct'] == pytest.approx(400/7)
    assert result['gate_only_at_expiry']['trades'] == 3
    actual = result['operational']
    assert actual['pnl_usd'] == -2.
    assert actual['return_on_staked_pct'] == -100.
    assert actual['account_stop']['drawdown_pct'] == pytest.approx(20.)
    assert actual['unfilled_after_stop'] == 2
    assert trace.loc[3, 'settlement_utc'] == rows.index[3] + pd.Timedelta(minutes=5)


def test_early_exits_use_first_executable_quote_for_the_entered_contract(tmp_path):
    rows = cohort([0, 5, 60], [.8, .8, .8], [1, -1, 1])
    first, second, third = rows.index
    quotes = pd.DataFrame(dict(entry_utc=[first, first, second, third],
                               quote_utc=[first + pd.Timedelta(minutes=2), first + pd.Timedelta(minutes=1),
                                          second + pd.Timedelta(minutes=1), third + pd.Timedelta(minutes=1)],
                               side=['CALL', 'CALL', 'CALL', 'PUT'], sell_proceeds_usd=[16., 6., 14., 5.]))
    quotes['expiry_utc'] = quotes.entry_utc + pd.Timedelta(minutes=5)
    path = tmp_path/'sell-quotes.csv'
    quotes.to_csv(path, index=False)
    exits = ContractExits.from_csv(path, stop_loss=.4, take_profit=.4)
    result, trace = replay(rows, AccountConfig(), exits=exits)
    assert trace.exit_reason.tolist() == ['stop_loss', 'take_profit', 'expiry']
    assert trace.pnl_usd.tolist() == [-4., 4., 8.]
    assert result['operational']['quoted_trades'] == 2
    assert result['operational']['profit_factor'] == 3.
    assert result['operational']['return_on_staked_pct'] == pytest.approx(800/30)
    quotes.loc[0, 'quote_utc'] = quotes.loc[0, 'expiry_utc']
    quotes.to_csv(path, index=False)
    with pytest.raises(ValueError, match='fuera de la vida'):
        ContractExits.from_csv(path, stop_loss=.4, take_profit=.4)
