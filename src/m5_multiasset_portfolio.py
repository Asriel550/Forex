"""M5 multi-asset replay on ONE bankroll with a portfolio-wide hourly cap.

Input streams must contain externally VERIFIED, causal decisions of the
frozen PR38 GARCH=0% / p85 head. This module cannot create those streams
from price data and never uses Q3 or M1 indicators.
"""
from collections import Counter, deque
import numpy as np
import pandas as pd

from src.evolution_direction import unit_payoff
from src.evolution_lineage import MARKET

SYMBOLS=("EURUSD","GBPUSD","AUDUSD","USDCAD")
START=pd.Timestamp("2026-01-01T00:00:00Z")
Q2=pd.Timestamp("2026-04-01T00:00:00Z")
END=pd.Timestamp("2026-07-01T00:00:00Z")
REASONS=("accepted","conviction","conviction_warmup")


def check_streams(streams):
    if not isinstance(streams,dict) or not streams or set(streams)-set(SYMBOLS):
        raise ValueError("Canasta de símbolos no autorizada")
    for symbol,rows in streams.items():
        if not isinstance(rows,pd.DataFrame) or not {"side","outcome","gate_reason"}<=set(rows):
            raise ValueError("Stream M5 incompleto para "+symbol)
        times=rows.index
        if (not isinstance(times,pd.DatetimeIndex) or times.tz is None or
                len(times)==0 or times.hasnans or not times.is_unique or
                not times.is_monotonic_increasing or times[0]<START or times[-1]>=END
                or (np.diff(times.as_unit("ns").asi8)<MARKET.step.value).any()
                or not times.equals(times.floor("5min"))
                or not np.isin(rows.side,(-1,1)).all()
                or not np.isin(rows.outcome,(-1,0,1)).all()
                or not np.isin(rows.gate_reason,REASONS).all()):
            raise ValueError("Stream no causal, no M5, o fuera de H1: "+symbol)
    return True


def _replay(streams,account,risk,begin,finish,hours,cap):
    if not isinstance(cap,int) or not 1<=cap<=24 or not np.isfinite(hours) or hours<=0:
        raise ValueError("Cupo/horas inválidos")
    symbols=list(streams)
    events=[]
    for sym,df in streams.items():
        part=df.loc[(df.index>=begin)&(df.index<finish)]
        events.extend((ts,sym,int(side),int(outcome),str(reason))
                      for ts,side,outcome,reason in zip(
                          part.index,part.side,part.outcome,part.gate_reason,strict=True))
    # Deterministic, outcome-independent rotating tie breaker when symbols
    # signal on the same 5m close (avoid prioritizing EURUSD every time).
    events.sort(key=lambda z:(z[0],(SYMBOLS.index(z[1])-
                          z[0].value//MARKET.step.value)%len(SYMBOLS)))
    initial=float(account.initial_balance);stake=float(account.stake)
    limit=min(float(account.max_drawdown),float(risk.drawdown_limit))
    if not np.isfinite([initial,stake,limit]).all() or not 0<stake<=initial or not 0<limit<=1:
        raise ValueError("Cuenta heredada inválida")
    balance=peak=initial
    drawdown=0.
    stop=None
    pending=[]
    hourly=deque()
    counts=Counter()
    per={s:dict(wins=0,losses=0,refunds=0,pnl_usd=0.,
                gross_win=0.,gross_loss=0.,gate_accepted=0,
                executed=0,rejected_global_cap=0) for s in symbols}
    trades=[]
    max_concurrent=0
    def settle(now):
        nonlocal balance,peak,drawdown,stop
        matured=[p for p in pending if p[0]<=now]
        if not matured: return
        pending[:]=[p for p in pending if p[0]>now]
        step_profit=0.
        for expiry,sym,entry,side,outcome in matured:
            profit=float(unit_payoff(np.array([side]),np.array([outcome]),account)[0])*stake
            rec=per[sym]
            rec["executed"]+=1;rec["pnl_usd"]+=profit
            if profit>0: rec["wins"]+=1;rec["gross_win"]+=profit
            elif profit<0: rec["losses"]+=1;rec["gross_loss"]+=-profit
            else: rec["refunds"]+=1
            trades.append(dict(symbol=sym,entry_utc=entry.isoformat(),
                               expiry_utc=expiry.isoformat(),side=side,
                               outcome=outcome,pnl_usd=profit))
            step_profit+=profit
        # All contracts with same expiry settle simultaneously before any
        # drawdown stop is evaluated (not sequential in arbitrary symbol order).
        balance+=step_profit
        peak=max(peak,balance)
        drawdown=max(drawdown,100.*(1.-balance/peak))
        if stop is None and (balance<stake or drawdown+1e-10>=100*limit):
            stop=dict(utc=now.isoformat(),drawdown_pct=float(drawdown),
                      balance_usd=float(balance),
                      reason="insufficient_balance" if balance<stake else "drawdown")
    n=0
    while n<len(events):
        now=events[n][0];end=n+1
        while end<len(events) and events[end][0]==now:end+=1
        settle(now)
        while hourly and hourly[0]<=now-pd.Timedelta(hours=1):hourly.popleft()
        for t,sym,side,outcome,reason in events[n:end]:
            counts["proposals"]+=1
            if reason!="accepted":
                counts[reason]+=1
                continue
            counts["gate_accepted"]+=1
            per[sym]["gate_accepted"]+=1
            if stop:
                counts["account_stopped"]+=1
                continue
            if len(hourly)>=cap:
                counts["global_hourly_cap"]+=1
                per[sym]["rejected_global_cap"]+=1
                continue
            if balance-(len(pending)+1)*stake < -1e-8:
                counts["insufficient_free_capital"]+=1
                continue
            hourly.append(now)
            pending.append((now+MARKET.step,sym,now,side,outcome))
            counts["opened"]+=1
            max_concurrent=max(max_concurrent,len(pending))
        n=end
    settle(finish)
    if pending:raise ValueError("Liquidaciones posteriores al periodo IS")
    wins=sum(v["wins"] for v in per.values())
    losses=sum(v["losses"] for v in per.values())
    refunds=sum(v["refunds"] for v in per.values())
    gross_win=sum(v["gross_win"] for v in per.values())
    gross_loss=sum(v["gross_loss"] for v in per.values())
    pnl=sum(v["pnl_usd"] for v in per.values())
    if not np.isclose(balance-initial,pnl,atol=1e-8):
        raise ValueError("Ledger compartido no reconcilia")
    for v in per.values():
        v["profit_factor"]=v["gross_win"]/v["gross_loss"] if v["gross_loss"] else None
        v["executed_hourly"]=v["executed"]/hours
    return dict(
        observed_hours=float(hours),executed=wins+losses+refunds,
        wins=wins,losses=losses,refunds=refunds,
        pf=gross_win/gross_loss if gross_loss else None,
        pf_status=("finite" if gross_loss else "infinite_no_losses"
                   if gross_win else "no_trades"),
        pnl_usd=float(pnl),drawdown_pct=float(drawdown),
        executed_hourly=(wins+losses+refunds)/hours,
        stopped=stop is not None,stop=stop,
        starting_balance_usd=initial,ending_balance_usd=balance,
        max_simultaneously_open=max_concurrent,
        proposals=counts["proposals"],
        gate_accepted=counts["gate_accepted"],
        global_hourly_cap_rejections=counts["global_hourly_cap"],
        rejected_after_stop=counts["account_stopped"],
        rejected_no_capital=counts["insufficient_free_capital"],
        contributions=per,executions=trades,
    )


def replay_portfolio(streams,account,risk,*,hours,global_hourly_cap=2):
    """Three ledgers: separate quarter starts, plus one CONTINUOUS semester."""
    check_streams(streams)
    if (not isinstance(hours,dict) or set(hours)!={"Q1","Q2","semester"}
            or not np.isclose(hours["Q1"]+hours["Q2"],hours["semester"],atol=1e-7)):
        raise ValueError("Denominador de horas Q1/Q2/semestre no verificable")
    return dict(
        Q1=_replay(streams,account,risk,START,Q2,hours["Q1"],global_hourly_cap),
        Q2=_replay(streams,account,risk,Q2,END,hours["Q2"],global_hourly_cap),
        semester=_replay(streams,account,risk,START,END,hours["semester"],global_hourly_cap),
        symbols=list(streams),hourly_cap_global=global_hourly_cap,
        tie_break="5m timestamp rotating symbol order",
        single_shared_account=True,signal_timeframes=["M5","M15"],
        q3_accessed=False,
    )


def meets_portfolio_targets(report,*,min_symbols=4,min_pf=1.20,
                            min_executed_hourly=1.,max_dd=15.):
    if len(report["symbols"])<min_symbols or report["q3_accessed"] is not False:
        return False
    q1,q2,full=report["Q1"],report["Q2"],report["semester"]
    return bool(
        all(q["pf"] is not None and q["pf"]>min_pf and
            q["pnl_usd"]>0 and not q["stopped"] for q in (q1,q2))
        and full["pf"] is not None and full["pf"]>min_pf
        and full["executed_hourly"]>=min_executed_hourly
        and full["drawdown_pct"]<max_dd and not full["stopped"]
    )
