"""Forex spot simulado: posición objetivo, spread al abrir y equity marcada.

Cuenta USD, par EUR/USD, lote fijo. No usa payout binario ni expiry_candles.
Spread completo se carga al abrir (coste de ida/vuelta asignado a entrada).
Sin financiación overnight, slippage, OHLC intrabar o conexión a broker.
"""
from dataclasses import dataclass
import numpy as np
from .environment import BinaryOptionsEnv
from .market import MarketConfig
from .decision import mc_probability_lower

@dataclass(frozen=True)
class SpotConfig:
    lots: float = .1
    contract_size: float = 100000.
    leverage: float = 100.
    spread_pips: float = 1.5
    def __post_init__(self):
        if not np.isfinite([self.lots,self.contract_size,self.leverage,self.spread_pips]).all() or min(self.lots,self.contract_size,self.leverage)<=0 or self.spread_pips<0:
            raise ValueError('Lote, apalancamiento o spread inválido')
    @property
    def units(self): return self.lots*self.contract_size


class ForexSpotEnv(BinaryOptionsEnv):
    """0=SHORT, 1=FLAT/cerrar, 2=LONG; repetir objetivo conserva posición."""
    def __init__(self,df,*,spot=None,market=None,**kwargs):
        self.spot=spot or SpotConfig()
        market=market or MarketConfig(trading_mode='forex_spot')
        if market.trading_mode!='forex_spot': raise ValueError('Contrato spot requerido')
        super().__init__(df,market=market,**kwargs)
        self.feature_names=(*self.feature_names[:-7],"equity_return","drawdown","position","floating_pips","margin_utilization","spread_pips","trade_activity")

    def reset(self,*,seed=None,options=None):
        self.position=0;self.entry_price=0.;self.floating=0.;self.current_cost=0.
        obs,info=super().reset(seed=seed,options=options)
        self.cash=self.balance
        return obs,info

    def _observation(self):
        position=getattr(self,'position',0);floating=getattr(self,'floating',0.)
        margin=self.spot.units*self._close[self._index]/self.spot.leverage if position else 0.
        state=np.array([self.balance/self.account.initial_balance-1,
            1-self.balance/self.peak_balance,position,floating/(self.spot.units*.0001),
            margin/max(self.balance,1e-9),self.spot.spread_pips,self.trade_activity],dtype=np.float32)
        return np.concatenate([self._market[self._index],state]).astype(np.float32)

    def _info(self):
        info=super()._info();info.update(position=getattr(self,'position',0),
            floating_pnl=getattr(self,'floating',0.),open_positions=int(bool(getattr(self,'position',0))))
        return info

    def step(self,action):
        if self._done: raise RuntimeError('Llamar reset antes de step')
        if not self.action_space.contains(action): raise ValueError('Acción inválida')
        requested=int(action);target=requested-1;index=self._index
        up,down,tie=self._probabilities[index]
        confidence=float(up if target>0 else down if target<0 else max(up,down))
        blocked=bool(self.risk.enabled and target!=0 and confidence<=self.risk.min_confidence)
        if blocked: target=0
        mid=self._close[index];next_mid=self._close[index+1];before=self.balance
        old_dd=1-before/self.peak_balance;closed=[]
        def close_position(price):
            if not self.position:return
            gross=self.spot.units*self.position*(price-self.entry_price)
            net=gross-self.current_cost
            self.cash+=gross;closed.append(dict(action=self.position+1,profit=float(net),
                outcome='win' if net>0 else 'loss' if net<0 else 'tie'))
            self.trades+=1;self.wins+=int(net>0);self.losses+=int(net<0);self.ties+=int(net==0)
            self.position=0;self.floating=0.;self.current_cost=0.
        if target!=self.position:
            close_position(mid)
            if target:
                cost=self.spot.units*.0001*self.spot.spread_pips
                required=self.spot.units*mid/self.spot.leverage
                if self.cash-cost<required:
                    blocked=True;target=0
                else:
                    self.position=target;self.entry_price=mid;self.current_cost=cost;self.cash-=cost
        self.floating=self.spot.units*self.position*(next_mid-self.entry_price) if self.position else 0.
        self.balance=self.cash+self.floating
        self.peak_balance=max(self.peak_balance,self.balance)
        self.max_drawdown=max(self.max_drawdown,1-self.balance/self.peak_balance)
        self._index+=1;self.steps+=1
        self.trade_activity=self.risk.activity_decay*self.trade_activity+(1-self.risk.activity_decay)*float(target!=0)
        self.previous_direction=float(self.position)
        required=self.spot.units*next_mid/self.spot.leverage if self.position else 0.
        margin_call=bool(self.position and self.balance<required)
        limit=min(self.account.max_drawdown,self.risk.drawdown_limit) if self.risk.enabled else self.account.max_drawdown
        terminated=margin_call or self.balance<=0 or self.max_drawdown+1e-12>=limit
        truncated=not terminated and self._index>=self._end
        if terminated or truncated:
            close_position(next_mid);self.balance=self.cash
        pnl=self.balance-before
        scale=max(self.spot.units*mid*self._sigma[index],.01)
        reward=pnl/scale
        if self.risk.enabled:
            reward-=self.risk.blocked_penalty*blocked
            reward-=self.risk.drawdown_penalty*max(0.,1-self.balance/self.peak_balance-old_dd)
            if terminated:reward-=self.risk.stop_penalty
        self.previous_reward=float(reward);self._done=bool(terminated or truncated)
        info=self._info();info.update(requested_action=requested,action=int(target+1),blocked=blocked,
            missed_opportunity=False,mc_confidence=confidence,
            mc_lower_bound=float(mc_probability_lower(confidence,self._n_paths,self.risk.mc_z)),
            expected_edge_lower=0.,filter_reason='spot_confidence_or_margin' if blocked else 'spot_target',
            mc_prob_up=float(up),mc_prob_down=float(down),mc_prob_tie=float(tie),
            profit=float(pnl),outcome='mark_to_market',closed_trades=closed,margin_call=margin_call,
            entry_index=index,entry_price=float(mid),settlement_price=float(next_mid),
            expiry_time=(self._times[self._index]+self.bar_step).isoformat(),
            end_reason='margin_call' if margin_call else 'risk_stop' if terminated else 'data_or_time_limit' if truncated else None)
        return self._observation(),float(reward),bool(terminated),bool(truncated),info
