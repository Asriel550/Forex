"""Reloj de decisiones elegibles para entrenamiento binario, sin forzar acciones.

Se omiten SOLO esperas gratuitas sin setup. No se usan resultados futuros para
seleccionar filas. El PPO aprende en tiempo de oportunidades (gamma por decisión),
no por minutos. Validación siempre usa el entorno completo sin este wrapper.
"""
import gymnasium as gym
import numpy as np
from .decision import mc_probability_lower

class OpportunityTraining(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        if not env.risk.enabled: raise ValueError("El entrenamiento por oportunidades requiere filtro de riesgo activo")
        up,down,_=env._probabilities.T
        confidence=np.maximum(up,down)
        lower=mc_probability_lower(confidence,env._n_paths,env.risk.mc_z)
        self.allowed=(confidence>env.risk.min_confidence)&(env.account.payout*lower-(1-lower)>env.risk.min_expected_edge)
        self.starts=env._eligible[self.allowed[env._eligible]]
        if not len(self.starts): raise ValueError('Train no contiene setups elegibles; no se relajan filtros ni se usa validación')

    def reset(self,*,seed=None,options=None):
        # Seed de Gym, sin usar resultados de liquidación ni validación.
        self.env.reset(seed=seed)
        options=dict(options or {})
        if 'start_index' not in options: options['start_index']=int(self.env.np_random.choice(self.starts))
        return self.env.reset(options=options)

    def step(self,action):
        obs,reward,terminated,truncated,first=self.env.step(action)
        saved={k:v for k,v in first.items() if k in ('requested_action','action','blocked','missed_opportunity','mc_confidence','mc_lower_bound','expected_edge_lower','filter_reason','mc_prob_up','mc_prob_down','mc_prob_tie','entry_index','entry_price','settlement_price','profit','outcome','expiry_time')}
        skipped=0
        if not (terminated or truncated) and not self.allowed[self.env._index]:
            begin=self.env._index
            last=self.env._end-self.env.expiry_candles
            candidates=np.flatnonzero(self.allowed[begin:last+1])
            destination=begin+int(candidates[0]) if len(candidates) else last+1
            skipped=destination-begin
            # Equivalente a N esperas gratuitas: ni precio futuro ni recompensa
            # intervienen en el salto. No hay posición abierta en binarias.
            self.env._index=destination;self.env.steps+=skipped
            self.env.trade_activity*=self.env.risk.activity_decay**skipped
            self.env.previous_direction=0.;self.env.previous_reward=0.
            truncated=destination>last
            self.env._done=bool(truncated)
            obs=self.env._observation()
        info=dict(self.env._info(),**saved,skipped_no_setup=skipped,
            end_reason='data_or_time_limit' if truncated else first.get('end_reason'))
        return obs,float(reward),terminated,truncated,info
