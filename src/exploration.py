"""Auditoría de exploración de PPO: entrenamiento estocástico != decisión greedy.

Las inspecciones usan no_grad y ningún muestreo adicional: no alteran el RNG.
No usan datos del holdout ni contribuyen al score de selección del checkpoint.
"""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from .metrics import _atomic_write

LABELS=('PUT','NEUTRO','CALL')


def policy_snapshot(model,observations):
    values=np.asarray(observations,dtype=np.float32)
    if len(values)==0:return None
    with torch.no_grad():
        tensor,_=model.policy.obs_to_tensor(values)
        distribution=model.policy.get_distribution(tensor).distribution
        probabilities=distribution.probs.cpu().numpy()
        entropy=float(distribution.entropy().mean().cpu())
    if probabilities.ndim!=2 or probabilities.shape[1]!=3 or not np.isfinite(probabilities).all() or not np.isfinite(entropy):
        raise ValueError('Distribución PPO inválida')
    return dict(observations=len(values),mean_entropy_nats=entropy,
                mean_action_probabilities={label:float(probabilities[:,i].mean()) for i,label in enumerate(LABELS)})


class ExplorationAuditCallback(BaseCallback):
    """Cuenta propuestas y ejecución del entorno; resume probabilidades por rollout."""
    def __init__(self,output_dir):
        super().__init__(verbose=0)
        self.output_dir=Path(output_dir);self.history=[]
        self.total_requested=np.zeros(3,dtype=np.int64)
        self.total_executed=np.zeros(3,dtype=np.int64)
        self.total_blocked=0;self.total_missed=0

    def _on_rollout_start(self):
        self.requested=np.zeros(3,dtype=np.int64);self.executed=np.zeros(3,dtype=np.int64)
        self.blocked=0;self.missed=0

    def _on_step(self):
        for info in self.locals.get('infos',[]):
            self.requested[int(info['requested_action'])]+=1
            self.executed[int(info['action'])]+=1
            self.blocked+=int(info['blocked']);self.missed+=int(info.get('missed_opportunity',False))
        return True

    def _observations(self):
        data=self.model.rollout_buffer.observations.reshape(-1,*self.model.observation_space.shape)
        return data[np.linspace(0,len(data)-1,min(len(data),256),dtype=int)]

    def _write(self,name,data):
        _atomic_write(self.output_dir/name,lambda p:Path(p).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'))

    def _on_rollout_end(self):
        snapshot=policy_snapshot(self.model,self._observations())
        row=dict(timesteps=int(self.num_timesteps),ent_coef=float(self.model.ent_coef),
                 policy_snapshot_phase='before PPO update',policy=snapshot,
                 requested={k:int(self.requested[i]) for i,k in enumerate(LABELS)},
                 executed={k:int(self.executed[i]) for i,k in enumerate(LABELS)},
                 blocked_attempts=self.blocked,missed_opportunities=self.missed)
        self.total_requested+=self.requested;self.total_executed+=self.executed
        self.total_blocked+=self.blocked;self.total_missed+=self.missed
        self.history.append(row);self._write('exploration_history.json',self.history)
        self.logger.record('exploration/entropy_nats',snapshot['mean_entropy_nats'])
        self.logger.record('exploration/call_requests',row['requested']['CALL'])
        self.logger.record('exploration/put_requests',row['requested']['PUT'])
        self.logger.record('exploration/executed_contracts',int(self.executed[0]+self.executed[2]))
        self.logger.record('exploration/blocked_attempts',self.blocked)

    def _on_training_end(self):
        self._write('exploration_summary.json',dict(actual_timesteps=int(self.num_timesteps),
            ent_coef=float(self.model.ent_coef),evaluation_deterministic=True,
            requested={k:int(self.total_requested[i]) for i,k in enumerate(LABELS)},
            executed={k:int(self.total_executed[i]) for i,k in enumerate(LABELS)},
            blocked_attempts=self.total_blocked,missed_opportunities=self.total_missed,
            final_training_policy=policy_snapshot(self.model,self._observations()),
            caveat='Training proposals are stochastic; final training policy is not necessarily the selected checkpoint.'))
