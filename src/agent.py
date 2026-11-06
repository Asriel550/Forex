"""Factoría PPO/MLP para BinaryOptionsEnv, con ejecución CPU reproducible."""
from __future__ import annotations

import math

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import torch


def build_agent(env, *, seed=42, n_steps=2048, batch_size=64,
                learning_rate=3e-4, n_epochs=10, ent_coef=.02, verbose=1):
    """Actor/crítico MLP separados 64x64. No usa API externa ni GPU.

    La reproducibilidad se espera con las mismas versiones/hardware; no es
    una promesa bit a bit entre distintas plataformas o versiones de Torch.
    """
    if n_steps < 2 or batch_size < 2 or batch_size > n_steps or n_steps % batch_size:
        raise ValueError("n_steps >= batch_size >= 2 y n_steps divisible por batch_size")
    if n_epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("n_epochs y learning_rate deben ser positivos")
    if not math.isfinite(ent_coef) or not 0<=ent_coef<=.1:
        raise ValueError('ent_coef debe ser finito y estar entre 0 y 0.1')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed, using_cuda=False)
    return PPO("MlpPolicy", env, device="cpu", seed=seed,
               learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
               n_epochs=n_epochs, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
               ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
               policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                  activation_fn=torch.nn.Tanh), verbose=verbose)


def load_agent(path, env=None):
    """La inferencia necesita además normalizer.json y el contrato del manifiesto."""
    torch.set_num_threads(1)
    return PPO.load(path, env=env, device="cpu")

