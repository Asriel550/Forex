"""Ablación M5 con admisión congelada y evaluación direccional pareada."""

import numpy as np
from scipy.stats import rankdata


DISAGREEMENT_WEIGHT = 2.
SPEARMAN_WEIGHT = .10
M15_ARMS = ('control', 'feature', 'anchor')
M15_WEIGHT_LIMIT = 6.
M15_ANCHOR_WEIGHT = 12.


def direction_genes(genome):
    return genome[len(genome) // 2:-1]


def with_direction(genome, genes):
    candidate = genome.copy()
    direction_genes(candidate)[:] = genes
    return candidate


def unit_payoff(side, change, account):
    """Liquida el lado elegido con stake unitario y la política de empates."""
    return np.where((side == 0.) | ((change == 0.) & (account.tie_policy == 'refund')), 0.,
                    np.where(side == change, account.payout, -1.))


def frozen_simulation(simulation, founder, frame, m15_bias=None, m15_config=None):
    """Prepara el término M15 en las mismas velas elegibles que la simulación."""
    if m15_bias is None:
        return FrozenDirectionSimulation(simulation, founder)
    bias = m15_bias.loc[frame.timestamp].to_numpy()[simulation.entries]
    volatility = frame.cond_volatility.to_numpy()[simulation.entries]
    anchor = m15_config['anchor_weight'] * (volatility > m15_config['high_volatility_cutoff'])
    return FrozenDirectionSimulation(simulation, founder, m15_bias=bias, m15_anchor=anchor)


def _centered_ranks(values):
    ranks = rankdata(values)
    return ranks - (len(ranks) + 1) / 2


class FrozenDirectionSimulation:
    """Usa la decisión de entrada y el umbral del fundador para todos los candidatos."""

    def __init__(self, simulation, founder, *, m15_bias=None, m15_anchor=0.):
        self.simulation = simulation
        entry, original_direction = simulation.decision_scores(founder)
        margin = np.abs(original_direction)
        self.threshold = float(founder[-1])
        eligible = (entry > 0.) & (margin > 0.)
        self.signals = int(eligible.sum())
        self.proposed = np.flatnonzero(eligible & (margin > self.threshold))
        self.mc_direction = np.where(simulation.mc_margin != 0., simulation.mc_margin, original_direction)
        self.mc_side = np.sign(self.mc_direction)
        self.resolved = simulation.change != 0.
        self.resolved_count = int(self.resolved.sum())
        self.target_ranks = _centered_ranks(simulation.change[self.resolved])
        self.target_norm = float(np.linalg.norm(self.target_ranks))
        self.mc_rho = self._spearman(simulation.mc_margin)
        self.mc_payoff = unit_payoff(self.mc_side, simulation.change, simulation.account)
        self.mc_hits = (self.mc_side == simulation.change) & self.resolved
        self.m15_bias = m15_bias
        self.m15_anchor = m15_anchor

    def _spearman(self, direction):
        ranks = _centered_ranks(direction[self.resolved])
        scale = float(np.linalg.norm(ranks)) * self.target_norm
        return float(np.dot(ranks, self.target_ranks) / scale) if scale else None

    def direction_scores(self, genome, *, m15_weight=0.):
        _, raw_direction = self.simulation.decision_scores(genome)
        if self.m15_bias is not None:
            raw_direction = raw_direction + (m15_weight + self.m15_anchor) * self.m15_bias
        # En un empate exacto se conserva la entrada con dirección MC o, si
        # también empata MC, con el lado original del fundador.
        direction = np.where(raw_direction != 0., raw_direction, self.mc_direction)
        return raw_direction, direction

    def evaluate(self, genome, *, m15_weight=0.):
        raw_direction, direction = self.direction_scores(genome, m15_weight=m15_weight)
        metrics = self.simulation.simulate_trades(
            direction, self.proposed, signals=self.signals, activation_threshold=self.threshold,
        )
        side = np.sign(direction)
        hits = (side == self.simulation.change) & self.resolved
        disagree = (side != self.mc_side) & self.resolved
        count = int(disagree.sum())
        rho = self._spearman(raw_direction)
        paired_gain = float(np.mean(unit_payoff(side, self.simulation.change, self.simulation.account)
                                    - self.mc_payoff))
        # Un acuerdo aporta cero al diferencial de payout.
        score = DISAGREEMENT_WEIGHT * paired_gain + SPEARMAN_WEIGHT * ((rho or 0.) - (self.mc_rho or 0.))
        metrics['directional'] = dict(
            score=score, eligible_bars=len(side), resolved_bars=self.resolved_count,
            spearman=rho, mc_spearman=self.mc_rho, paired_gain_per_eligible_bar=paired_gain,
            accuracy=float(hits.sum() / self.resolved_count) if self.resolved_count else None,
            mc_accuracy=float(self.mc_hits.sum() / self.resolved_count) if self.resolved_count else None,
            disagreements=count,
            disagreement_wins=int(hits[disagree].sum()),
            mc_disagreement_wins=int(self.mc_hits[disagree].sum()),
        )
        return metrics
