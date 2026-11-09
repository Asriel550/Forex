"""PR43: add 100 genetic generations to EACH sealed PR42 champion.

This is seeded continuation of the champion phenotype/22 weights, NOT
100 new initializations. Q1/Q2 IS only; Q3 never loaded. Original PR41/42
genomes remain immutable and are replay-verified before exploration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_m5_dynamic_upstream import key as pr41_rank
from scripts.run_pr41_native_asset_evolution import (
    GENE_COUNT_LOCAL, evaluate_local, genome_sha, result_summary, strict_asset_pass,
)
from scripts.run_pr42_isolated_silos import (
    SEEDS, SYMBOLS, SOURCE_SHA, frozen_candidate, native_problem,
)
from src.evolution_multiscale import POPULATION, evolve_population
from src.train import write_json

PROTOCOL = "PR43_continue_PR42_champions_100_gen_H1_IS_v1"
ADDITIONAL_GENERATIONS = 100
FIRST_GENERATION = 21
LAST_GENERATION = FIRST_GENERATION + ADDITIONAL_GENERATIONS - 1
NEW_EVALUATIONS = ADDITIONAL_GENERATIONS * POPULATION
OUTPUT = Path("reports/pr43-100-generation-continuations")


def _transport(value):
    """None PF is internal NaN, never a valid financial pass."""
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple)):
        return [_transport(v) for v in value]
    return value


def batch_fitness(population, problem):
    results = [evaluate_local(g, problem) for g in population]
    return {key: np.asarray([_transport(r[key]) for r in results])
            for key in results[0]}


def check_founder(frozen, founder, problem):
    """Reject changed champion, input data, replay economics or IS policy."""
    if frozen["input_universe_sha256"] != problem["is_snapshot_sha256"]:
        raise ValueError("El campeón y GARCH local no pertenecen al mismo H1")
    actual = evaluate_local(founder, problem)
    expected = frozen["champion"]
    if not strict_asset_pass(actual):
        raise ValueError("Fundador PR42 no satisface su filtro IS")
    comparisons = {
        "pf_q1": actual["quarter_pf"][0],
        "pf_q2": actual["quarter_pf"][1],
        "dd_semester_pct": actual["semester_dd"],
        "executed_hourly": actual["semester_rate"],
        "pnl_q1": actual["quarter_pnl_usd"][0],
        "pnl_q2": actual["quarter_pnl_usd"][1],
        "pnl_semester": actual["semester_pnl_usd"],
        "garch_lower": actual["regime_lower"],
        "garch_upper": actual["regime_upper"],
        "margin_threshold": actual["direction_margin_threshold"],
    }
    for name, value in comparisons.items():
        if (name not in expected or value is None or expected[name] is None
                or not np.isfinite(value) or not np.isfinite(expected[name])
                or not np.isclose(value, expected[name], atol=1e-7, rtol=0)):
            raise ValueError(f"Fundador no reproduce {name}")
    return actual


def _maybe_float(value):
    return float(value) if value is not None and np.isfinite(value) else None


def evolution_history(record):
    return dict(
        generation=int(record["generation"]),
        best_candidate=int(record["candidate"]),
        score=float(record["score"]),
        median_score=float(record["median_score"]),
        mutation_sigma=float(record["mutation_sigma"]),
        pf_q1=_maybe_float(record["quarter_pf"][0]),
        pf_q2=_maybe_float(record["quarter_pf"][1]),
        executed_hourly=float(record["semester_rate"]),
        dd_semester_pct=float(record["semester_dd"]),
        qualifies=bool(record["qualifies"]),
    )


def continue_champion(*, symbol, seed, report, universe, manifest,
                      golden=None, golden_manifest=None, output=OUTPUT):
    if symbol not in SYMBOLS or seed not in SEEDS:
        raise ValueError("Solo los 12 campeones originales PR42")
    frozen, founder = frozen_candidate(report, symbol, seed)
    problem = native_problem(symbol, universe, manifest,
                             golden=golden, golden_manifest=golden_manifest)
    if problem["symbol"] != symbol or problem["is_snapshot_sha256"] != SOURCE_SHA[symbol]:
        raise ValueError("Crossover de fuentes entre divisas")
    baseline = check_founder(frozen, founder, problem)

    # The PR41 stochastic state was not archived. Re-seed the evolutionary
    # operator from the FROZEN winning genome, not a zero genome or cross-pair
    # parent. This is new descent from the champion, not bitwise RNG resume.
    rng_seed = seed + 1000 * SYMBOLS.index(symbol)
    result = evolve_population(
        lambda population: batch_fitness(population, problem),
        founder.copy(), f"pr43_{symbol}_seed{seed}_from_PR42_champion",
        rng_seed, first_generation=FIRST_GENERATION,
        last_generation=LAST_GENERATION, ranking=pr41_rank,
    )
    history = [evolution_history(r) for r in result["generations"]]
    if (result["evaluations"] != NEW_EVALUATIONS or len(history) != ADDITIONAL_GENERATIONS
            or [r["generation"] for r in history] != list(range(21, 121))):
        raise ValueError("Faltan generaciones de continuidad")
    winner_genome = np.asarray(result["genome"], dtype="<f8")
    if winner_genome.shape != (GENE_COUNT_LOCAL,) or not np.isfinite(winner_genome).all():
        raise ValueError("Genoma posterior incompatible")
    winner = evaluate_local(winner_genome, problem)
    if (not np.isclose(winner["score"], result["champion"]["score"], atol=1e-10, rtol=0)
            or bool(winner["qualifies"]) != bool(result["champion"]["qualifies"])):
        raise ValueError("El campeón nuevo no reproduce su fitness")
    if genome_sha(founder) != frozen["champion_genome_sha256"]:
        raise ValueError("Se alteró el campeón original")
    report_out = dict(
        protocol=PROTOCOL, symbol=symbol, seed=seed,
        pr41_final_run_id=42576852134,
        source_pr42_generation=frozen["champion_generation"],
        source_pr42_genome_sha256=genome_sha(founder),
        source_pr42_genome=founder.tolist(),
        source_pr42_metrics=result_summary(baseline),
        source_pr42_qualified=True,
        started_from_champion=True, zero_initialized=False,
        stochastic_resume_exact=False,
        stochastic_resume_note="PR41 did not archive RNG and sigma state; population is reseeded from the signed winning genome.",
        first_new_generation=FIRST_GENERATION,
        last_new_generation=LAST_GENERATION,
        additional_generations=ADDITIONAL_GENERATIONS,
        population=POPULATION, new_evaluations=NEW_EVALUATIONS,
        rng_seed=rng_seed,
        evolved_champion_genome=winner_genome.tolist(),
        evolved_champion_sha256=genome_sha(winner_genome),
        evolved_champion_generation=int(result["champion"]["generation"]),
        evolved_champion_metrics=result_summary(winner),
        history=history,
        input_universe_sha256=problem["is_snapshot_sha256"],
        signal_timeframes=["M5", "M15"],
        garch_native_to_symbol=True,
        independent_bankroll=True, shared_account=False,
        cross_asset_evolution=False, q3_accessed=False, oos_evaluated=False,
        input_end_exclusive="2026-07-01T00:00:00Z",
        original_models_modified=False, auto_promoted=False,
        warning="100 additional generations reuse Q1/Q2 IS, not OOS evidence. Q3 untouched.",
    )
    directory = Path(output) / symbol.lower()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"seed-{seed}.json", report_out)
    print("PR43_CONTINUED_CHAMPION " + json.dumps(dict(
        symbol=symbol, seed=seed, original_generation=frozen["champion_generation"],
        additional_generations=ADDITIONAL_GENERATIONS,
        new_evaluations=NEW_EVALUATIONS,
        evolved_generation=report_out["evolved_champion_generation"],
        new_pf_q1=report_out["evolved_champion_metrics"]["pf_q1"],
        new_pf_q2=report_out["evolved_champion_metrics"]["pf_q2"],
        new_dd_h1=report_out["evolved_champion_metrics"]["dd_semester_pct"],
        new_executed_per_h=report_out["evolved_champion_metrics"]["executed_hourly"],
        q3_accessed=False), allow_nan=False), flush=True)
    return report_out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True, choices=SYMBOLS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    for arg in ("report", "universe", "manifest"):
        p.add_argument("--" + arg, required=True, type=Path)
    p.add_argument("--golden", type=Path)
    p.add_argument("--golden-manifest", type=Path)
    p.add_argument("--output", type=Path, default=OUTPUT)
    continue_champion(**vars(p.parse_args()))
