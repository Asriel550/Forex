"""PR43: 100 FRESH genetic evolutions for each native M5/M15 Forex asset.

Run 20 generations * 128 independent genomes per seed, 100 seeds/asset.
IS ONLY: 2026 Q1/Q2, Q3 never loaded or used. No shared account,
cross-asset features, PR41 champion warm starts, or automatic promotion.
One CI matrix shard runs exactly FIVE seeds for ONE symbol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_pr41_native_asset_evolution import (
    GENE_COUNT_LOCAL, asset_head, evaluate_local, genome_sha,
    result_summary, strict_asset_pass,
)
from scripts.run_m5_dynamic_upstream import key as strict_key
from scripts.run_pr42_isolated_silos import native_problem, SOURCE_SHA
from src.evolution_multiscale import POPULATION, evolve_population
from src.m5_multiasset_portfolio import SYMBOLS
from src.train import write_json


PROTOCOL = "PR43_100_fresh_per_asset_GARCH_H1_IS_v1"
FIRST_SEED = 1000
SEEDS_PER_ASSET = 100
SEEDS_PER_SHARD = 5
SHARDS = SEEDS_PER_ASSET // SEEDS_PER_SHARD
FIRST_GENERATION, LAST_GENERATION = 1, 20
EVALUATIONS_PER_SEED = (LAST_GENERATION - FIRST_GENERATION + 1) * POPULATION
OUTPUT = Path("reports/pr43-100-native-trainings")
ZERO = np.zeros(GENE_COUNT_LOCAL, dtype="<f8")


def seeds_for_shard(shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("Shard fuera de los 20 bloques predeclarados")
    return tuple(range(FIRST_SEED + shard * SEEDS_PER_SHARD,
                       FIRST_SEED + (shard + 1) * SEEDS_PER_SHARD))


def transport(value):
    """Undefined PF gets NaN internally, never passes the strict qualification gate."""
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple)):
        return [transport(x) for x in value]
    return value


def evaluate_batch(genomes, problem):
    records = [evaluate_local(g, problem) for g in genomes]
    return {name: np.asarray([transport(r[name]) for r in records])
            for name in records[0]}


def history_row(record):
    def maybe(value):
        return float(value) if value is not None and np.isfinite(value) else None
    return dict(
        generation=int(record["generation"]),
        best_candidate=int(record["candidate"]),
        score=float(record["score"]),
        median_score=float(record["median_score"]),
        pf_q1=maybe(record["quarter_pf"][0]),
        pf_q2=maybe(record["quarter_pf"][1]),
        dd_semester_pct=float(record["semester_dd"]),
        executed_hourly=float(record["semester_rate"]),
        mutation_sigma=float(record["mutation_sigma"]),
        qualified=bool(record["qualifies"]),
    )


def train_seed(symbol, seed, problem, *, output=OUTPUT):
    """Separate RNG and zero founder for EVERY training, no previous winner as input."""
    if symbol not in SYMBOLS or type(seed) is not int or not FIRST_SEED <= seed < FIRST_SEED + SEEDS_PER_ASSET:
        raise ValueError("Semilla fuera de las 100 nuevas corridas IS")
    if problem["symbol"] != symbol or problem["is_snapshot_sha256"] != SOURCE_SHA[symbol]:
        raise ValueError("Invasión de datos de otro mercado")
    initial_genome = ZERO.copy()
    # Stream separation across assets, even though jobs and data are disjoint.
    rng_seed = seed + 10000 * SYMBOLS.index(symbol)
    evo = evolve_population(
        lambda pop: evaluate_batch(pop, problem), initial_genome,
        f"pr43_{symbol}_native_H1", rng_seed,
        first_generation=FIRST_GENERATION, last_generation=LAST_GENERATION,
        ranking=strict_key,
    )
    if evo["evaluations"] != EVALUATIONS_PER_SEED:
        raise ValueError("Evolución incompleta, no publicar resultados parciales")
    champion = np.asarray(evo["genome"], dtype="<f8")
    measured = evaluate_local(champion, problem)
    if (champion.shape != (GENE_COUNT_LOCAL,) or
            not np.isclose(measured["score"], evo["champion"]["score"], rtol=0., atol=1e-10) or
            bool(measured["qualifies"]) != bool(evo["champion"]["qualifies"])):
        raise ValueError("No reproduce genoma/aptitud del vencedor")
    history = [history_row(row) for row in evo["generations"]]
    if [r["generation"] for r in history] != list(range(1, LAST_GENERATION + 1)):
        raise ValueError("Faltan generaciones IS")
    summary = result_summary(measured)
    if summary["qualified"] != strict_asset_pass(measured):
        raise ValueError("El filtro PF/DD original fue modificado")
    record = dict(
        protocol=PROTOCOL,
        source_pr41_run=42576852134,
        symbol=symbol, seed=seed, rng_seed=rng_seed,
        generations=LAST_GENERATION, population=POPULATION,
        genetic_evaluations=evo["evaluations"],
        genome_length=GENE_COUNT_LOCAL, first_genome_sha256=genome_sha(ZERO),
        champion_genome=champion.tolist(), champion_sha256=genome_sha(champion),
        champion_generation=int(evo["champion"]["generation"]),
        champion=summary, history=history,
        input_universe_sha256=problem["is_snapshot_sha256"],
        training_start="2026-01-01T00:00:00Z",
        training_end_exclusive="2026-07-01T00:00:00Z",
        signal_timeframes=["M5", "M15"], garch_per_asset=True,
        zero_initialized_each_training=True, cross_asset_training=False,
        shared_bankroll=False, old_pr41_genome_used=False,
        q3_accessed=False, oos_evaluated=False,
        promotion_approved=False,
        note="All 100 trials reuse the SAME H1 IS. More searches imply more selection bias, not OOS alpha.",
    )
    target = Path(output) / symbol.lower() / f"seed-{seed}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, record)
    print("PR43_TRAINING_COMPLETED " + json.dumps({
        "symbol": symbol, "seed": seed,
        "evaluations": evo["evaluations"],
        "generation": record["champion_generation"],
        "pf_q1": summary["pf_q1"], "pf_q2": summary["pf_q2"],
        "dd_h1": summary["dd_semester_pct"],
        "executed_hourly": summary["executed_hourly"],
        "qualified": summary["qualified"], "q3_accessed": False,
    }, allow_nan=False), flush=True)
    return record


def train_shard(*, symbol, shard, universe, manifest, golden=None,
                golden_manifest=None, output=OUTPUT):
    """One process, one symbol's data only; independent optimization per seed."""
    if symbol not in SYMBOLS:
        raise ValueError("Divisa no soportada")
    seeds = seeds_for_shard(shard)
    problem = native_problem(symbol, universe, manifest,
                             golden=golden, golden_manifest=golden_manifest)
    if problem["is_snapshot_sha256"] != SOURCE_SHA[symbol]:
        raise ValueError("Hash de fuente incompatible con PR41")
    completed = []
    for seed in seeds:
        item = train_seed(symbol, seed, problem, output=output)
        completed.append(dict(seed=seed, champion_sha256=item["champion_sha256"],
                              qualified=item["champion"]["qualified"]))
    assert len(completed) == SEEDS_PER_SHARD
    shard_report = dict(protocol=PROTOCOL, symbol=symbol, shard=shard,
                        seed_range=list(seeds), completions=completed,
                        genetic_evaluations=SEEDS_PER_SHARD * EVALUATIONS_PER_SEED,
                        q3_accessed=False, cross_asset_training=False)
    write_json(Path(output) / symbol.lower() / f"shard-{shard:02}.json", shard_report)
    return shard_report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True, choices=SYMBOLS)
    p.add_argument("--shard", required=True, type=int, choices=range(SHARDS))
    p.add_argument("--universe", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--golden", type=Path)
    p.add_argument("--golden-manifest", type=Path)
    p.add_argument("--output", type=Path, default=OUTPUT)
    train_shard(**vars(p.parse_args()))
