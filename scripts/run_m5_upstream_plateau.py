"""Gen398–497: continuación desde los campeones congelados Gen376–397.

100 generaciones ADICIONALES, población 128, semillas 42/43/44. Los
candidatos usan solo señales M5+M15, el Gate p85 causal y Q1/Q2 IS.
Esto es investigación de saturación, no OOS ni una promoción automática.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.run_m5_dynamic_upstream import key, load_problem, one_candidate
from scripts.verify_m5_durable_snapshot import SNAPSHOT_SHA256
from src.evolution_multiscale import POPULATION, evolve_population
from src.upstream_filters import GENE_COUNT
from src.train import write_json

FIRST_GENERATION = 398
LAST_GENERATION = 497
GENERATIONS = LAST_GENERATION - FIRST_GENERATION + 1
SEEDS = (42, 43, 44)
REFERENCE_RUN = 42544173112
PARENT_GENOME_SHA = {
    42: "b179c0ab3ae44af489ee5471b5cdfd958f9e04a0e829348716d2d5f59b21b6ad",
    43: "3e28eb796a9f95ab5333c3b177d97d0d497479d5c56953635220521834968a94",
    44: "ec497b444ba21d7a14b899c3265d804f5a6a1cdc1e7146ad9b18a5b8c1a32a8a",
}
OUTPUT = Path("reports/m5-upstream-plateau")


def genome_sha(genome):
    return sha256(np.asarray(genome, dtype="<f8").tobytes()).hexdigest()


def load_incumbent(path, *, seed, problem):
    """Verifica linaje, dataset, parámetros y reproducción del fitness original."""
    if seed not in SEEDS:
        raise ValueError("Semilla ajena al experimento original")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        report.get("protocol") != "PR36_causal_upstream_IS_v1"
        or report.get("seeds") != [seed]
        or report.get("population") != POPULATION
        or report.get("first_generation") != 376
        or report.get("last_generation") != 397
        or report.get("evaluations") != POPULATION * 22
        or report.get("expanded_is_sha256") != problem["is_snapshot_sha256"]
        or report.get("original_golden_sha256") != SNAPSHOT_SHA256
        or report.get("oos_accessed") is not False
        or report.get("percentile") != .85
        or len(report.get("experiments", [])) != 1
    ):
        raise ValueError("Reporte fundador incorrecto: no se continuará una rama distinta")
    item = report["experiments"][0]
    genes = np.asarray(item["genes"], dtype="<f8")
    if (
        item.get("seed") != seed
        or item.get("evaluations") != POPULATION * 22
        or genes.shape != (GENE_COUNT,)
        or not np.isfinite(genes).all()
        or genome_sha(genes) != PARENT_GENOME_SHA[seed]
        or item.get("genome_sha256") != PARENT_GENOME_SHA[seed]
        or not 376 <= item["selection"]["generation"] <= 397
        or len(item.get("generations", [])) != 22
    ):
        raise ValueError("El campeón Gen397 original no coincide con su firma sellada")

    measured = one_candidate(genes, problem)
    stored = item["champion"]
    scalar_fields = ("score", "semester_rate", "semester_pnl_usd", "semester_dd")
    vector_fields = ("quarter_pf", "quarter_dd", "quarter_rate", "quarter_pnl_usd")
    if (any(not np.isclose(measured[k], stored[k], atol=1e-10, rtol=0)
            for k in scalar_fields)
            or any(not np.allclose(measured[k], stored[k], atol=1e-10, rtol=0)
                   for k in vector_fields)
            or key(measured) != key(stored)):
        raise ValueError("El campeón anterior no reproduce sus métricas Q1/Q2 con el mismo IS")
    return genes, measured, int(item["selection"]["generation"])


def trajectory(history, initial):
    """Cambios en la mejor aptitud, y meseta durante las últimas 20 generaciones."""
    if len(history) != GENERATIONS or [
        r["generation"] for r in history
    ] != list(range(FIRST_GENERATION, LAST_GENERATION + 1)):
        raise ValueError("Faltan generaciones en la evolución de saturación")
    best = dict(initial)
    improved_at = []
    cumulative_scores = []
    for candidate in history:
        if key(candidate) > key(best):
            best = candidate
            improved_at.append(candidate["generation"])
        cumulative_scores.append(float(best["score"]))
    return dict(
        improving_generations=improved_at,
        improvements_count=len(improved_at),
        last_improvement_generation=improved_at[-1] if improved_at else None,
        generations_since_last_improvement=(
            LAST_GENERATION - improved_at[-1] if improved_at else GENERATIONS
        ),
        best_fitness_first_20=float(max(cumulative_scores[:20])),
        best_fitness_last_20=float(max(cumulative_scores[-20:])),
        fitness_gain_last_20=float(cumulative_scores[-1] - cumulative_scores[-21]),
        best_fitness_by_generation=cumulative_scores,
        median_fitness_by_generation=[float(r["median_score"]) for r in history],
    )


def run(*, seed, parent_report, universe, manifest, golden, golden_manifest):
    if seed not in SEEDS:
        raise ValueError("Solo semillas 42,43,44")
    problem = load_problem(universe, manifest, golden, golden_manifest)
    founder, old, old_generation = load_incumbent(parent_report, seed=seed, problem=problem)

    def evaluate(population):
        records = [one_candidate(genes, problem) for genes in population]
        return {field: np.asarray([record[field] for record in records])
                for field in records[0]}

    evolution = evolve_population(
        evaluate, founder, "m5_m15_upstream_100gen", seed,
        first_generation=FIRST_GENERATION, last_generation=LAST_GENERATION, ranking=key,
    )
    new_genome = np.asarray(evolution["genome"], dtype="<f8")
    latest = one_candidate(new_genome, problem)
    if (abs(latest["score"] - evolution["champion"]["score"]) > 1e-10
            or not np.allclose(latest["quarter_pf"], evolution["champion"]["quarter_pf"],
                               atol=1e-10, rtol=0)):
        raise ValueError("La mejor nueva generación no reproduce métricas de replay")

    better_rank = key(latest) > key(old)
    better_score = bool(latest["score"] > old["score"] + 1e-12)
    statistical = trajectory(evolution["generations"], old)
    financial = dict(
        score_delta=float(latest["score"] - old["score"]),
        q1_pf_delta=float(latest["quarter_pf"][0] - old["quarter_pf"][0]),
        q2_pf_delta=float(latest["quarter_pf"][1] - old["quarter_pf"][1]),
        semester_executed_hourly_delta=float(
            latest["semester_rate"] - old["semester_rate"]
        ),
        q1_dd_pct_delta=float(latest["quarter_dd"][0] - old["quarter_dd"][0]),
        q2_dd_pct_delta=float(latest["quarter_dd"][1] - old["quarter_dd"][1]),
        better_fitness_rank=bool(better_rank),
        better_score=better_score,
        financial_acceptance=bool(latest["qualifies"]),
    )
    report = dict(
        protocol="PR37_upstream_IS_100_more_generations_v1",
        base_run_id=REFERENCE_RUN,
        seed=seed, population=POPULATION, generations=GENERATIONS,
        first_generation=FIRST_GENERATION, last_generation=LAST_GENERATION,
        evaluations=POPULATION * GENERATIONS,
        signal_timeframes=["M5", "M15"], gate_percentile=.85,
        q1_q2_in_sample=True, oos_accessed=False,
        dataset_sha256=problem["is_snapshot_sha256"],
        original_garch_sha256=SNAPSHOT_SHA256,
        base_best_generation=old_generation,
        base_genome_sha256=PARENT_GENOME_SHA[seed],
        current_best_generation=int(evolution["champion"]["generation"]),
        current_genome_sha256=genome_sha(new_genome),
        base=old, current=latest,
        comparison=financial, plateau=statistical,
        genetic_generations=evolution["generations"],
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT / f"gen398-497-seed-{seed}.json"
    if dest.exists():
        raise ValueError("No sobrescribir resultados de un experimento sellado")
    write_json(dest, report)
    print("GEN398_497_PLATEAU " + json.dumps({
        "seed":seed, "base_best_generation":old_generation,
        "new_best_generation":evolution["champion"]["generation"],
        "score_before":old["score"], "score_after":latest["score"],
        "delta":financial["score_delta"], "better":better_rank,
        "q1_pf":latest["quarter_pf"][0], "q2_pf":latest["quarter_pf"][1],
        "semester_hourly":latest["semester_rate"],
        "q1_q2_dd":latest["quarter_dd"],
        "improvements":statistical["improvements_count"],
        "last_gain_generation":statistical["last_improvement_generation"],
        "accepted":bool(latest["qualifies"]), "oos_accessed":False,
    }, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    for field in ("parent_report","universe","manifest","golden","golden_manifest"):
        parser.add_argument("--"+field.replace("_", "-"), type=Path, required=True)
    opts = parser.parse_args()
    run(seed=opts.seed, parent_report=opts.parent_report,
        universe=opts.universe, manifest=opts.manifest, golden=opts.golden,
        golden_manifest=opts.golden_manifest)
