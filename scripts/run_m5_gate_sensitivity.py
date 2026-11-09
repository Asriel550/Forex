"""Fase 1 PR38: sensibilidad causal del gate p85/p80/p75 y volatilidad M5.

Cada semilla parte de su genoma Gen376–397 ORIGINAL, verificado por hash.
Se comparan las mismas propuestas antes de cualquier mutación y 2 generaciones
smoke de 128 individuos por modo. El cambio de percentil nunca ve Q3.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.run_m5_dynamic_upstream import load_problem, one_candidate, key
from scripts.run_m5_upstream_plateau import load_incumbent
from src.conviction_regime_gate import MODES
from src.evolution_multiscale import POPULATION, evolve_population
from src.train import write_json

OUTPUT = Path("reports/m5-alpha-research/phase1")
SMOKE_GENERATIONS = 2
SEEDS = (42, 43, 44)


def diagnostic(result):
    return dict(
        pf_q1=result["quarter_pf"][0], pf_q2=result["quarter_pf"][1],
        pnl_q1=result["quarter_pnl_usd"][0], pnl_q2=result["quarter_pnl_usd"][1],
        rate_q1=result["quarter_rate"][0], rate_q2=result["quarter_rate"][1],
        semester_executed_hourly=result["semester_rate"],
        semester_drawdown_pct=result["semester_dd"],
        dd_q1=result["quarter_dd"][0], dd_q2=result["quarter_dd"][1],
        gate_accepted_pre_cap=result["gate_p85_count"],
        upstream_proposals=result["eligible"], upstream_capacity=result["up_funnel_capacity"],
        score=result["score"], qualifies=result["qualifies"],
        stop_q1=result["quarter_stop"][0], stop_q2=result["quarter_stop"][1],
        stop_semester=result["semester_stop"],
    )


def run(*, seed, founder_report, universe, manifest, golden, golden_manifest):
    if seed not in SEEDS:
        raise ValueError("Solo semillas 42/43/44")
    problem = load_problem(universe, manifest, golden, golden_manifest)
    genome, reference, base_gen = load_incumbent(
        founder_report, seed=seed, problem=problem
    )
    result_modes = []
    for mode in MODES:
        fixed = one_candidate(genome, problem, gate_mode=mode)

        def evaluate(population):
            rec = [one_candidate(g, problem, gate_mode=mode) for g in population]
            return {field: np.asarray([r[field] for r in rec]) for field in rec[0]}

        evo = evolve_population(
            evaluate, genome, f"pr38_phase1_{mode}", seed,
            first_generation=1, last_generation=SMOKE_GENERATIONS, ranking=key,
        )
        champion = one_candidate(evo["genome"], problem, gate_mode=mode)
        if abs(champion["score"] - evo["champion"]["score"]) > 1e-10:
            raise ValueError("Fitness del smoke no se reproduce")
        result_modes.append(dict(
            gate_mode=mode, parent=diagnostic(fixed), champion=diagnostic(champion),
            best_smoke_generation=int(evo["champion"]["generation"]),
            genome_sha256=sha256(np.asarray(evo["genome"], dtype="<f8").tobytes()).hexdigest(),
            founder_genome_sha256=sha256(genome.astype("<f8").tobytes()).hexdigest(),
            evaluations=evo["evaluations"],
            history=[dict(generation=x["generation"],score=x["score"],
                          median_score=x["median_score"]) for x in evo["generations"]],
        ))
    fixed_p85 = result_modes[0]["parent"]
    if (not np.isclose(fixed_p85["score"], reference["score"], atol=1e-10, rtol=0)
            or not np.allclose(
                [fixed_p85["pf_q1"],fixed_p85["pf_q2"]],
                reference["quarter_pf"], atol=1e-10, rtol=0
            )):
        raise ValueError("El control p85 ha cambiado; no se compara ninguna hipótesis")
    p75 = next(x for x in result_modes if x["gate_mode"] == "p75")
    p75_f = p75["parent"]
    p75_more_gate = p75_f["gate_accepted_pre_cap"] > fixed_p85["gate_accepted_pre_cap"]
    severe_pf = min(p75_f["pf_q1"], p75_f["pf_q2"]) < .85
    success = any(
        min(v["champion"]["pf_q1"],v["champion"]["pf_q2"]) > 1.0
        and v["champion"]["semester_executed_hourly"] >= 1.0
        for v in result_modes
    )
    report=dict(
        protocol="PR38_phase1_causal_gate_sensitivity_v1",
        seed=seed, base_gen=int(base_gen), baseline_genome_sha256=p75["founder_genome_sha256"],
        modes=result_modes,
        p75_more_pre_cap_admissions=p75_more_gate,
        p75_severe_pf_failure=bool(severe_pf),
        phase1_success=bool(success),
        phase2_required=not success,
        dataset_sha256=problem["is_snapshot_sha256"], signal_timeframes=["M5","M15"],
        q1_q2_in_sample=True, oos_accessed=False,
        population=POPULATION, smoke_generations=SMOKE_GENERATIONS,
        evaluations=POPULATION * SMOKE_GENERATIONS * len(MODES),
    )
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_json(OUTPUT / f"phase1-seed-{seed}.json",report)
    print("PHASE1_GATE "+json.dumps({
        "seed":seed, "baseline":fixed_p85,
        "modes":[{"mode":v["gate_mode"],"pf_q1":v["champion"]["pf_q1"],
                  "pf_q2":v["champion"]["pf_q2"],
                  "executed_h":v["champion"]["semester_executed_hourly"],
                  "dd_semester":v["champion"]["semester_drawdown_pct"],
                  "qualifies":v["champion"]["qualifies"]} for v in result_modes],
        "phase2_required":report["phase2_required"],
        "oos_accessed":False,
    },allow_nan=False),flush=True)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    for field in ("founder_report","universe","manifest","golden","golden_manifest"):
        parser.add_argument("--"+field.replace("_","-"),type=Path,required=True)
    options=parser.parse_args()
    run(seed=options.seed,founder_report=options.founder_report,
        universe=options.universe,manifest=options.manifest,
        golden=options.golden,golden_manifest=options.golden_manifest)
