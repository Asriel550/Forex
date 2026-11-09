"""PR36: neuroevolución exclusiva de cabeza upstream + replay p85 semestral.

Todas las decisiones usan el universo completo de M5 SELLADO contra PR35;
el modelo Gen375/43 fija la dirección y el Gate p85 se reproduce sin cambios.
Q1/Q2 se usan como IS; Q3 jamás se carga. No se promociona sin cumplir los
objetivos operativos sobre riesgos reales y con una cuenta semestral continua.
"""
import argparse
import json
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd

from scripts.build_m5_full_is_universe import load_universe
from scripts.run_m5_adaptive_gate import GEN354_SHA256
from scripts.run_m5_temporal_evolution import causal_reasons
from scripts.run_m5_conviction_evolution import OUTPUT as EVOLUTION_OUTPUT
from scripts.run_m5_semester_replay import Q2_START, END, START, PERIODS, assert_isolated
from scripts.run_m5_temporal_evolution import frozen_gen375
from scripts.run_m5_conviction_evolution import setup
from scripts.verify_m5_durable_snapshot import verify as verify_golden
from src.evolution_engine import assess_temporal_multiregime_fitness
from src.evolution_multiscale import direction_scores, evolve_population, POPULATION
from src.evolution_lineage import MARKET
from src.model_training import directional_tanh_loss
from src.replay_engine import apply_hourly_cap, replay_contracts
from src.upstream_filters import GENE_COUNT, dynamic_upstream
from src.train import write_json


OUTPUT = Path("reports/m5-dynamic-upstream")
SEEDS = (42, 43, 44)
PERCENTILE = .85
MIN_FREQUENCY = 1.20
MIN_PF = 1.20
MAX_DD_PCT = 15.


def load_problem(universe, manifest, golden, golden_manifest):
    assert_isolated(PERIODS)
    verify_golden(golden, golden_manifest)
    data, info = load_universe(universe, manifest, golden, golden_manifest)
    reference, founder, _, risk, _, _, _, _, _ = setup()
    archive = json.loads((EVOLUTION_OUTPUT / "evolution.json").read_text(encoding="utf-8"))
    parent = frozen_gen375(GEN354_SHA256, archive)
    score = direction_scores(data["direction_inputs"], [parent])[0]
    side = np.where(score >= 0., 1, -1).astype(np.int8)
    times = pd.to_datetime(data["row_times_ns"], utc=True, unit="ns")
    hours = info["months_observed_hours"]
    q1_hours = sum(float(hours[p]) for p in PERIODS[:3])
    q2_hours = sum(float(hours[p]) for p in PERIODS[3:])
    baseline, _ = dynamic_upstream(
        data["entry"], data["margin"], data["micro"], data["volatility"],
        np.zeros(GENE_COUNT), float(founder.genome[-1])
    )
    if not np.array_equal(baseline, data["legacy_mask"]):
        raise ValueError("La cabeza neutra debe reproducir el upstream legacy")
    return dict(
        data=data, info=info, times=times, scores=score, side=side,
        founder=founder, risk=risk, account=founder.account,
        cap=reference["gate"]["hourly_cap"], legacy_threshold=float(founder.genome[-1]),
        hours={"Q1":q1_hours,"Q2":q2_hours,"semester":q1_hours+q2_hours},
        is_snapshot_sha256=info["universe_sha256"],
    )


def p85_reasons(rows, scores):
    """Gate exacto, ventanas y settlements causales; ningún filtro vol externo."""
    return causal_reasons(rows, scores, PERCENTILE)


def one_candidate(genes, problem, *, include_traces=False):
    data = problem["data"]
    upstream, thresholds = dynamic_upstream(
        data["entry"], data["margin"], data["micro"], data["volatility"],
        genes, problem["legacy_threshold"]
    )
    idx = np.flatnonzero(upstream)
    if len(idx) <= 500:
        return {"qualifies":False, "score":-1e6, "worst_quarter_return":-1e6,
                "quarter_return_std":1e6, "min_gate_only_frequency":0.,
                "eligible":int(len(idx)), "return_per_admission":-1e6, "quarter_pnl_usd":[0.,0.],
                "quarter_pf":[None,None], "quarter_rate":[0.,0.],
                "quarter_gate_rate":[0.,0.], "quarter_dd":[0.,0.],
                "semester_pnl_usd":0., "semester_pf":None,
                "semester_rate":0., "semester_dd":0.,
                "quarter_pass":[False,False], "passes_all":False,
                "threshold_min":float(thresholds.min()), "threshold_max":float(thresholds.max())}
    rows = pd.DataFrame({"outcome":data["outcome"][idx]}, index=problem["times"][idx])
    scores = problem["scores"][idx]
    sides = problem["side"][idx]
    reasons = p85_reasons(rows, scores)
    q1 = rows.index < Q2_START
    quarters, traces = [], {}
    for name, mask in (("Q1", q1), ("Q2", ~q1), ("semester", np.ones(len(rows), bool))):
        subset = rows.loc[mask]
        if subset.empty:
            return {"qualifies":False, "score":-1e6, "worst_quarter_return":-1e6,
                    "quarter_return_std":1e6, "min_gate_only_frequency":0.,
                    "eligible":int(len(idx)), "return_per_admission":-1e6, "quarter_pnl_usd":[0.,0.],
                    "quarter_pf":[None,None], "quarter_rate":[0.,0.],
                    "quarter_gate_rate":[0.,0.], "quarter_dd":[0.,0.],
                    "semester_pnl_usd":0., "semester_pf":None, "semester_rate":0.,
                    "semester_dd":0., "quarter_pass":[False,False],
                    "passes_all":False, "threshold_min":float(thresholds.min()),
                    "threshold_max":float(thresholds.max())}
        observed = problem["hours"][name]
        result, trace = replay_contracts(
            subset, sides[mask], problem["account"], problem["risk"],
            observed_hours=observed, hourly_cap=problem["cap"],
            candidate_reasons=reasons[mask],
        )
        if include_traces:
            traces[name] = trace
        operational = result["operational"]
        candidates = apply_hourly_cap(
            subset.index, np.full(len(subset), "accepted"), problem["cap"]
        ) == "accepted"
        if name != "semester":
            quarters.append(dict(
                **operational, admissions=len(subset), observed_hours=observed,
                gate_only_pnl_usd=result["gate_only_at_expiry"]["pnl_usd"],
                gate_only_contracts_per_observed_hour=(
                    result["gate_only_at_expiry"]["contracts_per_observed_hour"]
                ),
                capacity_contracts_per_observed_hour=float(candidates.sum() / observed),
            ))
        else:
            semester = operational

    selected_vol = data["volatility"][idx]
    high_cut = float(np.quantile(selected_vol[q1], 2 / 3))
    loss = directional_tanh_loss(
        scores, rows["outcome"].to_numpy(), selected_vol > high_cut
    )
    fit = assess_temporal_multiregime_fitness(
        quarters, stake=problem["account"].stake, direction_loss=loss["total"]
    )
    pf_ok = [
        q["profit_factor_status"] == "infinite_no_losses" or
        (q["profit_factor"] is not None and q["profit_factor"] > MIN_PF)
        for q in quarters
    ]
    quarter_ok = [
        pf_ok[i] and quarters[i]["max_drawdown_pct"] < MAX_DD_PCT
        and quarters[i]["pnl_usd"] > 0 and quarters[i]["account_stop"] is None
        for i in range(2)
    ]
    semester_ok = (
        semester["contracts_per_observed_hour"] >= MIN_FREQUENCY
        and semester["max_drawdown_pct"] < MAX_DD_PCT
        and semester["account_stop"] is None
    )
    qualifies = bool(all(quarter_ok) and semester_ok)
    report = dict(
        qualifies=qualifies,
        passes_all=qualifies,
        quarter_pass=quarter_ok,
        score=fit["score"], worst_quarter_return=fit["worst_quarter_return"],
        quarter_return_std=fit["quarter_return_std"],
        min_gate_only_frequency=fit["min_gate_only_frequency"],
        eligible=int(len(idx)),
        return_per_admission=fit["return_per_admission"],
        quarter_pnl_usd=[q["pnl_usd"] for q in quarters],
        quarter_pf=[q["profit_factor"] for q in quarters],
        quarter_rate=[q["contracts_per_observed_hour"] for q in quarters],
        quarter_gate_rate=[q["gate_only_contracts_per_observed_hour"] for q in quarters],
        quarter_dd=[q["max_drawdown_pct"] for q in quarters],
        quarter_stop=[q["account_stop"] is not None for q in quarters],
        semester_pnl_usd=semester["pnl_usd"],
        semester_pf=semester["profit_factor"],
        semester_rate=semester["contracts_per_observed_hour"],
        semester_dd=semester["max_drawdown_pct"],
        semester_stop=semester["account_stop"] is not None,
        threshold_min=float(thresholds.min()),
        threshold_max=float(thresholds.max()),
        gate_p85_count=int((reasons=="accepted").sum()),
        gate_warmup=int((reasons=="conviction_warmup").sum()),
        gate_conviction_rejected=int((reasons=="conviction").sum()),
        fitness_loss=loss["total"],
        up_funnel_capacity=float(sum(
            apply_hourly_cap(rows.loc[mask].index, np.full(int(mask.sum()), "accepted"),
                             problem["cap"]).tolist().count("accepted")
            for mask in (q1,~q1)
        ) / problem["hours"]["semester"]),
    )
    if include_traces:
        return report, traces
    return report


def key(f):
    """Fitness elegible primero; luego score temporal, peor trimestre y densidad."""
    return (f["qualifies"], f["score"], f["worst_quarter_return"],
            -f["quarter_return_std"], f["min_gate_only_frequency"])


def run(*, universe, manifest, golden, golden_manifest, generations=1, seeds=(43,)):
    if (type(generations) is not int or generations < 1 or generations > 21
            or not seeds or not set(seeds).issubset(SEEDS)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Gen 376+ requiere generaciones/semillas predeclaradas")
    problem = load_problem(universe, manifest, golden, golden_manifest)
    founder_genes = np.zeros(GENE_COUNT, dtype=float)
    baseline, _ = one_candidate(founder_genes, problem, include_traces=True)
    experiments = []
    for seed in seeds:
        def evaluate(population):
            records = [one_candidate(candidate, problem) for candidate in population]
            return {name:np.asarray([r[name] for r in records]) for name in records[0]}
        evolution = evolve_population(
            evaluate, founder_genes, "m5_dynamic_upstream", seed,
            first_generation=376, last_generation=375 + generations, ranking=key
        )
        winner, traces = one_candidate(evolution["genome"], problem, include_traces=True)
        if abs(winner["score"] - evolution["champion"]["score"]) > 1e-10:
            raise ValueError("La evolución no reproduce la puntuación de su ganador")
        experiments.append(dict(
            seed=seed, genes=evolution["genome"], genome_sha256=sha256(
                np.asarray(evolution["genome"], dtype="<f8").tobytes()).hexdigest(),
            founder=evolution["founder"], champion=winner,
            selection=evolution["champion"], generations=evolution["generations"],
            evaluations=evolution["evaluations"],
        ))
        print("UPSTREAM_GEN",json.dumps({
            "seed":seed,"qualifies":winner["qualifies"],"score":winner["score"],
            "q_pf":winner["quarter_pf"],"q_rate":winner["quarter_rate"],
            "s_rate":winner["semester_rate"],"q_dd":winner["quarter_dd"]
        },allow_nan=False),flush=True)
    report = dict(
        protocol="PR36_causal_upstream_IS_v1",
        original_golden_sha256=verify_golden(golden,golden_manifest)["snapshot_sha256"],
        expanded_is_sha256=problem["is_snapshot_sha256"],
        q1_q2_in_sample=True, oos_accessed=False,
        percentile=.85, neural_direction="frozen_Gen375_seed43_p80",
        risk="original hourly cap and account stop, no new volatility block",
        baseline=baseline, experiments=experiments,
        all_targets_met=any(r["champion"]["qualifies"] for r in experiments),
        primary_head_genome_width=GENE_COUNT, population=POPULATION,
        first_generation=376, last_generation=375+generations,
        seeds=list(seeds), evaluations=sum(r["evaluations"] for r in experiments),
    )
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_json(OUTPUT / "replay.json",report)
    print("PR36_REPLAY",json.dumps({
        "baseline":{k:baseline[k] for k in ("quarter_pf","quarter_rate","quarter_dd",
                     "semester_rate","semester_pnl_usd")},
        "champions":[{"seed":r["seed"],"pf":r["champion"]["quarter_pf"],
                      "frequency":r["champion"]["semester_rate"],
                      "dd":r["champion"]["quarter_dd"],
                      "passes":r["champion"]["qualifies"]} for r in experiments],
        "oos_accessed":False,
    },allow_nan=False),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("universe","manifest","golden","golden_manifest"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--generations",type=int,default=1)
    p.add_argument("--seeds",nargs="+",type=int,default=[43])
    a=p.parse_args()
    run(universe=a.universe,manifest=a.manifest,golden=a.golden,
        golden_manifest=a.golden_manifest,generations=a.generations,seeds=tuple(a.seeds))
