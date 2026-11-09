"""PR39: barrido A→B→C de banda GARCH y gate, 3 semillas, solo Q1/Q2.

A: bandas congeladas original/+7.5/+15/+22.5/+30% en mismo campeón.
B: 12 generaciones x128, PF geométrico × frecuencia real, con +15% fijo.
C: el campeón B con gates p85/p80/p75 y percentil local dependiente
   del régimen admitido, todos a banda +15%.

El genoma fundador es el campeón GARCH de PR38 (fase 3), recuperado
por artifact SHA. Ni etiquetas futuras, ni Q3, ni reajuste GARCH.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.run_m5_dynamic_upstream import load_problem, one_candidate, key
from scripts.run_m5_m15_target_experiments import (
    TOTAL_GENE_COUNT, score_and_mask,
)
from src.evolution_multiscale import POPULATION, evolve_population
from src.garch_sweetspot import (
    BAND_RELAXATIONS, GATE_MODES, TARGET_MIN_HOURLY,
    accepted, garch_band, local_regime_percentiles,
    pf_frequency_objective,
)
from src.upstream_filters import GENE_COUNT
from src.train import write_json


SEEDS=(42,43,44)
GENERATIONS=12
FIRST=1
LAST=12
MAIN_RELAXATION=.15
OUTPUT=Path("reports/m5-garch-sweetspot")
PROTOCOL="PR39_GARCH_sweetspot_IS_v1"


def sha_genome(genes):
    return sha256(np.asarray(genes,dtype="<f8").tobytes()).hexdigest()


def evaluate(genome, problem, relaxation, gate_mode):
    """GARCH actual/mediana pasada y Gate causal sobre exactamente el mismo IS."""
    if gate_mode not in GATE_MODES:
        raise ValueError("Gate no permitido")
    updated_score,_=score_and_mask(problem,genome,"garch_regime")
    selected,band,ratio=garch_band(
        genome,problem["data"]["volatility"],relaxation=relaxation
    )
    new_problem=dict(problem,scores=updated_score,
                     side=np.where(updated_score>=0.,1,-1).astype(np.int8))
    dynamic=None
    if gate_mode=="regime_dynamic":
        dynamic=local_regime_percentiles(ratio,band["applied_low"],band["applied_high"])
    result=one_candidate(
        np.asarray(genome,dtype="<f8")[:GENE_COUNT],
        new_problem,
        gate_mode=gate_mode if dynamic is None else "p85",
        extra_admissions=selected,
        custom_gate_percentiles=dynamic,
    )
    quality=float(result["score"])
    product_score,product=pf_frequency_objective(result)
    result.update(
        qualifies=accepted(result),passes_all=accepted(result),
        score=float(product_score),original_fitness=quality,
        pf_frequency_product=float(product),
    )
    return result,band


def record(result,band,*,name,seed,generation,genes=None):
    return dict(
        variant=name,seed=int(seed),generation=int(generation),
        band=band,
        pf_q1=result["quarter_pf"][0],pf_q2=result["quarter_pf"][1],
        q1_pnl=result["quarter_pnl_usd"][0],q2_pnl=result["quarter_pnl_usd"][1],
        q1_hourly=result["quarter_rate"][0],q2_hourly=result["quarter_rate"][1],
        executed_hourly_semester=result["semester_rate"],
        dd_q1_pct=result["quarter_dd"][0],dd_q2_pct=result["quarter_dd"][1],
        dd_semester_pct=result["semester_dd"],
        stops_q1_q2=result["quarter_stop"],
        stopped_semester=result["semester_stop"],
        proposals=result["eligible"],gate_accepted_pre_cap=result["gate_p85_count"],
        upstream_capacity_per_hour=result["up_funnel_capacity"],
        objective_score=result["score"],
        pf_frequency_product=result["pf_frequency_product"],
        original_fitness=result["original_fitness"],
        qualifies=result["qualifies"],
        genome_sha256=sha_genome(genes) if genes is not None else None,
    )


def verify_pr38_parent(path,problem,seed):
    """Prohíbe empezar de un fundador diferente o de la variante M1 retirada."""
    source=json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        source.get("protocol")!="PR38_evolved_targets_M5_M15_IS_v1"
        or source.get("seed")!=seed or source.get("phase")!=3
        or source.get("variant")!="garch_regime"
        or source.get("dataset_sha256")!=problem["is_snapshot_sha256"]
        or source.get("oos_accessed") is not False
        or source.get("signal_timeframes")!=["M5","M15"]
        or source.get("gate_mode")!="p85"
        or source.get("population")!=128
        or source.get("generations")!=15
        or source.get("evaluated")!=15*128
    ):
        raise ValueError("Campeón GARCH PR38 no corresponde a H1/M5+M15 sellado")
    genome=np.asarray(source["trained_genome"],dtype="<f8")
    if (genome.shape!=(TOTAL_GENE_COUNT,) or not np.isfinite(genome).all()
            or sha_genome(genome)!=source["genome_sha256"]):
        raise ValueError("Genoma PR38 alterado")
    prior,band=evaluate(genome,problem,0.0,"p85")
    champ=source["champion"]
    for field,key in (
        ("quarter_pf",["pf_q1","pf_q2"]),
        ("quarter_rate",None),
    ):
        if field=="quarter_pf":
            observed=prior[field]
            expected=[champ[k] for k in key]
            if not np.allclose(observed,expected,atol=1e-11,rtol=0):
                raise ValueError("PF del campeón PR38 no reproducido")
    if (not np.isclose(prior["semester_rate"],champ["executed_hourly"],atol=1e-11,rtol=0)
            or not np.isclose(prior["semester_dd"],champ["semester_dd"],atol=1e-11,rtol=0)
            or not np.isclose(prior["quarter_pnl_usd"][0],champ["pnl_q1"],atol=1e-11,rtol=0)
            or not np.isclose(prior["quarter_pnl_usd"][1],champ["pnl_q2"],atol=1e-11,rtol=0)):
        raise ValueError("Replay PR38 base no es reproducible")
    return genome,int(source["selected_generation"]),prior,band


def run(*,seed,founder_report,universe,manifest,golden,golden_manifest):
    if seed not in SEEDS:
        raise ValueError("Semilla no predeclarada")
    problem=load_problem(universe,manifest,golden,golden_manifest)
    founder,source_gen,baseline,base_band=verify_pr38_parent(
        founder_report,problem,seed
    )
    # Phase A: all widths on an UNCHANGED genome and fixed p85.
    a=[]
    for relax in BAND_RELAXATIONS:
        result,band=evaluate(founder,problem,relax,"p85")
        a.append(record(result,band,name=f"A_band_+{relax*100:g}pct_p85",
                        seed=seed,generation=source_gen,genes=founder))
    assert a[0]["genome_sha256"]==a[-1]["genome_sha256"]
    print("PR39_PHASE_A "+json.dumps({
        "seed":seed,"alternatives":[{
            "relax":x["band"]["relaxation"],"pf":[x["pf_q1"],x["pf_q2"]],
            "rate":x["executed_hourly_semester"],"dd":x["dd_semester_pct"]
        } for x in a],"oos_accessed":False,
    },allow_nan=False),flush=True)

    # Phase B: genome evolves after the +15% GARCH expansion, same p85.
    def evaluate_population(pop):
        results=[evaluate(row,problem,MAIN_RELAXATION,"p85")[0] for row in pop]
        return {k:np.asarray([r[k] for r in results]) for k in results[0]}
    evolution=evolve_population(
        evaluate_population,founder,"pr39_garch_pf_x_hourly",seed,
        first_generation=FIRST,last_generation=LAST,ranking=key
    )
    new=np.asarray(evolution["genome"],dtype="<f8")
    b_report,b_band=evaluate(new,problem,MAIN_RELAXATION,"p85")
    if not np.isclose(
        b_report["score"],evolution["champion"]["score"],atol=1e-11,rtol=0
    ):
        raise ValueError("Fitness B no reproduce al individuo ganado")
    b=record(b_report,b_band,name="B_genetic_PF_x_hourly_+15pct_p85",
             seed=seed,generation=int(evolution["champion"]["generation"]),
             genes=new)
    b["generations"]=GENERATIONS
    b["evaluations"]=evolution["evaluations"]
    b["generation_history"]=[
        dict(generation=x["generation"],score=x["score"],
             median_score=x["median_score"],candidate=x["candidate"])
        for x in evolution["generations"]
    ]
    print("PR39_PHASE_B "+json.dumps({
        "seed":seed,"selected_gen":b["generation"],
        "pf":[b["pf_q1"],b["pf_q2"]],
        "rate":b["executed_hourly_semester"],"dd":b["dd_semester_pct"],
        "pf_x_freq":b["pf_frequency_product"],"oos_accessed":False,
    },allow_nan=False),flush=True)

    # Phase C: gate sensitivity, SAME frozen Phase-B winner and SAME band.
    c=[]
    for mode in GATE_MODES:
        result,band=evaluate(new,problem,MAIN_RELAXATION,mode)
        c.append(record(result,band,name=f"C_band_+15pct_{mode}",
                        seed=seed,generation=b["generation"],genes=new))
    assert all(x["genome_sha256"]==b["genome_sha256"] for x in c)
    print("PR39_PHASE_C "+json.dumps({
        "seed":seed,"alternatives":[{
            "gate":x["variant"],"pf":[x["pf_q1"],x["pf_q2"]],
            "rate":x["executed_hourly_semester"],"dd":x["dd_semester_pct"]
        } for x in c],"oos_accessed":False,
    },allow_nan=False),flush=True)

    report=dict(
        protocol=PROTOCOL, seed=seed,
        initial_parent_generation=source_gen,
        initial_parent_genome_sha256=sha_genome(founder),
        new_genome_sha256=sha_genome(new),
        original_garch_sha256="7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504",
        universe_sha256=problem["is_snapshot_sha256"],
        signal_timeframes=["M5","M15"],
        q1_q2_in_sample=True,oos_accessed=False,
        population=POPULATION,genetic_generations=GENERATIONS,
        genetic_evaluations=POPULATION*GENERATIONS,
        gate_baseline="p85",primary_band_relaxation=.15,
        gate_variants=list(GATE_MODES),band_variants=list(BAND_RELAXATIONS),
        acceptance=dict(min_pf_each_quarter=1.20,min_executed_semester_per_hour=.50,
                        max_semester_dd_strict=12.,all_seeds_required=True),
        baseline=record(baseline,base_band,name="PR38_original_GARCH_p85",
                        seed=seed,generation=source_gen,genes=founder),
        phase_A=a,phase_B=b,phase_C=c,
        no_model_promotion=True,
    )
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_json(OUTPUT/f"seed-{seed}.json",report)
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed",type=int,choices=SEEDS,required=True)
    for name in ("founder_report","universe","manifest","golden","golden_manifest"):
        ap.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=ap.parse_args()
    run(seed=a.seed,founder_report=a.founder_report,
        universe=a.universe,manifest=a.manifest,golden=a.golden,
        golden_manifest=a.golden_manifest)
