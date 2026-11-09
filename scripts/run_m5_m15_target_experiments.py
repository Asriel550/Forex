"""Fases 2–3: targets ATR / R:R / loss asimétrica y cabezas M15/GARCH.

Se aprende por NEUROEVOLUCIÓN, no backprop: no hay gradientes en el motor.
Las etiquetas de futuro solo participan en la pérdida sobre Q1/Q2 IS.
El contrato de evaluación sigue liquidando a +80%/-100% después de M5,
no pretende estimar fills intrabar R:R. Q3 nunca se carga.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_m5_m15_alpha_targets import load_targets
from scripts.run_m5_dynamic_upstream import load_problem, one_candidate, key as base_key
from scripts.run_m5_upstream_plateau import load_incumbent
from src.evolution_multiscale import evolve_population, POPULATION
from src.upstream_filters import GENE_COUNT, SIGNAL_NAMES
from src.train import write_json

SEEDS=(42,43,44)
PHASE2=("atr_expansion","rr15","asymmetric_fp")
PHASE3=("m15_attention","garch_regime")
VARIANTS=PHASE2+PHASE3
DIR_GENE_COUNT=13
TOTAL_GENE_COUNT=GENE_COUNT+DIR_GENE_COUNT
FIRST=1
PHASE2_GENERATIONS=16
PHASE3_GENERATIONS=15
MIN_EXECUTED_HOUR=1.0
PF_GOAL=1.20
DD_GOAL=15.
OUTPUT=Path("reports/m5-alpha-research")

def _sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value,-20.,20.)))


def score_and_mask(problem, genome, variant):
    """Cabeza residual tanh con dos variantes estructurales M15 / GARCH."""
    genes=np.asarray(genome,dtype=float)
    if genes.shape!=(TOTAL_GENE_COUNT,) or not np.isfinite(genes).all():
        raise ValueError("Dimensiones o pesos genéticos inválidos")
    head=genes[GENE_COUNT:]
    tf=problem["data"]["timeframe"]
    bias=head[8]
    gain=4.0*np.tanh(head[9])
    if variant=="m15_attention":
        a=_sigmoid(head[10]*tf[:,4]+head[11]*tf[:,7]+head[12])
        f=tf.copy()
        # Atención simple sobre M15 recién cerrada y la M15 previa.
        f[:,4]=(a*tf[:,4]+(1-a)*tf[:,7])
        features=f
    else:
        features=tf
    updated=problem["scores"]+gain*np.tanh(features@head[:8]+bias)
    submask=None
    if variant=="garch_regime":
        sigma=problem["data"]["volatility"]
        previous=pd.Series(np.maximum(sigma,1e-12)).shift(1).rolling(
            48,min_periods=24
        ).median().to_numpy()
        ratio=sigma/np.where(np.isfinite(previous),previous,np.maximum(sigma,1e-12))
        lower=.35+_sigmoid(head[10])
        upper=1.2+2.*_sigmoid(head[11])
        # El rango se calcula a partir del GARCH observado al cerrar M5.
        submask=(ratio>=lower)&(ratio<=upper)
    return updated,submask


def asymmetric_target_loss(scores, label, valid, low_momentum):
    """Desalienta fuertemente falsos positivos/convicción en poco momentum."""
    scores=np.asarray(scores,dtype=float)
    label=np.asarray(label,dtype=np.int8)
    valid=np.asarray(valid,dtype=bool)
    low=np.asarray(low_momentum,dtype=bool)
    if (scores.shape!=label.shape or valid.shape!=label.shape
            or low.shape!=label.shape or not np.isfinite(scores).all()
            or not np.isin(label,[-1,0,1]).all()):
        raise ValueError("Target direccional inválido")
    usable=valid
    if not usable.any():
        return 1e3
    t=np.tanh(scores[usable])
    y=label[usable]
    silent=low[usable]
    nonzero=y!=0
    # Logistic target loss on settled directional targets.
    direction=float(np.logaddexp(0.,-2*y[nonzero]*t[nonzero]).mean()) if nonzero.any() else 0.
    wrong=nonzero & (np.sign(t)!=y)
    high_conviction_false_positive=float(
        (wrong.astype(float)*(t*t)*(1.+2.5*silent.astype(float))).mean()
    )
    # Una entrada confiada sin movimiento compatible con ATR es falso positivo.
    no_expansion=float(((y==0).astype(float)*(t*t)*(1.+2.5*silent.astype(float))).mean())
    return direction+2.0*high_conviction_false_positive+0.7*no_expansion


def accepted_by_pr38(report):
    pf=report["quarter_pf"]
    return bool(
        all(v is not None and v>PF_GOAL for v in pf)
        and all(p>0. for p in report["quarter_pnl_usd"])
        and all(d<DD_GOAL for d in report["quarter_dd"])
        and report["semester_dd"]<DD_GOAL
        and report["semester_rate"]>=MIN_EXECUTED_HOUR
        and not any(report["quarter_stop"])
        and not report["semester_stop"]
    )


def one_variant_candidate(genome, problem, targets, variant, *, gate_mode="p85"):
    genes=np.asarray(genome,dtype="<f8")
    score,mask=score_and_mask(problem,genes,variant)
    alt=dict(problem,scores=score,side=np.where(score>=0.,1,-1).astype(np.int8))
    report=one_candidate(genes[:GENE_COUNT],alt,gate_mode=gate_mode,extra_admissions=mask)
    if report["eligible"]<=500:
        report.update(
            base_score=float(report["score"]), target_loss=1e3,
            target_positive_rate=0., added_direction_flips=0,
        )
        return report
    from src.upstream_filters import dynamic_upstream
    upstream,_=dynamic_upstream(
        problem["data"]["entry"],problem["data"]["margin"],
        problem["data"]["timeframe"],problem["data"]["volatility"],
        genes[:GENE_COUNT],problem["legacy_threshold"]
    )
    if mask is not None:
        upstream &= mask
    indices=np.flatnonzero(upstream)
    if variant=="atr_expansion":
        label=targets["atr_expansion_label"]
        valid=targets["valid_label"]
    elif variant=="rr15":
        label=targets["rr15_label"]
        valid=targets["valid_label"]
    else:
        label=problem["data"]["outcome"]
        valid=np.ones(len(label),dtype=bool)
    loss=asymmetric_target_loss(
        score[indices],label[indices],valid[indices],targets["low_momentum"][indices]
    )
    # Conservar el fitness operativo temporal, cambiando solo el objetivo
    # regularizador de dirección; no confundir con PF del target R:R.
    old_score=float(report["score"])
    penalty_weight=.35 if variant in PHASE2 else .25
    report["score"]=float(old_score-penalty_weight*loss)
    report["base_score"]=old_score
    report["target_loss"]=float(loss)
    report["target_positive_rate"]=float(
        np.mean((label[indices]!=0)&valid[indices])
    )
    report["added_direction_flips"]=int(np.count_nonzero(
        np.where(score[indices]>=0,1,-1) != problem["side"][indices]
    ))
    report["qualifies"]=accepted_by_pr38(report)
    report["passes_all"]=report["qualifies"]
    return report


def compact_stats(record):
    return dict(
        score=float(record["score"]),operational_score=float(record["base_score"]),
        target_loss=float(record["target_loss"]),
        proposals=int(record["eligible"]),direction_flips=int(record["added_direction_flips"]),
        pf_q1=record["quarter_pf"][0],pf_q2=record["quarter_pf"][1],
        pnl_q1=record["quarter_pnl_usd"][0],pnl_q2=record["quarter_pnl_usd"][1],
        dd_q1=record["quarter_dd"][0],dd_q2=record["quarter_dd"][1],
        semester_dd=record["semester_dd"],
        executed_hourly=record["semester_rate"],
        gate_accepted_pre_cap=int(record["gate_p85_count"]),
        stops=bool(any(record["quarter_stop"]) or record["semester_stop"]),
        qualifies=bool(record["qualifies"]),
    )


def run(*,seed,variant,founder_report,universe,manifest,golden,golden_manifest,
        target_path,target_manifest):
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError("Solo tres semillas y cinco variantes predefinidas")
    p=load_problem(universe,manifest,golden,golden_manifest)
    targets,info=load_targets(target_path,target_manifest,p["is_snapshot_sha256"])
    if len(targets["rr15_label"])!=len(p["scores"]):
        raise ValueError("Etiquetas desalineadas con el modelo")
    old_genes,base_metrics,source_generation=load_incumbent(
        founder_report,seed=seed,problem=p
    )
    founder=np.r_[old_genes,np.zeros(DIR_GENE_COUNT,dtype=float)]
    generations=PHASE2_GENERATIONS if variant in PHASE2 else PHASE3_GENERATIONS
    base=one_variant_candidate(founder,p,targets,variant)

    def evaluate(population):
        values=[one_variant_candidate(g,p,targets,variant) for g in population]
        return {k:np.asarray([r[k] for r in values]) for k in values[0]}

    evolution=evolve_population(
        evaluate,founder,f"m5_m15_{variant}",seed,
        first_generation=FIRST,last_generation=generations,ranking=base_key
    )
    new_genes=np.asarray(evolution["genome"],dtype="<f8")
    winner=one_variant_candidate(new_genes,p,targets,variant)
    if abs(winner["score"]-evolution["champion"]["score"])>1e-10:
        raise ValueError("El score ganador no se reprodujo")
    report=dict(
        protocol="PR38_evolved_targets_M5_M15_IS_v1",
        phase=2 if variant in PHASE2 else 3,
        seed=seed,variant=variant,source_generation=source_generation,
        original_founder_sha256=sha256(old_genes.astype("<f8").tobytes()).hexdigest(),
        dataset_sha256=p["is_snapshot_sha256"],
        target_dataset_sha256=info["labels_sha256"],
        genome_sha256=sha256(new_genes.tobytes()).hexdigest(),
        trained_genome=new_genes.tolist(),
        baseline=compact_stats(base),champion=compact_stats(winner),
        selected_generation=int(evolution["champion"]["generation"]),
        evaluated=evolution["evaluations"],
        generations=generations,population=POPULATION,
        quality_goal=dict(pf_per_quarter=PF_GOAL,semester_executed_hourly=MIN_EXECUTED_HOUR,
                          semester_dd_below=DD_GOAL,reproducible_in_all_seeds=True),
        signal_timeframes=["M5","M15"],gate_mode="p85",
        q1_q2_in_sample=True,oos_accessed=False,
        no_backprop=True,
        caution="Target is an IS training label, binary 5m contract PF remains the only economic result.",
        history=[dict(generation=x["generation"],score=x["score"],median_score=x["median_score"],
                      candidate=x["candidate"]) for x in evolution["generations"]],
    )
    dest=OUTPUT/f"phase{report['phase']}"
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/f"{variant}-seed-{seed}.json",report)
    print("PR38_TRAIN_VARIANT "+json.dumps({
        "seed":seed,"variant":variant,"generation":report["selected_generation"],
        "pf_q1":winner["quarter_pf"][0],"pf_q2":winner["quarter_pf"][1],
        "executed_h":winner["semester_rate"],"dd_semester":winner["semester_dd"],
        "qualified":winner["qualifies"],"oos_accessed":False
    },allow_nan=False),flush=True)
    return report


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--seed",type=int,required=True,choices=SEEDS)
    arg.add_argument("--variant",choices=VARIANTS,required=True)
    for field in ("founder_report","universe","manifest","golden","golden_manifest",
                  "target_path","target_manifest"):
        arg.add_argument("--"+field.replace("_","-"),type=Path,required=True)
    a=arg.parse_args()
    run(seed=a.seed,variant=a.variant,founder_report=a.founder_report,
        universe=a.universe,manifest=a.manifest,golden=a.golden,
        golden_manifest=a.golden_manifest,target_path=a.target_path,
        target_manifest=a.target_manifest)
