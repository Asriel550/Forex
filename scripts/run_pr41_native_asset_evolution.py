"""PR41: independent 20-generation M5/M15/GARCH learning on each FX pair.

NO EURUSD direction, upstream or GARCH genes are copied into new assets.
Every pair, including EURUSD, starts with its OWN zero genome and 3
independent RNG streams. 8 closed M5/M15 features -> trainable tanh
direction head; separately trained entry head and GARCH regime band.

The COMMON economics (+80/-100 binary M5, bankroll risk, p85, 2/h)
come from the locked original experiment. No Q3 and no M1 model feature.
"""
from hashlib import sha256
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_dynamic_upstream import load_problem, one_candidate, key as old_key
from scripts.run_m5_conviction_evolution import setup
from src.evolution_multiscale import POPULATION, evolve_population
from src.garch_sweetspot import normalized_garch_ratio
from src.m5_multiasset_inputs import verify_new_asset
from src.m5_multiasset_portfolio import SYMBOLS,START,Q2,END
from src.artifacts import write_json
from src.upstream_filters import GENE_COUNT
from src.reproducibility import array_sha256


SEEDS=(42,43,44)
FIRST_GENERATION=1
GENERATIONS=20
LAST_GENERATION=FIRST_GENERATION+GENERATIONS-1
GENE_COUNT_LOCAL=22
PROTOCOL="PR41_native_asset_genetic_20x128_M5M15_IS_v1"
OUTPUT=Path("reports/pr41-independent-assets")
TARGET_PF=1.20
MAX_DD=15.0

# 8 entry weights + entry bias; 8 direction weights + direction bias
# + scale; 2 independent GARCH volatility-band thresholds;
# + 1 independent directional margin threshold. NOT PR38 weights.
ENTRY_W=slice(0,8)
ENTRY_B=8
DIR_W=slice(9,17)
DIR_B=17
DIR_GAIN=18
GARCH_LOW=19
GARCH_HIGH=20
MARGIN=21


def sigmoid(x):
    return float(1.0/(1.0+np.exp(-np.clip(x,-20.,20.))))


def genome_sha(genes):
    return sha256(np.asarray(genes,dtype="<f8").tobytes()).hexdigest()


def asset_head(data,genes):
    """Independent learnable direction, entry and relative-GARCH bands."""
    g=np.asarray(genes,dtype=float)
    tf=np.asarray(data["timeframe"],dtype=float)
    sigma=np.asarray(data["volatility"],dtype=float)
    n=len(tf)
    if (g.shape!=(GENE_COUNT_LOCAL,) or tf.shape!=(n,8) or sigma.shape!=(n,)
            or not np.isfinite(g).all() or not np.isfinite(tf).all()
            or not np.isfinite(sigma).all() or (sigma<0).any()):
        raise ValueError("Inputs neuronales M5/M15/GARCH inválidos")
    # No frozen EUR direction scores: THIS score is completely local to asset.
    magnitude=4.5+2.*np.tanh(g[DIR_GAIN])
    score=magnitude*np.tanh(tf@g[DIR_W]+g[DIR_B])
    entry=tf@g[ENTRY_W]+g[ENTRY_B]
    margin=np.abs(score)
    margin_floor=.30+2.*sigmoid(g[MARGIN])
    ratio=normalized_garch_ratio(sigma)
    low=.35+sigmoid(g[GARCH_LOW])
    high=1.2+2.*sigmoid(g[GARCH_HIGH])
    mask=(entry>0.)&(margin>margin_floor)&(ratio>=low)&(ratio<=high)
    # all decisions use current (already-known) M5/M15 and previous GARCH
    # history only, outcome is NOT consumed here.
    diagnostics=dict(
        band_lower=float(low),band_upper=float(high),
        margin_threshold=float(margin_floor),
        signal_score_gain=float(magnitude),
        upstream_before_garch=int(((entry>0)&(margin>margin_floor)).sum()),
        regime_eligible=int(((ratio>=low)&(ratio<=high)).sum()),
        proposed=int(mask.sum()),
    )
    return score,mask,diagnostics


def local_problem(symbol,universe,manifest,eur_universe,eur_manifest,golden,golden_manifest):
    if symbol not in SYMBOLS:raise ValueError("Divisa no predeclarada")
    # EUR establishes ONLY shared economics/contract and the verified hours,
    # not direction/entry/GARCH genes. The other pairs use their own
    # independently fitted and SHA-audited GARCH volatility and OHLC inputs.
    eur=load_problem(eur_universe,eur_manifest,golden,golden_manifest)
    if symbol=="EURUSD":
        data,info=eur["data"],eur["info"]
    else:
        data,info=verify_new_asset(symbol,universe,manifest)
    months=info["months_observed_hours"]
    keys=[f"2026{m:02d}" for m in range(1,7)]
    if set(months)!=set(keys):raise ValueError("Falta H1/Q1/Q2")
    q1=float(sum(months[k] for k in keys[:3]))
    q2=float(sum(months[k] for k in keys[3:]))
    t=pd.to_datetime(data["row_times_ns"],unit="ns",utc=True)
    if (len(t)<25000 or t[0]<START or t[-1]>=END
            or not (t<Q2).any() or not (t>=Q2).any()):
        raise ValueError("H1 no está completo o contiene Q3")
    return dict(
        data=data,info=info,times=t,scores=np.zeros(len(t),dtype=float),
        side=np.ones(len(t),dtype=np.int8),account=eur["account"],
        risk=eur["risk"],cap=eur["cap"],legacy_threshold=1.0,
        hours=dict(Q1=q1,Q2=q2,semester=q1+q2),
        is_snapshot_sha256=info["universe_sha256"],symbol=symbol,
        shared_contract_from="EURUSD original account/payout/risk ONLY",
        garch_source="independently refitted native-M5 GARCH" if symbol!="EURUSD" else "EURUSD signed original GARCH",
    )


def strict_asset_pass(rec):
    """Stand-alone admission, no special rate loophole or seed cherry-pick."""
    pf=rec["quarter_pf"]
    return bool(
        len(pf)==2 and all(p is not None and np.isfinite(p) and p>TARGET_PF for p in pf)
        and all(x>0 for x in rec["quarter_pnl_usd"])
        and all(x<MAX_DD for x in rec["quarter_dd"])
        and rec["semester_dd"]<MAX_DD
        and not any(rec["quarter_stop"]) and not rec["semester_stop"]
        and rec["semester_pnl_usd"]>0
    )


def evaluate_local(genes,problem):
    score,mask,band=asset_head(problem["data"],genes)
    # Preserve the precise original M5 binary payoff/p85/warmup/risk
    # evaluator. Feeds are now fully asset-native: entry/margin are a
    # boolean-to-fixed-cutoff transform, and no legacy EUR entry model runs.
    n=len(score)
    data=dict(problem["data"],
              entry=np.where(mask,2.,-2.),
              margin=np.where(mask,2.,0.))
    state=dict(problem,data=data,scores=score,
               side=np.where(score>=0.,1,-1).astype(np.int8))
    rec=one_candidate(np.zeros(GENE_COUNT),state,gate_mode="p85")
    pfs=[float(p) if p is not None and np.isfinite(p) else 0. for p in rec["quarter_pf"]]
    worst=min(pfs)
    geo=float(np.sqrt(np.clip(pfs[0],0,5)*np.clip(pfs[1],0,5)))
    rate=float(rec["semester_rate"])
    dd=float(rec["semester_dd"])
    stops=int(rec["semester_stop"])+sum(rec["quarter_stop"])
    operational=float(rec["score"])
    if rec["eligible"]<=500 or not all(np.isfinite(x) for x in (operational,rate,dd)):
        fitness=-1e6
    else:
        # Original multi-regime return signal plus independent asset-specific
        # PF/quarter quality, rate, drawdown and account-stop penalties.
        fitness=(
            .6*operational + .75*(worst-1.0)+.35*geo
            +.18*min(rate/.20,3.0)
            -.7*max(0.,(dd-MAX_DD)/MAX_DD)
            -1.5*stops
        )
    qualified=strict_asset_pass(rec)
    rec.update(
        score=float(fitness),original_operational_score=operational,
        qualifies=qualified,passes_all=qualified,
        regime_lower=band["band_lower"],regime_upper=band["band_upper"],
        direction_margin_threshold=band["margin_threshold"],
        direction_score_gain=band["signal_score_gain"],
        upstream_pre_garch=band["upstream_before_garch"],
        proposed_after_asset_garch=band["proposed"],
    )
    return rec


def result_summary(rec):
    return dict(
        pf_q1=rec["quarter_pf"][0],pf_q2=rec["quarter_pf"][1],
        pnl_q1=rec["quarter_pnl_usd"][0],pnl_q2=rec["quarter_pnl_usd"][1],
        dd_q1_pct=rec["quarter_dd"][0],dd_q2_pct=rec["quarter_dd"][1],
        dd_semester_pct=rec["semester_dd"],
        executed_hourly=rec["semester_rate"],
        executed_hourly_q1=rec["quarter_rate"][0],
        executed_hourly_q2=rec["quarter_rate"][1],
        pnl_semester=rec["semester_pnl_usd"],
        stop_q1=rec["quarter_stop"][0],stop_q2=rec["quarter_stop"][1],
        stop_semester=rec["semester_stop"],
        proposals=rec["eligible"],
        gate_accepted_pre_cap=rec["gate_p85_count"],
        garch_lower=rec["regime_lower"],garch_upper=rec["regime_upper"],
        margin_threshold=rec["direction_margin_threshold"],
        fitness=rec["score"],qualified=rec["qualifies"],
    )


def run(*,symbol,seed,universe,manifest,eur_universe,eur_manifest,golden,golden_manifest):
    if seed not in SEEDS or symbol not in SYMBOLS:
        raise ValueError("Solo símbolos predeclarados y 3 semillas")
    problem=local_problem(symbol,universe,manifest,eur_universe,eur_manifest,
                          golden,golden_manifest)
    # Deterministic BUT independent assets: separate RNG streams and no
    # carried-over EUR weights nor best genomes from any other market.
    rng_seed=int(seed+1000*SYMBOLS.index(symbol))
    zeros=np.zeros(GENE_COUNT_LOCAL,dtype="<f8")
    def batch(pop):
        values=[evaluate_local(g,problem) for g in pop]
        # The original evaluator uses None for undefined PF when no losses
        # occur. The genetic engine expects NumPy scalars with .tolist()
        # even on the first *invalid* founder candidate. NaN is a transport
        # marker here, NOT a reported profit factor or acceptance shortcut.
        def transport(value):
            if value is None:
                return float("nan")
            if isinstance(value,(list,tuple)):
                return [transport(x) for x in value]
            return value
        return {name:np.asarray([transport(r[name]) for r in values])
                for name in values[0]}
    evo=evolve_population(
        batch,zeros,f"pr41_{symbol}_independent_garch",rng_seed,
        first_generation=FIRST_GENERATION,last_generation=LAST_GENERATION,
        ranking=old_key,
    )
    champ_genes=np.asarray(evo["genome"],dtype="<f8")
    winner=evaluate_local(champ_genes,problem)
    if not np.isclose(winner["score"],evo["champion"]["score"],rtol=0,atol=1e-10):
        raise ValueError("No reproduce el campeón genético")
    initial=evaluate_local(zeros,problem)
    # evolve_population flattens each winning record into the generation
    # history (dict(record, median_score=...)), it does NOT nest it.
    history=[
        dict(generation=int(rec["generation"]),
             score=float(rec["score"]),
             median_score=float(rec["median_score"]),
             pf_q1=(float(rec["quarter_pf"][0]) if
                     np.isfinite(rec["quarter_pf"][0]) else None),
             pf_q2=(float(rec["quarter_pf"][1]) if
                     np.isfinite(rec["quarter_pf"][1]) else None),
             executed_hourly=float(rec["semester_rate"]),
             dd_semester_pct=float(rec["semester_dd"]),
             qualified=bool(rec["qualifies"]),
             regime_lower=float(rec["regime_lower"]),
             regime_upper=float(rec["regime_upper"]))
        for rec in evo["generations"]
    ]
    if [h["generation"] for h in history]!=list(range(1,21)):
        raise ValueError("Faltan generaciones o se alteró el horizonte")
    report=dict(
        protocol=PROTOCOL,symbol=symbol,seed=seed,optimizer_rng_seed=rng_seed,
        input_universe_sha256=problem["is_snapshot_sha256"],
        input_timeframes=["M5","M15"],raw_m1_signal_features=False,
        oos_q3_accessed=False,source_period="2026-01-01 to 2026-07-01 exclusive",
        generations=GENERATIONS,population=POPULATION,evaluations=evo["evaluations"],
        independent_from_other_pairs=True,
        euro_legacy_direction_weights_used=False,
        euro_legacy_entry_weights_used=False,
        shared_economics_only=True,
        gate_percentile=.85,asset_specific_garch=True,
        local_neural_head="8 M5/M15 closed features trainable tanh direction",
        gene_count=GENE_COUNT_LOCAL,initial_genome_sha256=genome_sha(zeros),
        champion_genome_sha256=genome_sha(champ_genes),
        champion_genome=champ_genes.tolist(),
        champion_generation=int(evo["champion"]["generation"]),
        founder=result_summary(initial),champion=result_summary(winner),
        qualified_standalone=bool(winner["qualifies"]),
        requires_three_seed_consistency=True,
        gate_for_portfolio="only if PF>1.20 both Q1/Q2 & DD<15% & no stops for all 3 seeds",
        history=history,automatic_promotion=False,
        caution="IS-only genetic selection; independent seeds are not OOS. No portfolio inclusion until all three seeds pass.",
    )
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_json(OUTPUT/f"{symbol.lower()}-seed-{seed}.json",report)
    print("PR41_INDEPENDENT_RESULT "+json.dumps({
        "symbol":symbol,"seed":seed,"generation":report["champion_generation"],
        "pf_q1":winner["quarter_pf"][0],"pf_q2":winner["quarter_pf"][1],
        "executed_hourly":winner["semester_rate"],"dd_h1":winner["semester_dd"],
        "qualified":winner["qualifies"],"oos_q3_accessed":False,
    },allow_nan=False),flush=True)
    return report


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--symbol",choices=SYMBOLS,required=True)
    a.add_argument("--seed",type=int,choices=SEEDS,required=True)
    for name in ("universe","manifest","eur_universe","eur_manifest","golden","golden_manifest"):
        a.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    opts=a.parse_args()
    run(symbol=opts.symbol,seed=opts.seed,universe=opts.universe,
        manifest=opts.manifest,eur_universe=opts.eur_universe,
        eur_manifest=opts.eur_manifest,golden=opts.golden,
        golden_manifest=opts.golden_manifest)
