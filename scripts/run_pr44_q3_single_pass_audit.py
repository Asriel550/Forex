"""PR44 — one-pass, no-training Q3 2026 OOS audit of ALL 24 frozen genomes.

Requires four independently reviewed, sealed Q3 M5/M15/GARCH universes.
Never downloads raw market data, fits GARCH, evolves, selects, or tunes.
Missing Q3 data is a BLOCKER, not a poor or invented PF.
"""
from __future__ import annotations
import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.adaptive_gate import AdaptiveGate
from src.decision import RiskConfig
from src.environment import AccountConfig
from src.replay_engine import replay_contracts
from src.reproducibility import array_sha256
def write_json(path, payload):
    Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')

SYMBOLS = ("EURUSD","GBPUSD","AUDUSD","USDCAD")
SEEDS = (42,43,44)
GROUPS = ("PR42_CONTROL","PR43_EVOLVED")
START = pd.Timestamp("2026-07-01T00:00:00Z")
END = pd.Timestamp("2026-10-01T00:00:00Z")
STEP = pd.Timedelta(minutes=5)
OUTPUT = Path("reports/pr44-q3-single-pass")
PROTOCOL = "PR44_Q3_24_frozen_genomes_single_pass_v1"
Q3_SCHEMA = "PR44_Q3_native_M5_M15_GARCH_universe_v1"
FIELD_DTYPES = dict(row_times_ns="<i8", outcome="<i1", volatility="<f8",
    entry="<f8", margin="<f8", timeframe="<f8", direction_inputs="<f8",
    legacy_mask="?")
SOURCE_SHA = {
 "EURUSD":"05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9",
 "GBPUSD":"2cc9ab95266cb3a87c2988ffd8006c79e9db68765d5ab54ded129c161fff4a0b",
 "AUDUSD":"8d1f34d5bf38ce1c4beeb5079d374a7af1a9d07a8161d07fc010d56b251b0eee",
 "USDCAD":"bf866689bb549bf2d647d8600bd343148dfb44040fb51c9cbb2baa7738f4f8b9",
}
# Recorded in the SUCCESS PR43 consolidation before any Q3 data was opened.
PINNED_SHA = {
 "EURUSD":{42:("ec79bd8729b162e1a74bbce406e62efbc7056ce236e2e6636d5769b5182f008a","64a4084a60d6bbce66b9cbe875e065dbc7ed69c1e438d4f6ec3c336023b2990b"),
           43:("d619568c12ebd7eca3098dd73cb717176a189cd8b3d6fae4e70e10f8d56f6996","56febfd2d9f4e69553e2a5612838bedfa63c71d5a1d97f9aa12c2b9509a38b43"),
           44:("91b79ff1ce32d6129e23d076f204e5af4adf0dc1461de3eb4a58ae8e69cc661c","86050d304a5b379b8d2612a68a10bbc77c687fdf83802cbc5d86408cfe6fd97f")},
 "GBPUSD":{42:("60c9974a10c7bdb7c424cf620f413d7161c093817fd1faee4690ed8c9da7d777","65026e5d01de6797f44472d07f74d156d6a701fdeddbf25cb4b1fbbb3fb88f86"),
           43:("6cccd602b8da59ee0fb14ca649f7c38f8abafac97b231a43c9be3ef3cc0b184d","64542cccbf880c94492486a87bb5a22f6fee0ffc46e963cf7a267abf442db34a"),
           44:("63661ab9cf390eff7ef528cb31723e3a6f4479a119675ebb02ac2410bb62bf48","6efc02ebd1efdcf5e9771ee0039204c605bdb2b3bdfda303dfb8982072e99c68")},
 "AUDUSD":{42:("fab51d173afe7da714c2f1ce8d6f82da57b263b4cecf89391e6922de15731d08","0fb0f0a3c54af45ce0cf41bc63309a8c8de636f889c8c04766425f7c164d4522"),
           43:("788db8e3db309c9c7d83b294aeac974a7f2ee9c792c18e5bff3753e07acb0fae","be33a1d48ac75b0f40fd99b6238d13db2a777128f35971bb5d41addf8926c72b"),
           44:("effef431fa6c52c0d945e9661b9808482999c2cf9537ebb5ed83fbe1de0a229d","d6fb77273f6eac271f038e9b1848ebe62d626fa3d8ecc3c1576e1cb08af23376")},
 "USDCAD":{42:("d9d212486f326af64ada6b25816db39ece25581885e5a1f1b23c1d812dbd4048","8d3578d57d85c3b117a7795d7fce5ed4f16e8244999fc57fc5129717829d9deb"),
           43:("75d616c6e0c5cc52ea4e3fd5e291aeb5a8504a32a1d41731e4641b8d3cbbee4b","722d9bf61664647a19ad82ada75b6f3b1e88151ebe7cbcccfc2db61b017aca93"),
           44:("dec20f1e6b0f9e370f961539d8ef0e589773a487ab729864e47350d3fc79f20a","158c53f1abc39b9ac91443695d8a4dee920642b7b3d2970153bb6a04e5b71c3b")},
}
RELEASE_PREFIX = "data-pr44-q3-"


def digest_genome(genes):
    a=np.asarray(genes,dtype="<f8")
    if a.shape!=(22,) or not np.isfinite(a).all():
        raise ValueError("Se requieren exactamente 22 genes finitos")
    return sha256(a.tobytes()).hexdigest()


def h1_profit_factor(pf1,pf2,pnl1,pnl2):
    """Reconstruct comparable H1 PF from FROZEN Q1+Q2 gross gains/losses.

    For finite PF>1 and positive quarter PnL, gross_loss=PnL/(PF-1).
    Explicitly precommitted before Q3 access; no OOS-derived calibration.
    """
    values=(pf1,pf2,pnl1,pnl2)
    if not all(isinstance(v,(float,int)) and np.isfinite(v) for v in values):
        raise ValueError("H1 PF/PnL no son finitos")
    if min(pf1,pf2)<=1 or min(pnl1,pnl2)<=0:
        raise ValueError("El campeón no tiene PF/PnL H1 apto")
    loss1=pnl1/(pf1-1.0)
    loss2=pnl2/(pf2-1.0)
    return float((pf1*loss1+pf2*loss2)/(loss1+loss2))


def assess_oos(pf_q3, dd_q3, h1_pf, *, trade_count, stopped):
    """All 3 STRICT rules, with no trade-count loophole or zero-loss infinite PF."""
    if not np.isfinite(h1_pf) or h1_pf<=1 or dd_q3 is None or not np.isfinite(dd_q3):
        raise ValueError("Métricas de riesgo inválidas")
    valid=pf_q3 is not None and np.isfinite(pf_q3) and pf_q3>=0
    decay=(1.0-pf_q3/h1_pf) if valid else None
    checks=dict(
        pf_gt_1_15=bool(valid and pf_q3>1.15),
        max_dd_lt_15=bool(dd_q3<15.),
        degradation_lt_40_pct=bool(valid and decay<.40),
        no_account_stop=bool(not stopped),
        at_least_one_executed=bool(trade_count>0),
    )
    return dict(pf_h1_reference=h1_pf, degradation_pct=(100.*decay if valid else None),
                checks=checks, qualifies=all(checks.values()))


def find_report(root,name):
    matches=list(Path(root).rglob(name))
    if len(matches)!=1:raise FileNotFoundError(f"Reporte congelado faltante/duplicado {name}: {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def frozen_24(control_root,evolved_root):
    """Validate ALL 24 signed models before reading any Q3 data."""
    frozen=[]
    for sym in SYMBOLS:
        for seed in SEEDS:
            c=find_report(control_root,f"{sym.lower()}-seed-{seed}.json")
            candidates=[p for p in Path(evolved_root).rglob(f"seed-{seed}.json")
                        if p.parent.name==sym.lower()]
            if len(candidates)!=1:
                raise FileNotFoundError(f"PR43 {sym} seed {seed} faltante/duplicado")
            e=json.loads(candidates[0].read_text(encoding="utf-8"))
            shas=PINNED_SHA[sym][seed]
            if (c.get("protocol")!="PR41_native_asset_genetic_20x128_M5M15_IS_v1"
                or c.get("symbol")!=sym or c.get("seed")!=seed
                or c.get("qualified_standalone") is not True
                or c.get("input_universe_sha256")!=SOURCE_SHA[sym]
                or c.get("oos_q3_accessed") is not False
                or c.get("champion_genome_sha256")!=shas[0]
                or digest_genome(c.get("champion_genome"))!=shas[0]):
                raise ValueError(f"PR42 control mal firmado: {sym} {seed}")
            if (e.get("protocol")!="PR43_continue_PR42_champions_100_gen_H1_IS_v1"
                or e.get("symbol")!=sym or e.get("seed")!=seed
                or e.get("new_evaluations")!=12800
                or e.get("additional_generations")!=100
                or e.get("q3_accessed") is not False
                or e.get("oos_evaluated") is not False
                or e.get("started_from_champion") is not True
                or e.get("source_pr42_genome_sha256")!=shas[0]
                or digest_genome(e.get("source_pr42_genome"))!=shas[0]
                or e.get("evolved_champion_sha256")!=shas[1]
                or digest_genome(e.get("evolved_champion_genome"))!=shas[1]
                or e.get("input_universe_sha256")!=SOURCE_SHA[sym]):
                raise ValueError(f"PR43 linaje mal firmado: {sym} {seed}")
            frozen.append(dict(symbol=sym,seed=seed,group="PR42_CONTROL",
                genes=np.asarray(c["champion_genome"],dtype="<f8"),
                h1_metrics=c["champion"],sha256=shas[0]))
            frozen.append(dict(symbol=sym,seed=seed,group="PR43_EVOLVED",
                genes=np.asarray(e["evolved_champion_genome"],dtype="<f8"),
                h1_metrics=e["evolved_champion_metrics"],sha256=shas[1]))
    assert len(frozen)==24 and len(set((r["symbol"],r["seed"],r["group"]) for r in frozen))==24
    return frozen


def require_q3_files(root):
    """Zero-compute preflight: fail before entering Q3 if any pair is absent."""
    mapping={}
    for sym in SYMBOLS:
        base=Path(root)/sym.lower()
        archive,manifest=base/"universe.npz",base/"manifest.json"
        if not archive.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"Falta Q3 M5/M15/GARCH nativo validado: {sym}")
        mapping[sym]=(archive,manifest)
    return mapping


def load_native_q3(symbol,archive,manifest):
    """Read exactly one native asset's Q3 series; reject H1 and unlabeled sources."""
    meta=json.loads(Path(manifest).read_text(encoding="utf-8"))
    hex64=lambda s:isinstance(s,str) and len(s)==64 and all(c in "0123456789abcdef" for c in s)
    months=("202607","202608","202609")
    required=("source_m5_ohlcv_sha256","source_m15_ohlcv_sha256",
              "garch_m5_recipe_sha256","frozen_h1_normalizer_sha256")
    if (meta.get("schema")!=Q3_SCHEMA or meta.get("symbol")!=symbol
        or meta.get("signal_timeframes")!=["M5","M15"]
        or meta.get("source_timeframes")!=["M5","M15"]
        or meta.get("raw_m1_signal_features") is not False
        or meta.get("garch_native_causal_refit") is not True
        or meta.get("frozen_h1_normalizer") is not True
        or meta.get("m15_closed_no_lookahead") is not True
        or meta.get("q3_full_source_coverage") is not True
        or len(meta.get("q2_eligible_volatility_tail",[]))!=48
        or meta.get("start_utc") not in (START.isoformat(),"2026-07-01T00:00:00Z")
        or meta.get("end_exclusive_utc") not in (END.isoformat(),"2026-10-01T00:00:00Z")
        or meta.get("months")!=list(months)
        or meta.get("frozen_h1_input_sha256")!=SOURCE_SHA[symbol]
        or any(not hex64(meta.get(x)) for x in required)
        or not isinstance(meta.get("source_zip_sha256"),dict)
        or set(meta["source_zip_sha256"])!=set(months)
        or any(not hex64(v) for v in meta["source_zip_sha256"].values())):
        raise ValueError("Q3 sin procedencia, GARCH propio o cobertura auditables: "+symbol)
    prior=np.asarray(meta["q2_eligible_volatility_tail"],dtype="<f8")
    if (prior.shape!=(48,) or not np.isfinite(prior).all() or (prior<0).any()
            or array_sha256(prior)!=meta.get("q2_eligible_volatility_tail_sha256")):
        raise ValueError("Q2 no verifica estado previo GARCH del mismo par")
    blob=Path(archive)
    if sha256(blob.read_bytes()).hexdigest()!=meta.get("universe_sha256"):
        raise ValueError("SHA256 del universo Q3 no coincide "+symbol)
    with np.load(blob,allow_pickle=False) as f:
        if set(f.files)!=set(FIELD_DTYPES):
            raise ValueError("Esquema de features PR41 alterado en Q3 "+symbol)
        data={k:f[k].copy() for k in FIELD_DTYPES}
    size=len(data["row_times_ns"])
    if (size<10000 or data["timeframe"].shape!=(size,8)
        or data["direction_inputs"].shape[0]!=size
        or any(data[k].shape!=(size,) for k in
               ("row_times_ns","outcome","volatility","entry","margin","legacy_mask"))
        or data["legacy_mask"].dtype!=np.dtype("?")):
        raise ValueError("Faltan señales M5/M15 o Q3 incompleto "+symbol)
    if not isinstance(meta.get("arrays_sha256"),dict):
        raise ValueError("Q3 sin hashes por array")
    for name,dtype in FIELD_DTYPES.items():
        if array_sha256(data[name],dtype=dtype)!=meta["arrays_sha256"].get(name):
            raise ValueError("Arrays Q3 alterados: "+name)
    t=pd.to_datetime(data["row_times_ns"],unit="ns",utc=True)
    if (not t.is_monotonic_increasing or not t.is_unique
        or t[0]<START or t[-1]>=END
        or not t.equals(t.floor("5min"))
        or not np.isin(data["outcome"],[-1,0,1]).all()
        or not np.isfinite(data["timeframe"]).all()
        or not np.isfinite(data["volatility"]).all()
        or (data["volatility"]<0).any()
        or any(not (t.month==month).any() for month in (7,8,9))):
        raise ValueError("Q3 fechas, inputs causales o outcomes inválidos")
    hours=meta.get("months_observed_hours")
    if (not isinstance(hours,dict) or set(hours)!=set(months)
        or any(not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<v<=744
               for v in hours.values())):
        raise ValueError("Horas Q3 no verificables")
    return data,t,float(sum(hours.values())),meta


def local_asset_head(data,genes,*,q2_garch_tail=None):
    """Exact PR41 entry/tanh/direction/GARCH-band formula; no EUR model."""
    g=np.asarray(genes,dtype=float)
    tf=np.asarray(data["timeframe"],dtype=float)
    sig=np.asarray(data["volatility"],dtype=float)
    def sigmoid(v):return 1./(1.+np.exp(-np.clip(v,-20.,20.)))
    score=(4.5+2.*np.tanh(g[18]))*np.tanh(tf@g[9:17]+g[17])
    entry=tf@g[:8]+g[8]
    floor=.3+2.*sigmoid(g[21])
    if q2_garch_tail is None:
        all_sigma=sig
        offset=0
    else:
        tail=np.asarray(q2_garch_tail,dtype="<f8")
        if tail.shape!=(48,) or not np.isfinite(tail).all() or (tail<0).any():
            raise ValueError("Cola de volatilidad Q2 incompatible")
        all_sigma=np.r_[tail,sig]
        offset=48
    past=pd.Series(np.maximum(all_sigma,1e-12)).shift(1).rolling(48,min_periods=24).median().to_numpy()[offset:]
    ratio=sig/np.where(np.isfinite(past),past,np.maximum(sig,1e-12))
    low=.35+sigmoid(g[19]);high=1.2+2.*sigmoid(g[20])
    mask=(entry>0)&(np.abs(score)>floor)&(ratio>=low)&(ratio<=high)
    return score,mask,dict(garch_low=float(low),garch_high=float(high),margin=float(floor))


def strictly_causal_p85(rows,scores):
    """One time-ordered pass; never see an outcome before its M5 expiry."""
    gate=AdaptiveGate(percentile=.85,elastic=True)
    pending=deque()
    reasons=[]
    for now,score,outcome in zip(rows.index,scores,rows["outcome"].to_numpy(),strict=True):
        while pending and pending[0][0]<=now:
            _,won=pending.popleft()
            gate.observe_settlement(won)
        reasons.append(gate.decide(float(score)).reason)
        won=None if outcome==0 else bool((1 if score>=0 else -1)==outcome)
        pending.append((now+STEP,won))
    return np.asarray(reasons,dtype="<U24")


def audit_one(model,data,times,hours,*,q2_garch_tail):
    """Never update genes or use cross-pair balance/risk/cap."""
    score,mask,band=local_asset_head(data,model["genes"],q2_garch_tail=q2_garch_tail)
    # Trades maturing at/after 00:00 Oct 1 cannot use Q4 prices.
    eligible=np.flatnonzero(mask & (times+STEP<END))
    if len(eligible):
        rows=pd.DataFrame({"outcome":data["outcome"][eligible]},index=times[eligible])
        scores=score[eligible]
        reasons=strictly_causal_p85(rows,scores)
        result,_=replay_contracts(
            rows, np.where(scores>=0,1,-1).astype(np.int8),
            AccountConfig(initial_balance=1000.,stake=10.,payout=.80),
            RiskConfig(drawdown_limit=.20),
            observed_hours=hours,hourly_cap=2,candidate_reasons=reasons,
        )
        op=result["operational"]
        pf=op["profit_factor"]
        dd=float(op["max_drawdown_pct"])
        count=int(op["trades"])
        stopped=op["account_stop"] is not None
        pnl=float(op["pnl_usd"])
    else:
        pf=None;dd=0.;count=0;stopped=False;pnl=0.
    m=model["h1_metrics"]
    ref=h1_profit_factor(m["pf_q1"],m["pf_q2"],m["pnl_q1"],m["pnl_q2"])
    verdict=assess_oos(pf,dd,ref,trade_count=count,stopped=stopped)
    return dict(
        group=model["group"],symbol=model["symbol"],seed=model["seed"],
        frozen_genome_sha256=model["sha256"],band=band,
        pf_q3=pf,dd_q3_pct=dd,executed_q3=count,
        executed_per_hour=count/hours,pnl_usd=pnl,stopped=stopped,
        proposed_after_local_garch=int(len(eligible)),observed_hours=hours,
        **verdict)


def run(*,control_root,evolved_root,q3_root,output=OUTPUT):
    models=frozen_24(control_root,evolved_root)
    inputs=require_q3_files(q3_root)
    results=[];provenance={}
    # One Q3 dataset load per asset; six frozen models per asset.
    for symbol in SYMBOLS:
        data,t,hours,meta=load_native_q3(symbol,*inputs[symbol])
        provenance[symbol]=dict(universe_sha256=meta["universe_sha256"],
                                source_zip_sha256=meta["source_zip_sha256"])
        for model in models:
            if model["symbol"]==symbol:
                results.append(audit_one(model,data,t,hours,
                    q2_garch_tail=meta["q2_eligible_volatility_tail"]))
    if len(results)!=24 or len({(r["group"],r["symbol"],r["seed"]) for r in results})!=24:
        raise ValueError("Auditoría incompleta")
    summary=dict(
        protocol=PROTOCOL,period_start=START.isoformat(),
        period_end_exclusive=END.isoformat(),
        genetic_evaluations=0,model_selection_performed=False,
        oos_first_pass=True,retraining=False,
        six_frozen_models_per_symbol=True,
        shared_cap=False,shared_bankroll=False,
        criteria=dict(pf_gt=1.15,dd_pct_lt=15.0,degradation_pct_lt=40.0,
             reference="H1 PF reconstructed from ORIGINAL Q1/Q2 PF and PnL",
             no_stop=True,no_zero_trades=True),
        data_provenance=provenance,completed=24,
        passes=sum(r["qualifies"] for r in results),
        results=results,
        caveat="This Q3 audit is one untouched OOS test; data provenance, finite sample size and repeated H1 optimization constrain alpha claims.",
    )
    Path(output).mkdir(parents=True,exist_ok=True)
    write_json(Path(output)/"q3-verdict.json",summary)
    print("PR44_Q3_AUDIT_FINAL "+json.dumps({
        "evaluated":24,"passing":summary["passes"],
        "q3_accessed":True,"genetic_evaluations":0,
        "pf_by_model":[dict(pair=r["symbol"],seed=r["seed"],group=r["group"],
                           pf=r["pf_q3"],dd=r["dd_q3_pct"],decay=r["degradation_pct"],
                           passes=r["qualifies"]) for r in results],
    },allow_nan=False),flush=True)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-root",type=Path,required=True)
    p.add_argument("--evolved-root",type=Path,required=True)
    p.add_argument("--q3-root",type=Path,required=True)
    p.add_argument("--output",type=Path,default=OUTPUT)
    run(**vars(p.parse_args()))
