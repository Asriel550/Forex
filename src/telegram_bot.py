"""Inferencia M15 causal y envío Telegram sin ejecutar órdenes.

Usa el paquete de una ejecución exitosa: pesos, normalizador, manifiesto y
Parquet enriquecido ORIGINAL. Continúa el filtro GARCH desde su estado final;
no reancla las reestimaciones en una ventana arbitraria de datos recientes.
Solo se analiza la última vela CERRADA y no se envía si su señal ya caducó.
El estado de cuenta usado para una señal es una cuenta simulada inicial flat,
no un saldo obtenido de un broker. No se actualiza el agente con datos live.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

from filelock import FileLock
import numpy as np
import pandas as pd
import requests

from .agent import load_agent
from .decision import RiskConfig, resolve_action
from .data_loader import STEP, DEFAULT_INPUT_DIR, expected_grid, load_data, read_closures, utc
from .http_utils import retry_after_seconds, wait_seconds
from .environment import AccountConfig, market_feature_names, prepare_market_data
from .quant_engine import QuantConfig, MonteCarloGBM, SCALE, _fit_garch

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
LABELS = {0:'PUT (Vender)',1:'NEUTRO (No operar)',2:'CALL (Comprar)'}


class AlertError(ValueError):
    """Error seguro para logs: no contiene URLs con token ni respuestas remotas."""


class SignalSkipped(AlertError):
    """Vela atrasada, mercado cerrado o alerta ya tramitada."""


class TelegramRejected(AlertError):
    """Telegram confirmó que no aceptó el mensaje."""


def current_utc():
    return pd.Timestamp.now(tz='UTC')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as file:
            json.dump(data,file,allow_nan=False,indent=2)
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_bundle(model_dir, history_path):
    """Hashes vinculan el baseline, normalizador y checkpoint seleccionados."""
    model_dir, history_path = Path(model_dir), Path(history_path)
    manifest = json.loads((model_dir/'training_manifest.json').read_text())
    if manifest.get('manifest_version')!=3 or 'risk' not in manifest or manifest.get('reward_version')!='risk_adjusted_v1':
        raise AlertError('Se requiere bundle v3 con filtro de riesgo; regenerar features y entrenar')
    for path,key in [(model_dir/'best_ppo_agent.zip','best_model_sha256'),
                      (model_dir/'normalizer.json','normalizer_sha256'),
                      (history_path,'input_sha256')]:
        if not manifest.get(key) or sha256_file(path) != manifest[key]:
            raise AlertError('Paquete incoherente o antiguo: volver a ejecutar entrenamiento y publicar bundle')
    baseline = pd.read_parquet(history_path)
    if sha256_file(history_path) != manifest['input_sha256']:
        raise AlertError('El baseline cambió durante la lectura')
    metadata = baseline.attrs
    if metadata.get('direction_probability_horizon')!=1:
        raise AlertError('El bundle no contiene probabilidades de decisión a una vela')
    if not {'source_rows','origin_timestamp','garch_fits','quant_config'}.issubset(metadata):
        raise AlertError('Falta estado GARCH para inferencia incremental; regenerar features y entrenar')
    if metadata['quant_config'] != manifest['source_quant_metadata']['quant_config']:
        raise AlertError('Configuración cuantitativa distinta a la del modelo')
    normalizer = json.loads((model_dir/'normalizer.json').read_text())
    if sha256_file(model_dir/'normalizer.json') != manifest['normalizer_sha256']:
        raise AlertError('El normalizador cambió durante la lectura')
    names = list(market_feature_names(manifest['lookback']))
    if names != manifest['features'] or names != normalizer['features']:
        raise AlertError('Orden de observaciones incompatible')
    if manifest.get('actual_timesteps',0) <= 0:
        raise AlertError('Entrenamiento incompleto')
    return baseline, normalizer, manifest


def extend_quant_history(baseline, recent):
    """Actualiza solo barras nuevas, con la misma fase de refit que el histórico.

    El forecast sigma[t+1|t] archivado proporciona h para el próximo retorno.
    omega/alpha/beta están en puntos básicos² según quant_engine. Los fits
    nuevos usan únicamente su ventana pasada; MC reutiliza la misma semilla.
    """
    config = QuantConfig(**baseline.attrs['quant_config'])
    count = baseline.attrs['source_rows']
    if not isinstance(count,int) or count < len(baseline):
        raise AlertError('Contador de origen GARCH inválido')
    fresh = recent.loc[recent.timestamp > baseline.timestamp.iloc[-1]].copy()
    fresh = fresh.sort_values('timestamp').reset_index(drop=True)
    if fresh.timestamp.duplicated().any():
        raise AlertError('Datos recientes duplicados')
    result = pd.concat([baseline,fresh],ignore_index=True)
    fits = [dict(fit) for fit in baseline.attrs['garch_fits']]
    if not fits or fits[-1]['row'] >= count:
        raise AlertError('Estado GARCH inválido')
    omega,alpha,beta = (float(fits[-1][key]) for key in ('omega','alpha','beta'))
    next_h = float(baseline.cond_volatility.iloc[-1]*SCALE)**2
    mc = MonteCarloGBM(config.n_paths,config.horizon,config.seed)
    closes = result.close.to_numpy(dtype=float)
    if not np.isfinite(closes).all() or (closes<=0).any() or not np.isfinite(next_h) or next_h<=0:
        raise AlertError('Precios o varianza base inválidos')
    for k in range(len(fresh)):
        i,global_row = len(baseline)+k,count+k
        sample_start = max(0,i-config.window)
        current_return = (np.log(closes[i])-np.log(closes[i-1]))*SCALE
        if (global_row-config.min_observations)%config.refit_every == 0:
            sample = np.diff(np.log(closes[sample_start:i+1]))*SCALE
            if len(sample)<config.min_observations or len(sample)!=min(config.window,global_row):
                raise AlertError('Contexto histórico insuficiente para reestimar GARCH')
            omega,alpha,beta,h_t = _fit_garch(sample,config.max_iterations)
            fits.append(dict(row=global_row,first_return=global_row-len(sample),
                             omega=omega,alpha=alpha,beta=beta))
        else:
            h_t = next_h
        next_h = omega+alpha*current_return**2+beta*h_t
        if not np.isfinite(next_h) or next_h<=0:
            raise AlertError('Varianza GARCH no válida')
        sigma = np.sqrt(next_h)/SCALE
        p05,p95 = mc.terminal_quantiles(closes[i],sigma,config.drift_per_bar)
        up,down,tie = mc.direction_probabilities(sigma,config.drift_per_bar)
        for name,value in [('mc_prob_up',up),('mc_prob_down',down),('mc_prob_tie',tie)]:
            result.loc[i,name] = float(value)
        for name,value in [('cond_volatility',sigma),('mc_p05',float(p05)),('mc_p95',float(p95)),
                           ('dist_p05',(float(p05)-closes[i])/closes[i]),
                           ('dist_p95',(float(p95)-closes[i])/closes[i])]:
            result.loc[i,name] = value
    result.attrs = dict(baseline.attrs,source_rows=count+len(fresh),garch_fits=fits)
    return result


def latest_observation(prepared, normalizer, manifest):
    """Observación de la ÚLTIMA fila; no crea una vela futura ficticia.

    Equivale al reset del entorno en una cuenta inicial flat. Al ser un bot
    de señales no se afirma conocer saldo/operaciones reales del destinatario.
    """
    names = list(market_feature_names(manifest['lookback']))
    row = prepared.iloc[-1]
    values = row[names].to_numpy(dtype=float)
    mean,std = np.asarray(normalizer['mean']),np.asarray(normalizer['std'])
    clip = normalizer['clip']
    if (mean.shape!=values.shape or std.shape!=values.shape or (std<=0).any()
            or not np.isfinite(values).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not np.isfinite(clip) or clip<=0):
        raise AlertError('Normalización u observación inválida')
    account = AccountConfig(**manifest['account'])
    state = [0.,0.,0.,0.,account.stake/account.initial_balance,account.payout,0.]
    observation = np.r_[np.clip((values-mean)/std,-clip,clip),state].astype(np.float32)
    if not np.isfinite(observation).all():
        raise AlertError('Observación fuera de rango')
    return observation


def assert_fresh(candle_open, now, max_delay_seconds=180):
    close = utc(candle_open)+STEP
    clock = utc(now)
    if close != clock.floor('15min') or not 0 <= (clock-close).total_seconds() <= max_delay_seconds:
        raise SignalSkipped('Última vela cerrada ausente o atrasada; no se envía señal')
    if clock >= close+STEP:
        raise SignalSkipped('La expiración de la señal ya pasó')
    return close,close+STEP


def format_signal(action, row):
    close,expiry = utc(row.timestamp)+STEP,utc(row.timestamp)+2*STEP
    return (f'🔔 SEÑAL EUR/USD M15\nAcción: {LABELS[action]}\n'
            f'Volatilidad Cond.: {row.cond_volatility:.4f}\n'
            f'Soporte/Resist. (MC): P05: {row.mc_p05:.4f} | P95: {row.mc_p95:.4f}\n'
            f'Cierre analizado: {close.strftime("%Y-%m-%d %H:%M UTC")}\n'
            f'Expiración teórica: {expiry.strftime("%H:%M UTC")}')


def send_telegram(token, chat_id, text, *, session=None, max_attempts=3, max_wait=60., freshness_check=None):
    """POST sin reintentos de red/5xx ambiguos: Telegram no ofrece idempotencia.

    Solo se reintenta 429 explícito (petición rechazada), respetando retry_after.
    Errores nunca imprimen URL/token, cuerpo remoto ni excepciones de requests.
    """
    if not token or not chat_id:
        raise TelegramRejected('Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID')
    client = session or requests.Session()
    waited = 0.
    try:
        for attempt in range(max_attempts):
            if freshness_check:
                freshness_check()
            try:
                response = client.post(f'https://api.telegram.org/bot{token}/sendMessage',
                                       json={'chat_id':chat_id,'text':text},timeout=(10,20))
            except requests.RequestException:
                raise AlertError('Telegram: entrega no confirmada; no se reintenta automáticamente') from None
            try:
                payload = response.json()
                if not isinstance(payload,dict):
                    payload = {}
            except (ValueError,TypeError):
                payload = {}
            status = response.status_code
            if status >= 500:
                # Un fallo del servidor no confirma que sendMessage no se
                # procesó, aun si el cuerpo incluye ok:false. La outbox debe
                # seguir pending para impedir un segundo envío incierto.
                response.close()
                raise AlertError('Telegram: entrega no confirmada por fallo del servidor; no se reintenta automáticamente')
            if status == 429 or payload.get('error_code') == 429:
                delay = max(retry_after_seconds(response.headers.get('Retry-After')),
                            retry_after_seconds(payload.get('parameters',{}).get('retry_after')),1.)
                response.close()
                if attempt+1==max_attempts or waited+delay>max_wait:
                    raise TelegramRejected('Telegram limita el envío; presupuesto de espera agotado')
                wait_seconds(delay)
                waited += delay
                continue
            response.close()
            if 400<=status<500 or payload.get('ok') is False:
                raise TelegramRejected('Telegram rechazó el mensaje; comprobar secretos y permisos del bot')
            if status!=200 or payload.get('ok') is not True or not isinstance(payload.get('result',{}).get('message_id'),int):
                raise AlertError('Telegram: respuesta ambigua; revisar entrega antes de repetir')
            return payload['result']['message_id']
        raise TelegramRejected('Telegram: sin intentos disponibles')
    finally:
        if session is None:
            client.close()


def run_signal(*, model_dir=ROOT/'models', history_path=ROOT/'data/eurusd_m15_features.parquet',
               state_path=ROOT/'data/telegram_state.json', live_path=ROOT/'data/live_m15.parquet',
               closures=(), dry_run=False, max_delay_seconds=180, max_model_age_days=14, now=None,
               input_dir=DEFAULT_INPUT_DIR, input_timeframe='auto', timezone='EST'):
    clock = utc(now if now is not None else current_utc())
    boundary = clock.floor('15min')
    if not len(expected_grid(boundary,boundary+STEP,closures)):
        raise SignalSkipped('Mercado cerrado para la siguiente vela M15')
    if (clock-boundary).total_seconds()>max_delay_seconds:
        raise SignalSkipped('Runner demasiado tarde para esta vela')
    baseline,normalizer,manifest = load_bundle(model_dir,history_path)
    if clock-(utc(baseline.timestamp.iloc[-1])+STEP)>pd.Timedelta(days=max_model_age_days):
        raise AlertError('Bundle demasiado antiguo; volver a ejecutar pipeline de entrenamiento')
    recent = load_data(input_dir=input_dir,input_timeframe=input_timeframe,timezone=timezone,
                       output=live_path,closures=closures,now=clock)
    assert_fresh(recent.timestamp.iloc[-1],clock,max_delay_seconds)
    anchor = recent.loc[recent.timestamp==baseline.timestamp.iloc[-1]]
    if anchor.empty or not np.allclose(anchor[['open','high','low','close']].iloc[-1].to_numpy(dtype=float),
                                       baseline[['open','high','low','close']].iloc[-1].to_numpy(dtype=float),rtol=0,atol=1e-10):
        raise AlertError('El archivo estático cambió o no contiene la vela de anclaje; regenerar bundle')
    enriched = extend_quant_history(baseline,recent)
    prepared = prepare_market_data(enriched,manifest['lookback'],manifest['indicator_period'])
    row = prepared.iloc[-1]
    close,_ = assert_fresh(row.timestamp,clock,max_delay_seconds)
    observation = latest_observation(prepared,normalizer,manifest)
    model = load_agent(Path(model_dir)/'best_ppo_agent.zip')
    if sha256_file(Path(model_dir)/'best_ppo_agent.zip') != manifest['best_model_sha256']:
        raise AlertError('El modelo cambió durante la carga')
    if not model.observation_space.contains(observation):
        raise AlertError('Modelo incompatible con observación actual')
    action,_ = model.predict(observation,deterministic=True)
    action = np.asarray(action).item()
    if not model.action_space.contains(action) or action not in LABELS:
        raise AlertError('Predicción inválida')
    risk = RiskConfig(**manifest['risk'])
    decision = resolve_action(int(action),row.mc_prob_up,row.mc_prob_down,row.mc_prob_tie,
                              payout=manifest['account']['payout'],tie_policy=manifest['account']['tie_policy'],
                              risk=risk,n_paths=baseline.attrs['quant_config']['n_paths'])
    action = decision['action']
    text = format_signal(int(action),row)
    if dry_run:
        return dict(action=int(action),message=text,status='dry_run',decision=decision)
    token,chat_id = os.environ.get('TELEGRAM_BOT_TOKEN'),os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        raise TelegramRejected('Configurar TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID como secretos')
    # Se verifica de nuevo el reloj real tras descarga/features/inferencia.
    fresh_check = lambda:assert_fresh(row.timestamp,current_utc(),max_delay_seconds)
    key = hashlib.sha256(f'{chat_id}|EURUSD|M15|{close.isoformat()}'.encode()).hexdigest()
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(state_path)+'.lock',timeout=30):
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('key')==key and state.get('status') in ('sent','pending'):
            raise SignalSkipped('Esta vela ya se envió o tiene una entrega no confirmada')
        fresh_check()
        record = dict(key=key,candle_close=close.isoformat(),status='pending')
        atomic_json(state_path,record)  # outbox: evita duplicar ante un timeout ambiguo.
        try:
            message_id = send_telegram(token,chat_id,text,freshness_check=fresh_check)
        except TelegramRejected:
            atomic_json(state_path,dict(record,status='rejected'))
            raise
        atomic_json(state_path,dict(record,status='sent',message_id=message_id))
    return dict(action=int(action),status='sent',message_id=message_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--history',type=Path,default=ROOT/'data/eurusd_m15_features.parquet')
    parser.add_argument('--state',type=Path,default=ROOT/'data/telegram_state.json')
    parser.add_argument('--live-data',type=Path,default=ROOT/'data/live_m15.parquet')
    parser.add_argument('--closures',type=Path)
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT_DIR)
    parser.add_argument('--input-timeframe',choices=['auto','M1','M15'],default='auto')
    parser.add_argument('--timezone',default='EST')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--max-delay-seconds',type=int,default=180)
    args = parser.parse_args()
    if not 1<=args.max_delay_seconds<900:
        parser.error('max-delay-seconds debe estar entre 1 y 899')
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    try:
        result = run_signal(model_dir=args.model_dir,history_path=args.history,state_path=args.state,
                            live_path=args.live_data,closures=read_closures(args.closures),
                            dry_run=args.dry_run,max_delay_seconds=args.max_delay_seconds,
                            input_dir=args.input_dir,input_timeframe=args.input_timeframe,timezone=args.timezone)
    except SignalSkipped as exc:
        LOG.info('%s',exc)
        return
    except AlertError as exc:
        LOG.error('%s',exc)
        raise SystemExit(1) from None
    except Exception:
        LOG.error('Inferencia o datos no disponibles; no se envió una señal confirmada')
        raise SystemExit(1) from None
    print(result.get('message',result['status']))


if __name__=='__main__':
    main()

