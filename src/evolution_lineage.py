"""Carga y migra genomas de investigación M5 con hashes de contenido."""

import base64
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .environment import AccountConfig, market_feature_names
from .evolution_context import CONTEXT_FEATURES
from .market import MarketConfig

MARKET = MarketConfig('M5', 'binary', 1)
LEGACY_FORMAT = 'forex_m5_evolution_candidate_v3'
CONTEXT_FORMAT = 'forex_m5_evolution_candidate_v4'
FORMAT = 'forex_m5_evolution_candidate_v5'
PINNED_SEED = Path(__file__).resolve().parents[1] / 'config/evolution_m5_seed_gen20.json'
INITIAL_SIGMA = .12
MARKET_FEATURES = market_feature_names()
OBSERVATION_FEATURES = (*MARKET_FEATURES, *CONTEXT_FEATURES)


@dataclass(frozen=True)
class Founder:
    genome: np.ndarray
    normalizer: dict
    account: AccountConfig
    drawdown_limit: float
    next_generation: int
    mutation_sigma: float
    provenance: dict


def genome_payload(genome):
    genome = np.asarray(genome, dtype=float)
    entry, direction = genome[:-1].reshape(2, -1)
    return {
        'entry': dict(weights=entry[:-1].tolist(), bias=float(entry[-1])),
        'direction_residual': dict(weights=direction[:-1].tolist(), bias=float(direction[-1])),
        'activation_threshold': float(genome[-1]),
    }


def _read_head(payload, width):
    weights = np.asarray(payload['weights'], dtype=float)
    bias = float(payload['bias'])
    if weights.shape != (width,) or not np.isfinite(weights).all() or not np.isfinite(bias):
        raise ValueError('Pesos o sesgo del genoma incompatibles')
    return weights, bias


def _check_normalizer(normalizer, names):
    mean, std = np.asarray(normalizer['mean'], dtype=float), np.asarray(normalizer['std'], dtype=float)
    clip = float(normalizer['clip'])
    if (normalizer['features'] != list(names) or mean.shape != (len(names),) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0.).any()
            or not np.isfinite(clip) or clip <= 0.):
        raise ValueError('Normalizador del linaje incompatible')


def load_founder(seed_bundle, context_train, train_end, search_end):
    """Migra v3/v4 con umbral cero y conserva el umbral heredado de v5."""
    origin = {}
    if seed_bundle is None:
        pinned = json.loads(PINNED_SEED.read_text(encoding='utf-8'))
        bundle = base64.b64decode(pinned['bundle_base64'], validate=True)
        if sha256(bundle).hexdigest() != pinned['bundle_sha256']:
            raise ValueError('SHA256 incorrecto para el fundador archivado')
        origin = {key: pinned[key] for key in ('run_id', 'artifact_id', 'artifact_sha256')}
    else:
        bundle = Path(seed_bundle).read_bytes()

    with ZipFile(BytesIO(bundle)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        contents = {}
        for name in ('genome.json', 'normalizer.json'):
            raw = archive.read(name)
            if sha256(raw).hexdigest() != manifest['sha256'].get(name):
                raise ValueError(f'SHA256 incorrecto para {name}')
            contents[name] = json.loads(raw)

    version = manifest.get('format')
    if (version not in (LEGACY_FORMAT, CONTEXT_FORMAT, FORMAT) or manifest.get('market') != asdict(MARKET)
            or manifest.get('profile') != 'scalping' or manifest.get('holdout_test_evaluated') is not False):
        raise ValueError('El fundador debe ser un candidato de investigación M5 binario compatible')
    for phase, cutoff in (('train', train_end), ('search', search_end)):
        if pd.Timestamp(manifest['splits'][phase]['end']) > cutoff:
            raise ValueError(f'El fundador utilizó datos posteriores al corte actual de {phase}')

    generation = manifest['selection']['generation']
    state = manifest.get('lineage_state', {})
    last = state.get('last_completed_generation', generation)
    sigma = float(state.get('next_mutation_sigma', INITIAL_SIGMA))
    if (type(generation) is not int or type(last) is not int or not 0 <= generation <= last
            or not .025 <= sigma <= .35):
        raise ValueError('Estado del linaje incompatible')
    if seed_bundle is None and (version != LEGACY_FORMAT or generation != pinned['generation']):
        raise ValueError('El fundador fijado debe ser el genoma de la generación 20 de la PR #18')

    payload, normalizer = contents['genome.json'], contents['normalizer.json']
    width = len(OBSERVATION_FEATURES)
    if version == LEGACY_FORMAT:
        _check_normalizer(normalizer, MARKET_FEATURES)
        inherited, bias = _read_head(payload, len(MARKET_FEATURES))
        context = context_train.loc[:, list(CONTEXT_FEATURES)].to_numpy(dtype=float)
        mean, std = context.mean(axis=0), context.std(axis=0)
        normalizer = dict(features=list(OBSERVATION_FEATURES),
                          mean=[*normalizer['mean'], *mean.tolist()],
                          std=[*normalizer['std'], *np.where(std < 1e-12, 1., std).tolist()],
                          clip=normalizer['clip'])
        entry = np.zeros(width)
        entry[:len(inherited)] = inherited
        direction, direction_bias = np.zeros(width), 0.
    else:
        entry, bias = _read_head(payload['entry'], width)
        direction, direction_bias = _read_head(payload['direction_residual'], width)

    activation_threshold = float(payload['activation_threshold']) if version == FORMAT else 0.
    if not np.isfinite(activation_threshold) or activation_threshold < 0.:
        raise ValueError('Umbral de activación del linaje inválido')
    _check_normalizer(normalizer, OBSERVATION_FEATURES)
    limit = float(manifest['risk']['drawdown_limit'])
    if not 0. < limit <= 1.:
        raise ValueError('Límite de drawdown del linaje inválido')
    provenance = dict(origin, format=version, bundle_sha256=sha256(bundle).hexdigest(),
                      source_generation=generation, last_completed_generation=last)
    return Founder(np.r_[entry, bias, direction, direction_bias, activation_threshold], normalizer,
                   AccountConfig(**manifest['account']), limit, last + 1, sigma, provenance)
