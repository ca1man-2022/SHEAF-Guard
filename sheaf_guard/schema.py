import hashlib
import unicodedata
import numpy as np


def query_groups(texts):
    normalized = [' '.join(unicodedata.normalize('NFKC', t).casefold().split()) for t in texts]
    return np.array([hashlib.sha256(t.encode()).hexdigest() for t in normalized])


def vectors(values, name, dimension=None):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(name + ' must be a nonempty finite matrix.')
    if dimension is not None and x.shape[1] != dimension:
        raise ValueError(name + ' has an incompatible embedding dimension.')
    if np.any(np.linalg.norm(x, axis=1) < 1e-12):
        raise ValueError(name + ' contains a zero vector.')
    return x


def identifiers(values, n, name, unique=False):
    x = np.asarray(values)
    if x.ndim != 1 or len(x) != n or x.dtype.kind not in 'US':
        raise ValueError(name + ' must contain one string per row.')
    if np.any(np.char.str_len(x) == 0) or (unique and len(set(x)) != n):
        raise ValueError(name + ' must be nonempty' + (' and unique.' if unique else '.'))
    return x


def training_labels(values, n):
    y = np.asarray(values)
    if y.ndim != 1 or len(y) != n or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Training labels must contain both ID=0 and OOD=1.')
    return y.astype(np.int64)


def load_embeddings(path):
    required = {'kb_embeddings', 'fit_embeddings', 'fit_labels', 'fit_ids', 'fit_groups',
                'query_embeddings', 'query_ids', 'query_groups'}
    with np.load(path, allow_pickle=False) as data:
        if not required.issubset(data.files):
            raise ValueError('Missing embedding-schema fields: ' + ', '.join(sorted(required - set(data.files))))
        out = {k: data[k] for k in required}
    kb = vectors(out['kb_embeddings'], 'kb_embeddings')
    for prefix in ['fit', 'query']:
        x = vectors(out[prefix + '_embeddings'], prefix + '_embeddings', kb.shape[1])
        identifiers(out[prefix + '_ids'], len(x), prefix + '_ids', True)
        identifiers(out[prefix + '_groups'], len(x), prefix + '_groups')
    training_labels(out['fit_labels'], len(out['fit_embeddings']))
    for suffix in ['ids', 'groups']:
        if set(out['fit_' + suffix]) & set(out['query_' + suffix]):
            raise ValueError('Fit/query ' + suffix + ' overlap.')
    return out
