import numpy as np
from scipy import sparse
from .utils import EPS, logit, top

class Memory:

    def __init__(self, bank):
        self.base = bank
        self.kind = 'HG'
        self.nd, self.ids, self.groups = (bank.nd, bank.ids, bank.groups)
        self.Y, self.prior, self.graphs = (bank.Y, bank.prior, bank.graphs)

    def propagate(self, k, alpha):
        if alpha != 0:
            raise ValueError('The fixed configuration uses alpha=0.')
        return self.Y

    def geometry(self, q, ids, groups):
        assert not set(ids) & set(self.ids) and (not set(groups) & set(self.groups))
        b = self.base
        cq = q @ b.x.T
        cd = q @ b.docs.T
        qi = top(cq, 20)
        di = top(cd, 20)
        qi_cos = np.take_along_axis(cq, qi, axis=1)
        di_cos = np.take_along_axis(cd, di, axis=1)
        ni = np.concatenate([di, self.nd + qi], axis=1) if self.nd else qi
        cos = np.concatenate([di_cos, qi_cos], axis=1) if self.nd else qi_cos
        geom = dict(qi=qi, qi_cos=qi_cos, di_cos=di_cos, ni=ni, cos=cos, class_gap=cq[:, b.y == 1].max(1) - cq[:, b.y == 0].max(1))
        if self.kind == 'HG':
            allcos = np.concatenate([cd, cq], axis=1) if self.nd else cq
            R = sparse.csr_matrix((np.ones(ni.size), (np.repeat(np.arange(len(q)), ni.shape[1]), ni.ravel())), shape=(len(q), len(self.Y)))
            geom['edge'] = {}
            for k, g in self.graphs.items():
                cand = (R @ g['H']).toarray() > 0
                mean = allcos @ g['H'] / g['card'][None]
                width = max((len(e['members']) for e in g['meta']['edges']))
                members = np.array([e['members'] + [e['members'][0]] * (width - len(e['members'])) for e in g['meta']['edges']])
                maxcos = np.empty_like(mean)
                for off in range(0, len(q), 32):
                    maxcos[off:off + 32] = allcos[off:off + 32, members].max(2)
                geom['edge'][k] = dict(candidate=cand, relevance=(mean + maxcos) / 2)
        return geom

    def read(self, geom, alpha, T, scales=(5, 10, 20)):
        b = self.base
        v = geom['qi_cos']
        w = np.exp((v - v.max(1, keepdims=True)) / T)
        knn = (w * b.y[geom['qi']]).sum(1) / w.sum(1)
        local = np.column_stack([logit(knn), geom['class_gap'], geom['di_cos'][:, 0], geom['di_cos'][:, :5].mean(1)])
        ps = []
        vs = []
        es = []
        covs = []
        masses = []
        fallback = []
        for k in scales:
            F = self.propagate(k, alpha)
            g = self.graphs[k]
            if self.kind == 'HG':
                G = g['H'].T @ F / g['card'][:, None]
                mass = G.sum(1)
                pe = G[:, 1] / np.maximum(mass, EPS)
                candidate = geom['edge'][k]['candidate']
                valid = candidate & (mass[None] > EPS)
                relevance = geom['edge'][k]['relevance']
                prob = pe[None]
                quality = mass[None]
            else:
                mass = F.sum(1)
                pe = F[:, 1] / np.maximum(mass, EPS)
                ni = geom['ni']
                candidate = np.ones_like(ni, dtype=bool)
                valid = mass[ni] > EPS
                relevance = geom['cos']
                prob = pe[ni]
                quality = mass[ni]
            bad = ~valid.any(1)
            relevance = np.where(valid, relevance, -np.inf)
            mx = np.max(relevance, axis=1, keepdims=True)
            mx[bad] = 0
            weights = np.exp((relevance - mx) / T)
            weights /= np.maximum(weights.sum(1, keepdims=True), EPS)
            p = (weights * prob).sum(1)
            var = (weights * (prob - p[:, None]) ** 2).sum(1)
            p[bad] = self.prior
            var[bad] = 0
            ent = -(p * np.log(np.maximum(p, EPS)) + (1 - p) * np.log(np.maximum(1 - p, EPS)))
            cov = valid.sum(1) / np.maximum(candidate.sum(1), 1)
            ps.append(p)
            vs.append(var)
            es.append(ent)
            covs.append(cov)
            masses.append((weights * quality).sum(1))
            fallback.append(bad)
        p = np.array(ps).T
        feature = np.column_stack([logit(p), np.array(vs).T, np.array(es).T, np.array(covs).T, (p - p[:, :1]).std(1)])
        assert np.isfinite(feature).all()
        return dict(local=local, structure=feature, p=p, mass=np.array(masses).T, fallback=np.array(fallback).T)
