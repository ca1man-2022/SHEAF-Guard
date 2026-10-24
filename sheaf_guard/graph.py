import numpy as np
from scipy import sparse
from .utils import top

class Bank:

    def __init__(self, docs, x, y, ids, groups):
        order = np.argsort(ids, kind='stable')
        self.x, self.y = (x[order], y[order])
        self.ids, self.groups = (ids[order], groups[order])
        self.docs, self.nd = (docs, len(docs))
        self.bank = np.concatenate([docs, self.x])
        self.prior = float(self.y.mean())
        self.Y = np.zeros((len(self.bank), 2))
        self.Y[self.nd + np.arange(len(self.x)), self.y] = 1
        if not np.array_equal(np.unique(self.y), [0, 1]):
            raise ValueError('Fit memory requires both classes.')
        self.graphs = {k: self.graph(k) for k in (5, 10, 20)}

    def graph(self, k):
        families = {}
        qq = self.x @ self.x.T
        np.fill_diagonal(qq, -np.inf)
        nq = top(qq, k - 1)
        families['QQ'] = [sorted(set([self.nd + i] + (self.nd + nq[i]).tolist())) for i in range(len(self.x))]
        if self.nd:
            dd = self.docs @ self.docs.T
            np.fill_diagonal(dd, -np.inf)
            nn = top(dd, k - 1)
            families['DD'] = [sorted(set([i] + nn[i].tolist())) for i in range(self.nd)]
            dq = top(self.docs @ self.x.T, k - 1)
            families['DQ'] = [sorted(set([i] + (self.nd + dq[i]).tolist())) for i in range(self.nd)]
        edges = []
        counts = {}
        for family in ['DD', 'QQ', 'DQ']:
            if family not in families:
                continue
            unique = {}
            for center, members in enumerate(families[family]):
                unique.setdefault(tuple(members), []).append(center)
            counts[family] = len(unique)
            for members, centers in unique.items():
                edges.append(dict(family=family, members=list(members), centers=centers, weight=1 / len(unique)))
        ri = []
        ci = []
        for i, e in enumerate(edges):
            ri.extend(e['members'])
            ci.extend([i] * len(e['members']))
        H = sparse.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(self.bank), len(edges)))
        w = np.array([e['weight'] for e in edges])
        card = np.asarray(H.sum(0)).ravel()
        meta = dict(k=k, edges=edges, weights=w, cardinality=card, family_counts=counts)
        return dict(H=H, w=w, card=card, meta=meta)
