import numpy as np
from scipy.linalg import cho_factor, cho_solve

def restriction(points, eps, rho=0.5):
    d = points.shape[1]
    centered = points - points.mean(0)
    cov = centered.T @ centered / len(points)
    if len(points) < 2 or np.trace(cov) <= eps * 1e-06:
        return (np.eye(d), True)
    v = (1 - rho) * cov + (rho * np.trace(cov) / d + eps) * np.eye(d)
    eig, vec = np.linalg.eigh(v)
    eig = np.maximum(eig, eps)
    raw = vec / np.sqrt(eig) @ vec.T
    return (raw / np.sqrt(np.sum(raw * raw) / d), False)

def extension(u, R, b, lam, no_extension=False):
    m, d, _ = R.shape
    A = lam * np.eye(d) + np.einsum('eji,ejk->ik', R, R) / m
    h = np.einsum('eji,ecj->ci', R, b) / m
    g = lam * u[None] + h
    z = np.broadcast_to(u, (2, d)).copy() if no_extension else cho_solve(cho_factor(A, lower=True), g.T).T
    residual = ((np.einsum('eij,cj->eci', R, z) - b) ** 2).sum(2)
    displacement = ((z - u) ** 2).sum(1)
    E = lam * displacement + residual.mean(0)
    closed = lam * (u @ u) + (b * b).sum(2).mean(0) - (g * z).sum(1)
    if not no_extension:
        assert np.allclose(E, closed, atol=1e-10, rtol=1e-09)
    assert np.isfinite(E).all() and E.min() >= 0
    return dict(E=E, z=z, displacement=displacement, residual=residual, closed=closed, A=A, h=h, gradient=2 * (z @ A - g))

class SheafMemory:

    def __init__(self, bank, pca):
        self.bank = bank
        self.pca = pca
        self.d = d = 8
        self.variant = variant = 'Full'
        self.u = pca.transform(self.bank.bank)[:, :d]
        self.eps = max(float(np.mean(pca.explained_variance_[:d])) * 0.001, 1e-12)
        uq = self.u[self.bank.nd:]
        means = np.array([uq[self.bank.y == c].mean(0) for c in [0, 1]])
        shared, _ = restriction(self.u, self.eps)
        self.maps, self.boundaries, self.counts, self.members, self.fallback = ({}, {}, {}, {}, {})
        for k, graph in self.bank.graphs.items():
            members = [np.array(e['members']) for e in graph['meta']['edges']]
            maps, boundaries, counts, fallback = ([], [], [], [])
            for ix in members:
                R, bad = restriction(self.u[ix], self.eps)
                if variant == 'Identity':
                    R = np.eye(d)
                if variant == 'Shared-Map':
                    R = shared
                qi = ix[ix >= self.bank.nd] - self.bank.nd
                n = np.array([np.sum(self.bank.y[qi] == c) for c in [0, 1]])
                mu = np.array([(uq[qi[self.bank.y[qi] == c]].sum(0) + 5 * means[c]) / (n[c] + 5) for c in [0, 1]])
                maps.append(R)
                boundaries.append(mu @ R.T)
                counts.append(n)
                fallback.append(bad)
            self.members[k] = members
            self.maps[k] = np.array(maps)
            self.boundaries[k] = np.array(boundaries)
            self.counts[k] = np.array(counts)
            self.fallback[k] = np.array(fallback)

    def select(self, q):
        sim = q @ self.bank.bank.T
        return {k: np.argsort(-np.array([sim[:, ix].max(1) for ix in members]).T, axis=1, kind='stable')[:, :8] for k, members in self.members.items()}

    def read(self, q, ids, groups, lam, selections=None, details=False):
        assert set(ids).isdisjoint(self.bank.ids)
        assert set(groups).isdisjoint(self.bank.groups)
        selections = self.select(q) if selections is None else selections
        u = self.pca.transform(q)[:, :self.d]
        X, info = ([], {})
        for k in (5, 10, 20):
            values = []
            zs = []
            displacements = []
            residuals = []
            for i, ix in enumerate(selections[k]):
                v = extension(u[i], self.maps[k][ix], self.boundaries[k][ix], lam, self.variant == 'No-Extension')
                values.append(v['E'])
                zs.append(v['z'])
                displacements.append(v['displacement'])
                residuals.append(v['residual'])
            E = np.array(values)
            X.extend([E[:, 0] - E[:, 1], E.min(1)])
            if details:
                info.update({f'{k}_E': E, f'{k}_z': np.array(zs), f'{k}_displacement': np.array(displacements), f'{k}_residual': np.array(residuals), f'{k}_edges': selections[k], f'{k}_counts': self.counts[k][selections[k]], f'{k}_condition': np.linalg.cond(self.maps[k][selections[k]])})
        return (np.column_stack(X), info)
