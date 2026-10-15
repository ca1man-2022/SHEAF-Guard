import numpy as np


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    out = raw.astype(np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def extension(query, maps, boundaries, fidelity, weights=None):
    u = _array(query, "query")
    R = _array(maps, "maps")
    b = _array(boundaries, "boundaries")
    lam = _array(fidelity, "fidelity")
    if u.ndim != 1 or u.size == 0:
        raise ValueError("query must have shape (d,) with d > 0")
    d = u.size
    if R.ndim != 3 or R.shape[0] == 0 or R.shape[1:] != (d, d):
        raise ValueError("maps must have shape (M,d,d) with M > 0")
    m = len(R)
    if b.shape != (m, 2, d):
        raise ValueError("boundaries must have shape (M,2,d)")
    if lam.ndim != 0 or lam <= 0:
        raise ValueError("fidelity must be a positive scalar")
    if weights is not None:
        w = _array(weights, "weights")
        if w.shape != (m,) or np.any(w < 0):
            raise ValueError("weights must be nonnegative with shape (M,)")
        if not np.allclose(w, np.full(m, 1.0/m), rtol=0, atol=1e-14):
            raise ValueError("only uniform normalized weights 1/M are supported")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        A = lam*np.eye(d) + np.einsum("eji,ejk->ik", R, R)/m
        h = np.einsum("eji,ecj->ci", R, b)/m
        g = lam*u[None] + h
        L = np.linalg.cholesky(A)
        z = np.linalg.solve(L.T, np.linalg.solve(L, g.T)).T
        residual = ((np.einsum("eij,cj->eci", R, z)-b)**2).sum(2)
        displacement = ((z-u)**2).sum(1)
        E = lam*displacement + residual.mean(0)
        closed = lam*(u@u) + (b*b).sum(2).mean(0) - (g*z).sum(1)
        gradient = 2*(z@A-g)
    if not all(np.isfinite(v).all() for v in (E, z, A, closed, gradient)):
        raise FloatingPointError("nonfinite numerical solution")
    return dict(energies=E, states=z, residual=residual,
                displacement=displacement, closed=closed, matrix=A,
                rhs=g, gradient=gradient)


def compatibility_readout(energies):
    E = _array(energies, "energies")
    if E.shape != (2,) or np.any(E < 0):
        raise ValueError("energies must contain two nonnegative values")
    return np.array([E[0]-E[1], E.min()], dtype=np.float64)
