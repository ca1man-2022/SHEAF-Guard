import numpy as np
EPS = 1e-12

def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)

def top(sim, n):
    return np.argsort(-sim, axis=1, kind='stable')[:, :min(n, sim.shape[1])]

def logit(p):
    p = np.clip(p, 1e-06, 1 - 1e-06)
    return np.log(p / (1 - p))
