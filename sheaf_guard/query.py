import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class BaseModel:

    def __init__(self, config):
        self.config = config

    def kernel(self, x):
        dot = x @ self.reference.T
        if self.config['family'] == 'RBF-SVM':
            d = np.maximum((x * x).sum(1)[:, None] + (self.reference * self.reference).sum(1)[None] - 2 * dot, 0)
            return np.exp(-self.config['gamma'] * d)
        return dot

    def fit(self, x, y, K=None):
        self.reference = x
        if self.config['family'] == 'LR':
            self.model = LogisticRegression(C=self.config['C'], max_iter=10000, class_weight=None).fit(x, y)
        else:
            self.model = SVC(C=self.config['C'], kernel='precomputed', probability=False).fit(self.kernel(x) if K is None else K, y)
        assert np.array_equal(self.model.classes_, [0, 1])
        return self

    def predict(self, x, K=None):
        return self.model.decision_function(x if self.config['family'] == 'LR' else self.kernel(x) if K is None else K)

def resolve(c, x):
    out = dict(c)
    if c['family'] == 'RBF-SVM':
        out['gamma'] = c['gamma_factor'] / (x.shape[1] * float(x.var()))
    return out
QUERY_CONFIGS = ({'family': 'LR', 'prep': 'kb_standardized', 'C': 0.30000000000000004}, {'family': 'Linear-SVM', 'prep': 'kb_standardized', 'C': 0.01}, {'family': 'RBF-SVM', 'prep': 'unit_raw', 'C': 100, 'gamma_factor': 0.1})

class QueryModels:

    def fit(self, raw, unit, labels, scaler):
        self.scaler = scaler
        self.models = []
        standardized = scaler.transform(raw)
        for config in QUERY_CONFIGS:
            x = standardized if config['prep'] == 'kb_standardized' else unit
            if config['family'] != 'LR' and len(x) ** 2 * 8 > 2 ** 30:
                raise MemoryError('The exact dense SVM kernel exceeds 1 GiB.')
            self.models.append(BaseModel(resolve(config, x)).fit(x, labels))
        return self

    def transform(self, raw, unit):
        standardized = self.scaler.transform(raw)
        return np.column_stack([m.predict(standardized if c['prep'] == 'kb_standardized' else unit) for m, c in zip(self.models, QUERY_CONFIGS)])
