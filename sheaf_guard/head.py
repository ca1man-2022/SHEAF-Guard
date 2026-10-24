import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

class StableScaler(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        self.mean_ = X.mean(0)
        scale = np.std(X - X[:1], axis=0)
        self.constant_ = scale < 1e-10
        self.scale_ = np.where(self.constant_, 1, scale)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

def meta(C):
    return make_pipeline(StableScaler(), LogisticRegression(C=C, max_iter=10000))
