import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .graph import Bank
from .memory import Memory
from .csem import SheafMemory
from .query import QueryModels
from .head import meta
from .schema import vectors, identifiers, training_labels
from .utils import unit


FEATURE_NAMES = (
    'LR_score', 'LinearSVM_score', 'RBFSVM_score',
    'labelled_query_knn_logit', 'class_nearest_cosine_gap', 'MaxDoc', 'top5_doc_mean',
    'edge_label_logit_k5', 'edge_label_logit_k10', 'edge_label_logit_k20',
    'edge_label_variance_k5', 'edge_label_variance_k10', 'edge_label_variance_k20',
    'entropy_k5', 'entropy_k10', 'entropy_k20',
    'valid_context_coverage_k5', 'valid_context_coverage_k10', 'valid_context_coverage_k20',
    'cross_scale_probability_std', 'E0_minus_E1_k5', 'min_E0_E1_k5',
    'E0_minus_E1_k10', 'min_E0_E1_k10', 'E0_minus_E1_k20', 'min_E0_E1_k20',
)


class Branch:
    def __init__(self, docs, raw, x, y, ids, groups, scaler, pca):
        self.query = QueryModels().fit(raw, x, y, scaler)
        self.bank = Bank(docs, x, y, ids, groups)
        self.memory = Memory(self.bank)
        self.sheaf = SheafMemory(self.bank, pca)

    def transform(self, raw, ids, groups):
        if set(ids) & set(self.bank.ids) or set(groups) & set(self.bank.groups):
            raise ValueError('Read queries or their groups occur in fit memory.')
        q = unit(raw)
        base = self.query.transform(raw, q)
        rows = []
        for offset in range(0, len(q), 32):
            sl = slice(offset, offset + 32)
            geom = self.memory.geometry(q[sl], ids[sl], groups[sl])
            memory = self.memory.read(geom, alpha=0, T=0.05)
            sheaf, _ = self.sheaf.read(q[sl], ids[sl], groups[sl], lam=0.1)
            rows.append(np.column_stack([base[sl], memory['local'], memory['structure'], sheaf]))
        out = np.concatenate(rows)
        if out.shape != (len(raw), 26) or not np.isfinite(out).all():
            raise ArithmeticError('Invalid detector features.')
        return out


class SheafGuard:
    feature_names = FEATURE_NAMES

    def __init__(self):
        self.threshold = 0.0

    def fit(self, kb_embeddings, query_embeddings, labels, ids, groups, folds=None):
        with threadpool_limits(limits=4):
            return self._fit(kb_embeddings, query_embeddings, labels, ids, groups, folds)

    def _fit(self, kb_embeddings, query_embeddings, labels, ids, groups, folds):
        docs_raw = vectors(kb_embeddings, 'kb_embeddings')
        raw = vectors(query_embeddings, 'query_embeddings', docs_raw.shape[1])
        if min(docs_raw.shape) < 16:
            raise ValueError('At least 16 KB rows and embedding dimensions are required.')
        y = training_labels(labels, len(raw))
        ids = identifiers(ids, len(raw), 'ids', True)
        groups = identifiers(groups, len(raw), 'groups')
        if len(np.unique(groups)) < 3:
            raise ValueError('At least three training groups are required.')
        docs, x = unit(docs_raw), unit(raw)
        self.scaler_ = StandardScaler().fit(docs_raw)
        self.pca_ = PCA(n_components=16, svd_solver='full').fit(docs)
        self.dimension_ = raw.shape[1]
        if folds is None:
            folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(raw, y, groups))
        else:
            folds = list(folds)
        if len(folds) != 3:
            raise ValueError('The fixed procedure uses three grouped OOF folds.')
        normalized = []
        seen = np.zeros(len(y), dtype=int)
        for a, b in folds:
            a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
            if (a.ndim != 1 or b.ndim != 1 or len(a) == 0 or len(b) == 0 or
                min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= len(y) or
                len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b) or
                set(a) | set(b) != set(range(len(y)))):
                raise ValueError('Each OOF fold must partition the training rows.')
            if set(groups[a]) & set(groups[b]):
                raise ValueError('OOF query groups overlap.')
            training_labels(y[a], len(a))
            seen[b] += 1
            normalized.append((a, b))
        if not np.all(seen == 1):
            raise ValueError('Each training row must have exactly one OOF feature vector.')
        self.oof_features_ = np.empty((len(raw), 26))
        self.membership_ = []
        for a, b in normalized:
            branch = Branch(docs, raw[a], x[a], y[a], ids[a], groups[a], self.scaler_, self.pca_)
            self.oof_features_[b] = branch.transform(raw[b], ids[b], groups[b])
            self.membership_.append(dict(fit_indices=a.copy(), read_indices=b.copy(),
                fit_ids=branch.bank.ids.copy(), fit_groups=branch.bank.groups.copy(),
                read_ids=ids[b].copy(), read_groups=groups[b].copy(),
                query_nodes=len(branch.bank.x), document_seed_mass=float(branch.bank.Y[:len(docs)].sum())))
        self.head_ = meta(1).fit(self.oof_features_, y)
        self.branch_ = Branch(docs, raw, x, y, ids, groups, self.scaler_, self.pca_)
        return self

    def transform(self, embeddings, ids, groups):
        if not hasattr(self, 'branch_'):
            raise ValueError('Fit the detector before reading queries.')
        raw = vectors(embeddings, 'embeddings', self.dimension_)
        ids = identifiers(ids, len(raw), 'ids', True)
        groups = identifiers(groups, len(raw), 'groups')
        with threadpool_limits(limits=4):
            return self.branch_.transform(raw, ids, groups)

    def decision_function(self, embeddings, ids, groups):
        return self.head_.decision_function(self.transform(embeddings, ids, groups))

    def predict(self, embeddings, ids, groups):
        return (self.decision_function(embeddings, ids, groups) >= self.threshold).astype(int)
