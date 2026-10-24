import json
import numpy as np
from sheaf_guard import SheafGuard


def fixture():
    rng = np.random.default_rng(73)
    dimension = 24
    directions = rng.normal(size=(2, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    docs = directions[0] + 0.65 * rng.normal(size=(32, dimension))
    labels = np.tile([0, 1], 30)
    queries = directions[labels] + 0.65 * rng.normal(size=(60, dimension))
    reads = directions[np.arange(8) % 2] + 0.65 * rng.normal(size=(8, dimension))
    return dict(kb_embeddings=docs, fit_embeddings=queries, fit_labels=labels,
                fit_ids=np.array([f'fit-{i:03d}' for i in range(60)]),
                fit_groups=np.array([f'family-{i // 2:03d}' for i in range(60)]),
                query_embeddings=reads, query_ids=np.array([f'read-{i:03d}' for i in range(8)]),
                query_groups=np.array([f'new-family-{i:03d}' for i in range(8)]))


def main():
    data = fixture()
    model = SheafGuard().fit(data['kb_embeddings'], data['fit_embeddings'], data['fit_labels'],
                            data['fit_ids'], data['fit_groups'])
    features = model.transform(data['query_embeddings'], data['query_ids'], data['query_groups'])
    scores = model.head_.decision_function(features)
    print(json.dumps(dict(kb_documents=len(data['kb_embeddings']), fit_queries=len(data['fit_labels']),
        oof_shape=list(model.oof_features_.shape), query_feature_shape=list(features.shape),
        edge_families={str(k): g['meta']['family_counts'] for k, g in model.branch_.bank.graphs.items()},
        scores=scores.tolist(), predictions=(scores >= 0).astype(int).tolist(),
        all_finite=bool(np.isfinite(features).all() and np.isfinite(scores).all())), indent=2))


if __name__ == '__main__':
    main()
