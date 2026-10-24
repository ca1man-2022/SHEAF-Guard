import inspect
import io
import unittest
from unittest.mock import patch

import numpy as np

from examples.demo import fixture
from sheaf_guard import SheafGuard
from sheaf_guard.detector import Branch, FEATURE_NAMES
from sheaf_guard.schema import identifiers, load_embeddings, query_groups, training_labels, vectors
from sheaf_guard.utils import unit


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.branches = []

        def construct(*args, **kwargs):
            branch = Branch(*args, **kwargs)
            cls.branches.append(branch)
            return branch

        with patch('sheaf_guard.detector.Branch', side_effect=construct):
            cls.model = SheafGuard().fit(
                cls.data['kb_embeddings'], cls.data['fit_embeddings'], cls.data['fit_labels'],
                cls.data['fit_ids'], cls.data['fit_groups'],
            )
        cls.inputs = (
            cls.data['query_embeddings'], cls.data['query_ids'], cls.data['query_groups'],
        )

    def test_feature_dimensions_and_order(self):
        features = self.model.transform(*self.inputs)
        self.assertEqual(self.model.oof_features_.shape, (60, 26))
        self.assertEqual(features.shape, (8, 26))
        self.assertEqual(len(FEATURE_NAMES), 26)
        self.assertEqual(len(set(FEATURE_NAMES)), 26)
        self.assertEqual(FEATURE_NAMES[:3], ('LR_score', 'LinearSVM_score', 'RBFSVM_score'))
        self.assertEqual(FEATURE_NAMES[20:], (
            'E0_minus_E1_k5', 'min_E0_E1_k5',
            'E0_minus_E1_k10', 'min_E0_E1_k10',
            'E0_minus_E1_k20', 'min_E0_E1_k20',
        ))
        branch = self.model.branch_
        raw, ids, groups = self.inputs
        q = unit(raw)
        local = branch.memory.read(branch.memory.geometry(q, ids, groups), 0, 0.05)
        sheaf, _ = branch.sheaf.read(q, ids, groups, 0.1)
        expected = np.column_stack([
            branch.query.transform(raw, q), local['local'], local['structure'], sheaf,
        ])
        self.assertEqual(local['local'].shape[1], 4)
        self.assertEqual(local['structure'].shape[1], 13)
        self.assertEqual(sheaf.shape[1], 6)
        np.testing.assert_allclose(features, expected, atol=1e-10, rtol=1e-10)
        self.assertTrue(np.isfinite(self.model.oof_features_).all())
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(self.model.head_[-1].n_features_in_, 26)

    def test_edge_families_and_unlabelled_kb(self):
        bank = self.model.branch_.bank
        np.testing.assert_array_equal(bank.Y[:bank.nd], np.zeros((32, 2)))
        self.assertEqual(set(bank.graphs), {5, 10, 20})
        for k, graph in bank.graphs.items():
            edges = graph['meta']['edges']
            self.assertEqual(set(graph['meta']['family_counts']), {'DD', 'QQ', 'DQ'})
            for family in ('DD', 'QQ', 'DQ'):
                members = [e for e in edges if e['family'] == family]
                self.assertGreater(len(members), 0)
                np.testing.assert_allclose(sum(e['weight'] for e in members), 1.0)
                self.assertEqual(len({tuple(e['members']) for e in members}), len(members))
                for edge in members:
                    self.assertEqual(len(edge['members']), k)
                    if family == 'DD':
                        self.assertTrue(all(i < bank.nd for i in edge['members']))
                    elif family == 'QQ':
                        self.assertTrue(all(i >= bank.nd for i in edge['members']))
                    else:
                        self.assertEqual(sum(i < bank.nd for i in edge['members']), 1)

    def test_oof_excludes_read_queries_and_families_from_actual_banks(self):
        self.assertEqual(len(self.model.membership_), 3)
        self.assertEqual(len(self.branches), 4)
        seen = np.zeros(len(self.data['fit_labels']), dtype=int)
        for record, branch in zip(self.model.membership_, self.branches[:3]):
            a, b = record['fit_indices'], record['read_indices']
            seen[b] += 1
            self.assertFalse(set(branch.bank.ids) & set(record['read_ids']))
            self.assertFalse(set(branch.bank.groups) & set(record['read_groups']))
            self.assertEqual(set(branch.bank.ids), set(self.data['fit_ids'][a]))
            self.assertEqual(set(branch.bank.groups), set(self.data['fit_groups'][a]))
            self.assertEqual(len(branch.bank.bank), len(self.data['kb_embeddings']) + len(a))
            self.assertEqual(record['query_nodes'], len(a))
            self.assertEqual(record['document_seed_mass'], 0.0)
            expected = branch.transform(
                self.data['fit_embeddings'][b], self.data['fit_ids'][b], self.data['fit_groups'][b],
            )
            np.testing.assert_allclose(self.model.oof_features_[b], expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(seen, np.ones_like(seen))

    def test_self_and_same_group_reads_are_rejected(self):
        raw, ids, groups = self.inputs
        self_ids = ids.copy()
        self_ids[0] = self.data['fit_ids'][0]
        with self.assertRaises(ValueError):
            self.model.transform(raw, self_ids, groups)
        shared_groups = groups.copy()
        shared_groups[0] = self.data['fit_groups'][0]
        with self.assertRaises(ValueError):
            self.model.transform(raw, ids, shared_groups)

    def test_inductive_batch_and_single_read_parity(self):
        raw, ids, groups = self.inputs
        features = self.model.transform(raw, ids, groups)
        reversed_features = self.model.transform(raw[::-1], ids[::-1], groups[::-1])[::-1]
        singleton = self.model.transform(raw[:1], ids[:1], groups[:1])
        np.testing.assert_allclose(features, reversed_features, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(features[:1], singleton, atol=1e-10, rtol=1e-10)
        score = self.model.decision_function(raw, ids, groups)
        single_score = self.model.decision_function(raw[:1], ids[:1], groups[:1])
        np.testing.assert_allclose(score[:1], single_score, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(self.model.predict(raw, ids, groups), (score >= 0).astype(int))
        self.assertEqual(tuple(inspect.signature(self.model.transform).parameters), ('embeddings', 'ids', 'groups'))
        with self.assertRaises(TypeError):
            self.model.decision_function(raw, ids, groups, labels=np.zeros(len(raw)))

    def test_frozen_query_models(self):
        models = self.model.branch_.query.models
        self.assertEqual([m.config['family'] for m in models], ['LR', 'Linear-SVM', 'RBF-SVM'])
        self.assertEqual([m.config['C'] for m in models], [0.30000000000000004, 0.01, 100])
        self.assertEqual([m.config['prep'] for m in models], ['kb_standardized', 'kb_standardized', 'unit_raw'])
        self.assertEqual(models[1].model.kernel, 'precomputed')
        self.assertEqual(models[2].model.kernel, 'precomputed')
        fit = unit(self.data['fit_embeddings'])
        np.testing.assert_allclose(models[2].config['gamma'], 0.1 / (fit.shape[1] * fit.var()))
        self.assertEqual(self.model.pca_.n_components_, 16)
        np.testing.assert_allclose(self.model.scaler_.mean_, self.data['kb_embeddings'].mean(axis=0))

    def test_memory_and_csem_have_distinct_read_rules(self):
        branch = self.model.branch_
        q = branch.bank.docs[:1]
        geometry = branch.memory.geometry(q, np.array(['external-document-query']), np.array(['external-family']))
        selected = branch.sheaf.select(q)
        similarities = q @ branch.bank.bank.T
        for k, graph in branch.bank.graphs.items():
            members = [np.asarray(e['members']) for e in graph['meta']['edges']]
            maximum = np.column_stack([similarities[:, ix].max(axis=1) for ix in members])
            mean = np.column_stack([similarities[:, ix].mean(axis=1) for ix in members])
            expected_selection = np.argsort(-maximum, axis=1, kind='stable')[:, :8]
            np.testing.assert_array_equal(selected[k], expected_selection)
            np.testing.assert_allclose(geometry['edge'][k]['relevance'], (mean + maximum) / 2, atol=1e-12)
            expected_candidates = np.array([[bool(set(ix) & set(geometry['ni'][0])) for ix in members]])
            np.testing.assert_array_equal(geometry['edge'][k]['candidate'], expected_candidates)
            seed_mass = (graph['H'].T @ branch.bank.Y).sum(axis=1)
            valid = geometry['edge'][k]['candidate'] & (seed_mass[None] > 1e-12)
            self.assertTrue(any(graph['meta']['edges'][i]['family'] == 'DD' for i in selected[k][0]))
            self.assertTrue(np.any(~valid[0, selected[k][0]]))

    def test_document_only_contexts_use_global_class_prototypes(self):
        bank = self.model.branch_.bank
        sheaf = self.model.branch_.sheaf
        coordinates = sheaf.u[bank.nd:]
        means = np.stack([coordinates[bank.y == c].mean(axis=0) for c in (0, 1)])
        for k, graph in bank.graphs.items():
            dd = np.array([i for i, e in enumerate(graph['meta']['edges']) if e['family'] == 'DD'])
            self.assertGreater(len(dd), 0)
            np.testing.assert_array_equal(sheaf.counts[k][dd], np.zeros((len(dd), 2)))
            expected = np.einsum('eij,cj->eci', sheaf.maps[k][dd], means)
            np.testing.assert_allclose(sheaf.boundaries[k][dd], expected, atol=1e-10, rtol=1e-10)
            for e in dd:
                ix = sheaf.members[k][e]
                np.testing.assert_array_equal(bank.Y[ix], np.zeros((len(ix), 2)))

    def test_explicit_oof_folds_cannot_split_families(self):
        d = self.data
        ix = np.arange(len(d['fit_labels']))
        folds = [(np.setdiff1d(ix, ix[j::3]), ix[j::3]) for j in range(3)]
        with self.assertRaisesRegex(ValueError, 'OOF query groups overlap'):
            SheafGuard().fit(d['kb_embeddings'], d['fit_embeddings'], d['fit_labels'],
                             d['fit_ids'], d['fit_groups'], folds=folds)

    def test_invalid_detector_inputs(self):
        with self.assertRaises(ValueError):
            SheafGuard().transform(*self.inputs)
        raw, ids, groups = self.inputs
        with self.assertRaises(ValueError):
            self.model.transform(raw[:, :-1], ids, groups)
        with self.assertRaises(ValueError):
            self.model.transform(np.zeros_like(raw), ids, groups)
        with self.assertRaises(ValueError):
            self.model.transform(raw, np.repeat(ids[:1], len(ids)), groups)
        d = self.data
        with self.assertRaises(ValueError):
            SheafGuard().fit(d['kb_embeddings'], d['fit_embeddings'], np.zeros(60), d['fit_ids'], d['fit_groups'])
        with self.assertRaises(ValueError):
            SheafGuard().fit(d['kb_embeddings'][:8], d['fit_embeddings'], d['fit_labels'], d['fit_ids'], d['fit_groups'])


class SchemaTests(unittest.TestCase):
    def archive(self, data):
        buffer = io.BytesIO()
        np.savez(buffer, **data)
        buffer.seek(0)
        return buffer

    def test_valid_schema_uses_no_pickle(self):
        data = fixture()
        loader = np.load
        with patch('sheaf_guard.schema.np.load', wraps=loader) as checked:
            restored = load_embeddings(self.archive(data))
            self.assertIs(checked.call_args.kwargs['allow_pickle'], False)
        self.assertEqual(set(restored), set(data))
        for name in data:
            np.testing.assert_array_equal(restored[name], data[name])

    def test_schema_rejects_object_arrays_and_missing_fields(self):
        data = fixture()
        data['query_ids'] = data['query_ids'].astype(object)
        with self.assertRaises(ValueError):
            load_embeddings(self.archive(data))
        data = fixture()
        del data['fit_labels']
        with self.assertRaises(ValueError):
            load_embeddings(self.archive(data))

    def test_schema_rejects_overlap_and_invalid_values(self):
        data = fixture()
        data['query_groups'][0] = data['fit_groups'][0]
        with self.assertRaises(ValueError):
            load_embeddings(self.archive(data))
        for invalid in (np.zeros((2, 4)), np.full((2, 4), np.nan), np.ones(4), np.empty((0, 4))):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                vectors(invalid, 'vectors')
        with self.assertRaises(ValueError):
            identifiers(np.array(['same', 'same']), 2, 'ids', unique=True)
        with self.assertRaises(ValueError):
            training_labels(np.array([0, 2]), 2)
        with self.assertRaises(ValueError):
            training_labels(np.array([1, 1]), 2)

    def test_normalized_query_grouping(self):
        groups = query_groups(['  MODEL\tCoverage ', 'model coverage', 'ＭＯＤＥＬ coverage'])
        self.assertEqual(len(set(groups)), 1)
        self.assertNotEqual(query_groups(['model scope'])[0], groups[0])


if __name__ == '__main__':
    unittest.main()
