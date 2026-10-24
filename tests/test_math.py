import unittest

import numpy as np

from sheaf_guard.csem import extension, restriction


class SheafExtensionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(73)
        self.m = 8
        self.d = 4
        self.lam = 0.1
        self.u = self.rng.normal(size=self.d)
        self.R = np.stack([
            restriction(
                self.rng.normal(size=(12, self.d)) * np.array([0.5, 1.0, 2.0, 4.0]),
                1e-3,
            )[0]
            for _ in range(self.m)
        ])
        self.b = self.rng.normal(size=(self.m, 2, self.d))

    def test_nonidentity_solution_matches_least_squares(self):
        result = extension(self.u, self.R, self.b, self.lam)
        design = np.concatenate([
            np.sqrt(self.lam) * np.eye(self.d),
            self.R.reshape(-1, self.d) / np.sqrt(self.m),
        ])
        self.assertGreater(float(np.max(np.abs(self.R - np.eye(self.d)))), 0.1)
        for c in range(2):
            target = np.concatenate([
                np.sqrt(self.lam) * self.u,
                self.b[:, c].reshape(-1) / np.sqrt(self.m),
            ])
            optimum = np.linalg.lstsq(design, target, rcond=None)[0]
            energy = np.sum((design @ optimum - target) ** 2)
            np.testing.assert_allclose(result['z'][c], optimum, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(result['E'][c], energy, atol=1e-10, rtol=1e-10)

    def test_objective_closed_form_and_gradient(self):
        result = extension(self.u, self.R, self.b, self.lam)
        residual = np.empty((self.m, 2))
        for e in range(self.m):
            for c in range(2):
                residual[e, c] = np.sum((self.R[e] @ result['z'][c] - self.b[e, c]) ** 2)
        displacement = np.sum((result['z'] - self.u) ** 2, axis=1)
        direct = self.lam * displacement + residual.mean(axis=0)
        np.testing.assert_allclose(result['residual'], residual, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result['E'], direct, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result['E'], result['closed'], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result['gradient'], 0.0, atol=1e-10, rtol=0)

    def test_positive_definite_system_and_normalized_restrictions(self):
        for matrix in self.R:
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12, rtol=0)
            self.assertGreater(float(np.linalg.eigvalsh(matrix).min()), 0.0)
            np.testing.assert_allclose(np.sum(matrix ** 2), self.d, atol=1e-10, rtol=1e-10)
        result = extension(self.u, self.R, self.b, self.lam)
        np.testing.assert_allclose(result['A'], result['A'].T, atol=1e-12, rtol=0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(result['A']).min()), self.lam - 1e-12)

    def test_identity_mean_variance_decomposition(self):
        identity = np.repeat(np.eye(self.d)[None], self.m, axis=0)
        result = extension(self.u, identity, self.b, self.lam)
        mean = self.b.mean(axis=0)
        variance = np.sum((self.b - mean) ** 2, axis=2).mean(axis=0)
        expected = variance + self.lam / (1.0 + self.lam) * np.sum((self.u - mean) ** 2, axis=1)
        optimum = (self.lam * self.u + mean) / (1.0 + self.lam)
        np.testing.assert_allclose(result['E'], expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result['z'], optimum, atol=1e-10, rtol=1e-10)

    def test_class_swap_reverses_energy_and_margin(self):
        result = extension(self.u, self.R, self.b, self.lam)
        swapped = extension(self.u, self.R, self.b[:, ::-1], self.lam)
        np.testing.assert_allclose(swapped['E'], result['E'][::-1], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(swapped['z'], result['z'][::-1], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(swapped['E'][0] - swapped['E'][1], -(result['E'][0] - result['E'][1]), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(swapped['E'].min(), result['E'].min(), atol=1e-10, rtol=1e-10)

    def test_context_permutation_and_uniform_repetition(self):
        result = extension(self.u, self.R, self.b, self.lam)
        order = self.rng.permutation(self.m)
        permuted = extension(self.u, self.R[order], self.b[order], self.lam)
        repeated = extension(self.u, np.repeat(self.R, 2, axis=0), np.repeat(self.b, 2, axis=0), self.lam)
        for alternative in (permuted, repeated):
            np.testing.assert_allclose(alternative['E'], result['E'], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(alternative['z'], result['z'], atol=1e-10, rtol=1e-10)

    def test_fixed_query_has_no_lower_energy(self):
        result = extension(self.u, self.R, self.b, self.lam)
        fixed = extension(self.u, self.R, self.b, self.lam, no_extension=True)
        np.testing.assert_array_equal(fixed['z'], np.broadcast_to(self.u, (2, self.d)))
        self.assertTrue(np.all(result['E'] <= fixed['E'] + 1e-10))
        np.testing.assert_allclose(fixed['E'], fixed['residual'].mean(axis=0), atol=1e-10, rtol=1e-10)

    def test_degenerate_and_repeated_points(self):
        point = self.rng.normal(size=(1, self.d))
        for points in (point, np.repeat(point, 8, axis=0), np.zeros((8, self.d))):
            matrix, fallback = restriction(points, 1e-3)
            self.assertTrue(fallback)
            np.testing.assert_array_equal(matrix, np.eye(self.d))
            result = extension(self.u, matrix[None], self.b[:1], self.lam)
            self.assertTrue(np.isfinite(result['E']).all())
        rank_one = np.outer(np.arange(6.0), np.arange(1.0, self.d + 1.0))
        original, fallback = restriction(rank_one, 1e-3)
        duplicated, duplicated_fallback = restriction(np.repeat(rank_one, 3, axis=0), 1e-3)
        self.assertFalse(fallback)
        self.assertFalse(duplicated_fallback)
        self.assertGreater(float(np.linalg.eigvalsh(original).min()), 0.0)
        np.testing.assert_allclose(duplicated, original, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(np.sum(original ** 2), self.d, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
