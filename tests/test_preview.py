"""Synthetic-only tests; no private imports, training or network."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from examples.demo_extension import toy_inputs
from sheaf_guard_preview import extension, compatibility_readout
from sheaf_guard_preview.data_io import read_jsonl, validate_record


class NumericalTests(unittest.TestCase):
    def test_stationarity_and_objective(self):
        v = extension(*toy_inputs())
        np.testing.assert_allclose(v['gradient'], 0, atol=1e-12)
        np.testing.assert_allclose(v['energies'], v['closed'], atol=1e-12)
        self.assertTrue((v['energies'] >= 0).all())

    def test_permutation(self):
        u, R, b, lam = toy_inputs()
        a, z = extension(u, R, b, lam), extension(u, R[::-1], b[::-1], lam)
        np.testing.assert_allclose(a['states'], z['states'], atol=1e-12)
        np.testing.assert_allclose(a['energies'], z['energies'], atol=1e-12)

    def test_readout_and_class_swap(self):
        u, R, b, lam = toy_inputs()
        E = extension(u, R, b, lam)['energies']
        np.testing.assert_array_equal(compatibility_readout(E), [E[0]-E[1], min(E)])
        np.testing.assert_allclose(extension(u,R,b[:,::-1],lam)['energies'], E[::-1])

    def test_shapes_finite_fidelity(self):
        args = list(toy_inputs())
        for index, bad in [(0, [[1., 2.]]), (1, np.zeros((0,2,2))),
                           (2, np.zeros((2,1,2))), (0, [np.nan,0]),
                           (1, np.full((2,2,2),np.inf)), (3, 0), (3,-1),
                           (3,np.nan), (3,[1.]), (0,[1j,0])]:
            values = copy.deepcopy(args); values[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError):
                extension(*values)

    def test_weights(self):
        np.testing.assert_array_equal(extension(*toy_inputs())['energies'],
                                      extension(*toy_inputs(), weights=[.5,.5])['energies'])
        for w in [[-.1,1.1], [0,0], [.3,.7], [1], [np.nan,.5]]:
            with self.assertRaises(ValueError):
                extension(*toy_inputs(), weights=w)

    def test_invalid_readout(self):
        for E in [[1], [-1,2], [np.inf,1]]:
            with self.assertRaises(ValueError): compatibility_readout(E)


class DataTests(unittest.TestCase):
    def fixture(self):
        return Path(__file__).resolve().parents[1]/'examples/schema_fixture.synthetic.jsonl'

    def test_fixture(self):
        records = read_jsonl(self.fixture(), {'synthetic'})
        self.assertEqual([r['label'] for r in records], [0,1])

    def test_invalid_records(self):
        record = read_jsonl(self.fixture())[0]
        for bad in [True, '0', 2]:
            with self.assertRaises(ValueError): validate_record(dict(record,label=bad))
        with self.assertRaises(ValueError): validate_record(dict(record,extra='no'))
        with self.assertRaises(ValueError): validate_record(dict(record,query=' '))
        with self.assertRaises(ValueError): validate_record(record, {'unknown'})

    def test_duplicate_and_malformed(self):
        line = self.fixture().read_text().splitlines()[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'invalid.jsonl'
            for content in [line+'\n'+line, '{bad json}', '\n']:
                path.write_text(content)
                with self.assertRaises(ValueError): read_jsonl(path)


if __name__ == '__main__':
    unittest.main()
