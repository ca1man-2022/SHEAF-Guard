# SxHxExAxFx-xGxuxaxrxdx


## Quick start

From the repository root:

```bash
python -m pip install -r requirements.txt
python -B -m examples.demo_extension
python -B -m examples.inspect_data
python -B -m unittest discover -s tests -v
```

`demo_extension` evaluates the two class hypotheses on synthetic inputs.
`inspect_data` prints the query-record schema metadata. The tests check the
numerical routines and the data reader.

## CSEM extension

```python
from sheaf_guard_preview import extension, compatibility_readout

result = extension(query, maps, boundaries, fidelity)
features = compatibility_readout(result["energies"])
```

| Input | Shape | Description |
| --- | --- | --- |
| `query` | `(d,)` | Query coordinate $u$ |
| `maps` | `(M, d, d)` | Context restriction matrices $R_e$ |
| `boundaries` | `(M, 2, d)` | Boundary values $b_{e,c}$ for ID ($c=0$) and OOD ($c=1$) |
| `fidelity` | scalar | Positive coefficient $\lambda$ |

At one memory scale, the solver minimizes

$$
E_c=\min_z\left[\lambda\lVert z-u\rVert_2^2+
\frac{1}{M}\sum_{e=1}^{M}\lVert R_ez-b_{e,c}\rVert_2^2\right].
$$

It solves the shared positive-definite system by Cholesky factorization in
float64 and returns the class energies, extension states, and diagnostic arrays.
Context weights are uniform; the optional `weights` argument accepts only
$w_e=1/M$. The compatibility readout produces the two downstream features
$[E_0-E_1,\min(E_0,E_1)]$.

## Data

The [data guide](data/README.md) lists dataset statistics, source references, and
the JSONL record format. Two synthetic records are included for reader tests;
benchmark datasets are obtained separately from the listed sources.

## Attribution

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency, data, and
code-license information.
