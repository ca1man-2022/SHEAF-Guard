# SHEAF-Guard 

## Included components

| Component | Description |
|---|---|
| `sheaf_guard/energy.py` | Standalone adaptation of single-query, single-scale extension and energy readout. |
| `sheaf_guard/data_io.py` | JSONL reader and display-schema validator. |
| `examples/demo_extension.py` | Executable calculation on explicitly synthetic tensors. |
| `examples/inspect_data.py` | Schema and actual sample-availability inspection. |
| `data/` | Dataset-level counts, schema and source references; no real query text. |
| `tests/test_preview.py` | Numerical and data-interface unit tests. |

## Running the preview

From this directory, with Python and NumPy available:

```bash
python -B -m examples.demo_extension
python -B -m examples.inspect_data
python -B -m unittest discover -s tests -v
```

The tested versions are recorded in `requirements.txt`. No model download,
network access or API credentials are required. The tests check this excerpt,
not the paper's benchmark results.

## Numerical scope

Inputs are an existing query coordinate `u` of shape `(d,)`, context maps `R`
of shape `(M,d,d)`, class-conditional boundaries `b` of shape `(M,2,d)` and
a positive fidelity coefficient. Class 0 is ID and class 1 is OOD.

For each class, the routine minimizes
`fidelity * ||z-u||^2 + mean_e ||R[e] z-b[e,c]||^2`.
It solves the shared positive-definite linear system with NumPy Cholesky
factorization. The returned features are `E_ID - E_OOD` and
`min(E_ID, E_OOD)`, without calibration or classification.

The frozen calculation uses uniform context weights. Optional supplied
weights must equal `1/M`; nonuniform weights are rejected rather than
silently changing the objective. The demonstration uses tiny invented
arrays, not trained memories, real embeddings or a complete OOD detector.

## Data

Source attribution and redistribution clearance are incomplete; see
`data/README.md` and `data/sources.json`. Two separately named synthetic
fixtures exercise the interface and are not benchmark examples.

`data/schema.json` is a preview display format, not the original CSV schema.
ID is `0`, OOD is `1`.