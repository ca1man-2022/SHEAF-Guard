# SxHxExAxFx-xGxuxaxrxd


## Installation

Use Python 3.11 or later. From this directory:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m examples.demo
python -m unittest discover -s tests -v
```

The deterministic example fits all three query classifiers, builds DD/QQ/DQ
hyperedges at scales 5, 10, and 20, and computes the 20 memory and six
compatibility features. It prints finite scores for eight unseen synthetic
queries. Its 24-dimensional random vectors exercise the complete detector;
they are not benchmark embeddings or evidence of detection accuracy.

## Inputs

`SheafGuard.fit(kb_embeddings, query_embeddings, labels, ids, groups)` accepts
raw embedding matrices and training labels (`ID=0`, `OOD=1`). IDs are unique
strings; groups bind duplicates and known paraphrase families. Three grouped
folds provide training features without self-label exposure. Test labels are
not accepted by the prediction interface. `decision_function(embeddings,
ids, groups)` returns OOD risk; the fixed classifier predicts OOD at scores
greater than or equal to zero. See [SCHEMA.md](SCHEMA.md) for array fields.

The text encoder interface uses a frozen 768-dimensional all-mpnet-base-v2
revision from an existing local cache. Its optional dependencies are listed
in `requirements-encoder.txt`; the example requires neither weights nor network
access. Real experiments require separately obtained Substance Use,
StackExchange, or MS MARCO data under their respective access terms.
