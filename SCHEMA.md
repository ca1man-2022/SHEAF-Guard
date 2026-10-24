# Embedding inputs

`sheaf_guard.encoder.Encoder()` uses the standard local model cache.
For a separately stored snapshot, pass `Encoder(snapshot_path=...)` with
the directory named after revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
The interface never downloads weights automatically.

`load_embeddings(path)` reads a NumPy NPZ file without pickle. All embedding
matrices must be finite, nonzero by row, and share their last dimension.
Raw, unnormalized encoder outputs are expected. The detector performs its
own KB-fitted preprocessing. At least 16 KB rows and 16 dimensions are needed.

| Field | Shape | Meaning |
| --- | --- | --- |
| `kb_embeddings` | D × p | Fixed KB document embeddings, in stable order |
| `fit_embeddings` | N × p | Training query embeddings |
| `fit_labels` | N | ID=0, OOD=1; both classes are required |
| `fit_ids` | N | Unique string identifiers |
| `fit_groups` | N | Duplicate/paraphrase-family identifiers |
| `query_embeddings` | Q × p | Unseen query embeddings |
| `query_ids` | Q | Unique strings, disjoint from training IDs |
| `query_groups` | Q | Strings, disjoint from training groups |

KB documents are unlabeled. Answers, label explanations, and evaluation
labels are not model inputs. `query_groups(texts)` hashes NFKC-normalized,
case-folded, whitespace-normalized texts; known paraphrase families must be
grouped separately because exact text normalization does not identify them.

`SheafGuard.feature_names` records the exact 26-column order. Columns 1–20
are structured-memory features; the remaining columns contain the ID-minus-OOD
energy and minimum class energy, in that order, at scales 5, 10, and 20.
The memory branch pools incident candidate edges; the compatibility branch
independently selects eight edges per scale by maximum member cosine.

`fit(..., folds=...)` optionally accepts three fixed `(fit_indices,
held_out_indices)` pairs over the training rows. Each pair must partition
those rows, keep groups intact, and retain both fit classes. Every row must
be held out exactly once. Standardization and the fusion head use only
these out-of-fold features. The exact precomputed-kernel SVM implementation
rejects kernels exceeding 1 GiB rather than silently subsampling data.
