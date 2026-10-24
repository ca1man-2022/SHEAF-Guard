MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
MODEL_REVISION = 'e8c3b32edf5434bc2275fc9bab85f82640a19130'


class Encoder:
    def __init__(self, snapshot_path=None):
        from pathlib import Path
        from sentence_transformers import SentenceTransformer
        location = MODEL_NAME
        if snapshot_path is not None:
            path = Path(snapshot_path)
            if not path.is_dir() or path.name != MODEL_REVISION:
                raise ValueError('Expected the local directory of the frozen encoder revision.')
            location = str(path)
        self.model = SentenceTransformer(location, revision=MODEL_REVISION,
                                        local_files_only=True, device='cpu')
        self.model.max_seq_length = 384

    def encode(self, texts):
        from threadpoolctl import threadpool_limits
        import torch
        old = torch.get_num_threads()
        try:
            torch.set_num_threads(min(old, 4))
            with threadpool_limits(limits=4):
                result = self.model.encode(list(texts), batch_size=32, normalize_embeddings=False,
                                           convert_to_numpy=True, show_progress_bar=False)
        finally:
            torch.set_num_threads(old)
        if result.ndim != 2 or result.shape[1] != 768:
            raise ValueError('Expected 768-dimensional MPNet embeddings.')
        return result
