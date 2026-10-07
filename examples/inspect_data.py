"""Inspect schema and actual sample availability without network access."""
import json
from pathlib import Path
from sheaf_guard_preview.data_io import read_jsonl


def main():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root/"data/schema.json").read_text())
    sources = json.loads((root/"data/sources.json").read_text())
    samples = root/"data/examples.jsonl"
    count = len(read_jsonl(samples, sources)) if samples.exists() else 0
    print(json.dumps(dict(required_fields=schema["required"], real_samples=count,
                          status="samples" if count else "schema-and-source-only"), indent=2))


if __name__ == "__main__":
    main()
