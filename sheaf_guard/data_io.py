"""Read the preview display schema, not the original benchmark format."""
import json

REQUIRED = {"example_id", "task", "target_domain", "query", "label",
            "label_provenance", "source_ref"}
OPTIONAL = {"source_domain"}


def validate_record(record, source_refs=None):
    """Validate schema only; neither licensing nor answerability is inferred."""
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    if not REQUIRED <= record.keys() or record.keys() - REQUIRED - OPTIONAL:
        raise ValueError("record fields do not match the preview schema")
    if type(record["label"]) is not int or record["label"] not in (0, 1):
        raise ValueError("label must be integer 0 or 1")
    for key in record.keys() - {"label"}:
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError("text fields must be nonempty strings")
    if source_refs is not None and record["source_ref"] not in source_refs:
        raise ValueError("unknown source_ref")
    return record


def read_jsonl(path, source_refs=None):
    """Read UTF-8 records; reject duplicate display IDs and blank lines."""
    records, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                record = validate_record(json.loads(line), source_refs)
                if record["example_id"] in seen:
                    raise ValueError("duplicate example_id")
            except (ValueError, TypeError):
                raise ValueError(f"invalid preview record at line {number}") from None
            seen.add(record["example_id"])
            records.append(record)
    return records
