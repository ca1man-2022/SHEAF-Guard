# Data

This directory provides dataset statistics, upstream source links, and the
JSONL schema used by the example reader. Obtain the benchmark datasets from
[sources.json](sources.json).

## Datasets

| Dataset | Target domains | Test instances | OOD source |
| --- | ---: | ---: | --- |
| StackExchange | 8 | 18,060 | Cross-domain queries |
| MS MARCO | 6 | 11,744 | Cross-domain queries |
| Substance Use | 1 | 126 | Cross-domain queries |
| Substance Use LLM-attack | 1 | 108 | LLM-generated attack queries |

Counts are summed over target-task test sets; a query may occur in more than
one task. The clean setting covers 15 domains, with SU LLM-Attack reported
separately. Labels follow the source-domain and attack protocols: ID = 0 and
OOD = 1.

## JSONL records

The example reader uses [schema.json](schema.json):

| Field | Type | Description |
| --- | --- | --- |
| `example_id` | string | Unique record identifier |
| `task` | string | Target detection task |
| `target_domain` | string | Domain of the target KB |
| `source_domain` | string, optional | Source domain of the query |
| `query` | string | Query text |
| `label` | integer | ID = 0; OOD = 1 |
| `label_provenance` | string | Label source |
| `source_ref` | string | Key in the source registry |

`examples/schema_fixture.synthetic.jsonl` contains two synthetic records for
interface testing. The reader validates fields, source keys, and unique IDs.
