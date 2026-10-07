# Data interface and availability

`sources.json` provides upstream acquisition references, not a new license.
`dataset_summary.json` gives dataset-level counts from existing evaluations.
These count target-task test instances, not globally unique query strings.
The available clean pool spans 15 domains; this is not the full 16-domain
author setting. COVID data is not represented in this summary.

`schema.json` defines a small display interface, not the original CSV format.
Label 0 means ID and label 1 means OOD under the source task protocol;
source-protocol labels are not independent human answerability annotations.
`source_ref` must resolve to a source record. Display IDs are not split IDs.

The two records in `examples/schema_fixture.synthetic.jsonl` are invented
interface fixtures, not benchmark samples, reviewed labels or evaluation data.
