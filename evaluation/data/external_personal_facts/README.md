# External personal-fact labels for writer development

The [Personal Facts (MSC) dataset](https://huggingface.co/datasets/adugeen/personal-facts-msc)
contains 2,779 single-annotator fact examples (2,223 train, 556 test), each
paired with dialogue context and labels for topic, time, referent, duration,
validity, and follow-up. The CI importer pins revision
`22854562a2934ea07983ecda4f4fd8ff3aef2e85`, downloads **train only**,
audits the schema, and saves counts and a source checksum as an artifact. The
source is CC BY-SA 4.0; cite both its card and original MSC. Its raw Parquet
is not copied into this repository.

The pinned training file has 2,223 rows; 1,368 meet the dataset's Self,
Long-term, and valid-label filter. The committed `train_audit_v1.json`
records the exact source checksum and category counts. Those 1,368 are
candidate facts, not 1,368 progression routing decisions.

Use it for supervised *fact characterization*: identify self versus other,
long versus short duration, and invalid candidate facts. The supplied `text`
is a candidate paraphrase grounded in `context`, not an adjudicated before/after
state. These labels do not say which process a fact belongs to, whether it
starts a root or child, whether a phase changed, or whether multiple processes
should be updated. They are also derived from MSC and lack conversation IDs;
do not count them as independent evidence of performance on our MSC attachment
pairs. LoCoMo and LongMemEval stay held out.

Official MSC also has persona-summary training annotations; these could help
train a grounded summarizer, with source alignment and independent review.
Neither source supplies a gold progression graph. Full-sequence routing still
needs annotation or a separately reviewed weak-labeling process.
