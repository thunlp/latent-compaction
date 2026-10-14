# Adaptation

## New datasets

Write one UTF-8 JSON per document:

```json
{
  "format_version": 1,
  "context": "The document text, excluding downstream questions.",
  "train_queries": [],
  "validation_queries": [],
  "test_queries": ["\nQuestion: A held-out question?\nAnswer:"],
  "metadata": {
    "task": "my_dataset",
    "record_id": "document-00000",
    "max_length": 4096,
    "references": [["accepted answer", "another accepted answer"]],
    "metric_kind": "any",
    "generation_tokens": 128
  }
}
```

Use the shared runner for your new split:

```bash
python run.py --method latent --split your-document.json --model MODEL \
  --budget 128 --output outputs/custom
```

Training reads only `context` and uses the fixed bank in `prompts.py`; downstream
questions are used only for scoring.
Preserve boundary whitespace: document plus query should reconstruct the intended
raw input. Do not apply a chat template unless explicitly studying a new protocol.

Available metrics are `phonebook` (first ten-digit number), `all`/`coverage` (mean
case-insensitive answer coverage), `any` (any accepted reference), `exact` (whole
trimmed answer), `niah` (normalized containment), and `longbench` (official metrics).
For new metrics, add a branch to `evaluation.metric` and a small semantic test.
Avoid selecting prompts, schedules, or checkpoints using test questions. Report
all adaptation choices and distinguish them from paper reproduction.

## New models

Use a full Hub ID or local path. `models.load_model` accepts `qwen3` or `llama`
model types and installs adapters for their attention classes. MiniCPM5 uses the
Llama path. Gated checkpoints require accepting their license and authenticating
outside the repository. Pin checkpoint/tokenizer revisions using a downloaded
snapshot, and use that snapshot for both training and evaluation.

For another architecture, implement its query/key projections, any Q/K
normalization, RoPE, grouped-query expansion, cache offset, and causal continuation
attention in `models.py`. Preserve the document's original position count after
compaction. The combination weights must use the two log normalizers. Validate
native full-cache parity, gradient flow through keys/values/biases, masking,
serialization, and teacher-forced continuation alignment before benchmarking.
Do not infer compatibility solely from a similar configuration name.

## Training controls

Copy `configs/paper.json` and change explicit `FitConfig` fields. `budget` counts
average pairs per layer/KV head; `soft_slots` is a per-head count. Global preserver
allocation gives uneven head lengths, which are padded and masked. Padding and
bias storage mean physical bytes differ from logical slot counts.

Set stage durations, query sampling, batch sizes and learning rates explicitly
for a new workload. Initialization lives
in `selection.py`; freeze schedules live in `memory.py`; the grouped distillation
objective lives in `methods/latent.py`. The API accepts an alternate prompt list for
research; the paper recipes always use all 480 prompts. Changes to initialization,
bias freezing or attention mixing require a separately labeled experiment.
