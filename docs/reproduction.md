# Reproduce the paper

All commands use `run.py` and the same implementations in
`latent_compaction/methods/`. A dataset, model, context length, and compression
ratio are arguments, not separate copies of a method. Generated data and results
live under the Git-ignored `data/` and `outputs/` directories.

## Shared settings

The default `configs/paper.json` uses 1,000 attention-fitting updates per layer
and 1,000 joint-refinement updates. Both stages use the same 480 generic source
prompts, with greedy teacher continuations of up to 24 tokens. Stage 1 samples
up to 128 attention positions per update from 4,096 evenly spaced captured
positions. Stage 2 samples four rollouts per update. AdamW uses learning rates
0.003 and 0.0001, zero weight decay, default momentum/epsilon, and gradient clipping
at norm one. Model weights are frozen; CUDA model computation uses BF16 and
cache parameters use FP32. One quarter of the source budget is learned entries.

Evaluation questions are excluded from cache construction. Each fitted cache or
summary is reused for every held-out question associated with its document.
All scores are saved as fractions in [0, 1]; multiply by 100 for percentages.

## Prepare the datasets once

Use the Qwen3-1.7B tokenizer to construct the reference documents. Reuse these
same document texts when changing reader models. The preparer saves immutable
split hashes, record identities, and paired optimization seeds in each manifest.

### Phonebook Lookup

```bash
python prepare.py phonebook --model Qwen/Qwen3-1.7B \
  --lengths 4096 8192 16384 32768 --samples 20 --output data/phonebook
```

Documents contain synthetic 10-character uppercase names and 10-digit numbers.
The seed is `918000 + nominal_length * 100 + document_index`; lookup targets use
seed + 1. Each document has 20 distinct lookup questions. Recall checks the first
standalone ten-digit number in the answer. Average questions within a document,
then average documents.

### RULER

```bash
git clone https://github.com/NVIDIA/RULER.git source-download/RULER
git -C source-download/RULER checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
python -m pip install wonderwords nltk tenacity html2text beautifulsoup4
python -m nltk.downloader punkt punkt_tab
(cd source-download/RULER/scripts/data/synthetic/json && bash download_qa_dataset.sh)
```

Run `download_paulgraham_essay.py` from the same upstream JSON directory for the
essay haystacks. Then prepare all 13 tasks at the four context lengths:

```bash
python prepare.py ruler --upstream source-download/RULER \
  --model Qwen/Qwen3-1.7B --lengths 4096 8192 16384 32768 \
  --samples 20 --output data/ruler
```

Generation uses seed 42 and the official task-specific output caps. Raw input
boundaries are preserved. Scores use case-insensitive answer coverage for
multi-answer tasks and any-reference containment for QA. Aggregate documents
within each task, then equally weight the 13 tasks.

### LongBench v1

```bash
python prepare.py longbench --model Qwen/Qwen3-1.7B \
  --samples 20 --output data/longbench
```

The preparer downloads `data.zip` from dataset revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672` and verifies its recorded checksum.
For offline use, pass `--data-zip /path/to/data.zip`. It selects 20 records per task
with `random.Random(42)`, covering 16 English/code tasks. Official prompt templates,
metrics, and generation caps are included with their upstream license.

The input cap is 16,384 tokens. If necessary, remove the middle of the context
while preserving both ends and the complete query suffix. The compression budget
is `max(4, floor(context_tokens / ratio / 4) * 4)`. RULER and Phonebook use nominal
context length divided by compression ratio. Average document scores per task,
then macro-average tasks.

## RULER and Phonebook compression curves

Run all methods at 16K context and ratios 2× through 256×. The same loop works
for both datasets; only `--manifest` and the output prefix change:

```bash
for dataset in ruler phonebook; do
  for method in latent kvzip attention_matching snapkv pyramidkv summary; do
    python run.py --method "$method" --model Qwen/Qwen3-1.7B \
      --manifest "data/$dataset/manifest.json" --lengths 16384 \
      --ratios 2 4 8 16 32 64 128 256 --output "outputs/$dataset/$method"
  done
done
```

Every method's evaluation also computes the full-cache reference, so no additional
uncompressed run is required for those scores.

## RULER and Phonebook context-length curves

Use the same code, manifests, and output locations. Select the four context
lengths at 16× and 32× compression; completed overlapping cells are reused:

```bash
for dataset in ruler phonebook; do
  for method in latent kvzip attention_matching snapkv pyramidkv summary; do
    python run.py --method "$method" --model Qwen/Qwen3-1.7B \
      --manifest "data/$dataset/manifest.json" --lengths 4096 8192 16384 32768 \
      --ratios 16 32 --output "outputs/$dataset/$method"
  done
done
```

## LongBench comparison across models

Use the same prepared document cohort for Qwen3-0.6B, 1.7B, and 4B. Change only
the reader and the output location:

```bash
for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  for method in latent kvzip attention_matching snapkv pyramidkv summary; do
    python run.py --method "$method" --model "Qwen/$model" \
      --manifest data/longbench/manifest.json --ratios 4 8 16 32 \
      --output "outputs/longbench/$model/$method"
  done
done
```

## Phonebook comparison across model families

Reuse the prepared Phonebook cohort. Model IDs accepted by the same runner
include `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`,
`openbmb/MiniCPM5-1B`, and `openbmb/MiniCPM5-2B`. For each model and method, run
the compression and context-length commands above with a separate output prefix.
For example:

```bash
python run.py --method latent --model openbmb/MiniCPM5-2B \
  --manifest data/phonebook/manifest.json --lengths 16384 \
  --ratios 2 4 8 16 32 64 128 256 --output outputs/phonebook/MiniCPM5-2B/latent

python run.py --method latent --model openbmb/MiniCPM5-2B \
  --manifest data/phonebook/manifest.json --lengths 4096 8192 16384 32768 \
  --ratios 16 32 --output outputs/phonebook/MiniCPM5-2B/latent
```

## Method definitions

- `latent.py`: global KVZip+ preservers plus attention anchors and farthest-key
  diversifiers; two-stage fitting with the 1,000/1,000 schedule. Stage 1 trains
  soft K/V/bias; Stage 2 trains all K/V/bias with top-20-plus-tail KL and token CE.
- `kvzip.py`: globally selected reconstruction-scored original KV entries.
- `attention_matching.py`: repeat-prefill reference queries, RMS key selection,
  two NNLS iterations bounded by `exp(-3)` and `exp(3)`, and least-squares values.
- `snapkv.py`: pooled attention from the final document observation window,
  with a uniform per-layer budget and shared GQA entries.
- `pyramidkv.py`: the same observation scores with beta 20 layer allocation,
  rounded to preserve the total budget and capped by document length. Pooling
  width is five. Both observation-window methods use `min(32, budget)` positions;
  a window-only allocation retains the document tail.
- `summary.py`: same-model factual text compression, without evaluation queries.
  Generation uses the model's chat template, thinking disabled, greedy decoding,
  and `max(512, 2 * budget)` tokens. Clip to the reader-token budget and verify
  decode/re-encode length. Prefill the summary as raw text and reuse its cache.
- `full.py`: the uncompressed frozen reader.

## One document, resume, and aggregation

Any method also accepts a single split and explicit budget:

```bash
python run.py --method summary --model Qwen/Qwen3-1.7B \
  --split data/phonebook/splits/phonebook-16384-00000.json \
  --budget 1024 --seed 422 --output outputs/one-document
```

Latent runs save `cache.pt`, its training metadata, and `result.json`. Baselines
save `result.json`; summary results include both clipped and original text,
token counts, and generation settings. Settings and input checksums are recorded.
Use a local checkpoint directory in `--model` for offline runs; optionally pin
Hub revisions with `hf download --revision` for archival use.

`--shard K --num-shards N` assigns independent documents across processes or
machines. Use `--device cuda:N` to choose a GPU. Each result cell must have one
writer. Resume by repeating the same command; completed matching cells are reused.

```bash
python aggregate.py --results outputs/ruler/latent \
  --output outputs/ruler/latent/summary.json
```

The output reports document counts, scores per task and context length, and task
macro-averages at each ratio. Check the completed cohort counts before comparing
results. Keep settings, code versions, model revisions, and input manifests with
reported results.
