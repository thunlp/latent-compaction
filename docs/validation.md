# Verification record

Verified on 2026-10-14 with Python 3.13, PyTorch 2.12.0, Transformers 5.6.0
and KVPress 0.5.4 using the setup script's explicit dependency metadata overrides.
Execution used CPU.

- Seventeen tests pass, covering: two-stage fitting and generation for tiny Qwen3 and Llama;
  native full-cache parity; artifact round-trip and model-ID rejection; global
  budgets; anchor/diversifier initialization; bias freeze schedules; masked
  attention outputs and log normalizers; metric semantics; deterministic
  phonebooks; real KVPress score hooks; portable baseline construction, including
  small and capped observation-window budgets; summary
  generation settings, decode/re-encode clipping, and reuse across held-out questions.
- `pip check` passes after dependency-only setup.
- Tests, benchmark runs, and saved-cache generation succeed with no installed
  project distribution or `latent-compaction` executable.
- A local tiny Qwen3 checkpoint completed CLI fitting with real KVPress scoring
  and all 480 generic prompts, serialization, reloading, and held-out evaluation
  against its native full-cache reference.
- All seven methods complete an end-to-end tiny-checkpoint run and completed-cell
  resume through the shared `run.py` entry point.
- All 480 generic prompt strings and their order match the research source.
- The 320 selected LongBench row indices match the recorded source cohort.
- Unified runner, data preparation, and aggregation
  help commands succeed.
- Python source compilation succeeds.
- `tools/audit_release.py` reports no matches for its private-path, private-IP,
  private-key, credential and generated-artifact checks.

Run `python tools/setup.py --check-only` to repeat package checks and tests.
Run `python tools/audit_release.py` to repeat the source content scan.
