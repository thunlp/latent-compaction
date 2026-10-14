# Environments

## Dependency contract

The tested model adapter uses Transformers 5.6.0. KVPress 0.5.4 declares
Transformers `<5.3`, so an unmodified `pip install` of both packages cannot resolve.
`tools/setup.py` downloads the exact KVPress wheel and adjusts its metadata to
use `transformers==5.6.0` and `fire>=0.7.1,<0.8`. The latter supports Python 3.13,
which removed the `pipes` module used by older Fire versions. The script
regenerates wheel RECORD hashes and installs the wheel plus `requirements.txt`
into the active virtual environment. It prints the override and then checks
dependencies and tests. The project code is imported directly from this checkout;
setup does not install the project, register a `latent-compaction` executable,
edit shell profiles, or change the system Python environment.
This is a deliberate compatibility patch, not a claim of upstream support.
LongBench
scoring uses Jieba's resource loader, so `requirements.txt` also keeps Setuptools
below 81 for that API.
Original dependency metadata: https://github.com/NVIDIA/kvpress/blob/v0.5.4/pyproject.toml.

PyTorch is hardware dependent and is not pinned to a single CUDA build. Use a
matching driver and wheel. The release validation report records the tested stack.
For an archival run, save `python -m pip freeze`, model and tokenizer revisions,
input manifest, code commit, and all cache metadata with the outputs.

## Hardware

- NVIDIA CUDA: the intended full-experiment path; frozen model computation uses
  BF16 and optimized cache parameters use FP32. The GPU must support BF16.
- CPU: FP32 functional validation and small checkpoints. Full 480-prompt training
  will be slow and may exceed available RAM.
- Apple MPS: an explicit `--device mps` is accepted; generator sampling uses CPU
  indices transferred to the device. For a reproducible Mac smoke check, use CPU.
- Other accelerators: the cache mathematics uses PyTorch operations, but adapter
  and KVPress support must be validated on the target backend. Do not assume CUDA
  performance or numerical equivalence.

Memory grows with model size, document length, and captured teacher queries.
No universal VRAM requirement is asserted. Start with the smoke config; measure
peak memory before launching a full sweep. Reducing `stage1_queries`, prompt count,
batch size, precision, or context length changes the experimental protocol.
FlashAttention is not required. The compacted path explicitly computes FP32
attention scores and log normalizers; a custom kernel must expose both quantities.

## Offline and restricted networks

Download dependencies and checkpoints on an internet-connected machine, then copy
them to the experiment machine. Use a local checkpoint path with `--model` and
set `HF_HUB_OFFLINE=1`. Prepare benchmark splits before going offline. No network proxy, mirror, VPN, or scheduler is set
by this repository. Hugging Face access tokens are supplied through standard local
authentication or environment settings, never committed files.

Use an isolated environment. Do not modify an unrelated research environment to
install the release. The main CLI and data workflows use portable Python and paths;
use distinct output directories for different model revisions and
protocols. Multiple workers must not write the same cell concurrently.
