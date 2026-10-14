# Third-party notices

The project-owned release code uses the MIT license. Bundled upstream components
retain their own copyright notices and licenses; public author names in those
notices are required attribution, not private identity records.

- LongBench official metrics, metric registry and two task-template configuration
  files: `third_party/longbench/`, from THUDM/LongBench, revision
  `2e00731f8d0bff23dc4325161044d0ed8af94c1e`. MIT license included.
  https://github.com/THUDM/LongBench
- Attention Matching base and highest-attention-key algorithm:
  `third_party/attention_matching/`. MIT license included. Source revision is
  recorded in `docs/upstream-revisions.json`.
  https://github.com/adamzweiger/compaction
- SnapKV/PyramidKV adaptations in `latent_compaction/eviction.py` derive from their public
  algorithms. Original MIT notices are in `third_party/licenses/`.
  https://github.com/FasterDecoding/SnapKV and https://github.com/Zefan-Cai/PyramidKV
- KVPress 0.5.4 is installed as a dependency, not copied into this tree. The setup
  script changes its Transformers version requirement as explained in
  `docs/environments.md`; its Python implementation is unmodified.
  https://github.com/NVIDIA/kvpress
- RULER generators and evaluation datasets are not bundled. Download through their
  official sources and follow their licenses and access requirements.
  https://github.com/NVIDIA/RULER

`docs/source-manifest.json` records hashes of local scientific source files used
for extraction. Release modifications include portable imports, removal of forced
mirrors/host identity, CLI/data preparation, and alignment of the primary cache
implementation with the current paper. The original research repository history
is not part of this release.
