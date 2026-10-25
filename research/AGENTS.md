# Research Execution Guidelines

## CPU Verification and User GPU Runs

User instruction recorded on 2026-10-25; applies to all research implementations.

- The agent has no GPU access. Implement every pipeline stage so it can run on CPU;
  the user runs full experiments on their GPU from their own terminal.
- Provide an explicit device option, such as `--device cpu` and `--device cuda`.
  Keep model, data handling, objectives, decoding and output formats shared between
  devices. Avoid unconditional CUDA calls or mandatory GPU-only kernels.
- Provide a small CPU smoke configuration or `--smoke` mode. It must exercise
  preparation, training for a few steps, checkpoint saving/loading, generation
  and evaluation, using tiny fixtures or a small user sample with full histories.
- Run meaningful CPU mini-tests and the smoke pipeline before handing over a GPU
  command. Do not run full CPU training as a substitute for the user's GPU run.
- Smoke runs must work without downloading large pretrained weights: use a small
  local fixture or tiny randomly initialized model where necessary. Clearly report
  which pretrained-model integration has and has not been verified.
- Make GPU-specific optimizations optional. Start CPU checks in FP32; enable mixed
  precision only where the selected device supports it. An explicitly requested
  GPU must fail clearly if unavailable, without silently falling back to CPU.
- Save persistent logs, resolved configuration, seeds, source/data fingerprints,
  stage timings, checkpoints and a completion/failure status in each run directory.
  Save tracebacks on failure. The agent must be able to inspect these files after
  the user finishes the GPU run.
- Document exact CPU smoke and GPU launch commands, output locations and how to
  follow logs. Distinguish successful CPU verification from unverified GPU behavior;
  smoke metrics are not evidence of recommendation quality.

## Environment and Data

Use only the repository's current local Python 3.12 environment
(`baseline/.venv/bin/python`), unless the user explicitly changes this instruction.
Do not use or modify unrelated environments. Keep generated artifacts separate
from source and excluded from Git. Preserve temporal cutoffs and prevent future
labels from entering tokenizers, inputs or caches.
