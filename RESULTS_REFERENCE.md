# Results: 226 JAX XLA Transformer Forward Pass

**Main goal:** Measure JAX/XLA transformer forward-pass performance  
**Tools:** Run gpu-bench-jax-xla-forwardpass.py via collect_jax_xla.py

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_138 · recorded 2026-11-12). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 NVL (95830 MiB) · AMD EPYC 9374F 32-Core Processor · Ubuntu 24.04.5 LTS · kernel 6.8.0-138-generic · driver 595.91.07
- **Run:** 2026-11-12T23:26:27Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:19 · benchmark 00:13 · whole run 01:42
- **Source:** run 226_20261112_232627
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype FP16 --sequence-len 16 --hidden-size 64 --num-layers 1 --num-heads 4 --vocab-size 32000 --batch-size 1 --warmup-iters 1 --num-iterations 3 --seed 42`

| # | Metric | Result |
|---|---|---|
| 1 | Forward-pass latency, ms (forward_latency_msec) | 0.08529433333333333 |
| 2 | Throughput, tokens/s (tokens_per_sec) | 195241.030763 |
| 3 | Modeled TFLOPS, full forward pass (tflops) | 0.813302 |
| 4 | XLA compilation time, ms (jit_compile_msec) | 1071.9575353333332 |
| 5 | Peak HBM, GB (peak_hbm_gb) | 0.14254 |
<!-- /profile:smoke -->
