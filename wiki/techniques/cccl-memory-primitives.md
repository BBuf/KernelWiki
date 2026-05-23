---
id: technique-cccl-memory-primitives
title: CCCL CUB Memory Primitives For Selection And Scan
type: technique
architectures:
- sm100
- sm90
tags:
- cuda-cpp
- top-k-selection
- parallel-scan
- vectorized-loads
- shared-memory-optimization
- tile-scheduling
confidence: source-reported
reproducibility: snippet
prerequisites:
- technique-vectorized-loads
- technique-swizzling
related:
- pattern-memory-bound
- technique-tile-scheduling
sources:
- pr-cccl-3559
- pr-cccl-6152
blackwell_relevance: CUB scan/top-k tuning PRs expose reusable policy and dispatch choices for B200 selection, fill, and prefix-style helper kernels.
---

## Use

Use CCCL/CUB PRs when the bottleneck is not tensor math but a memory primitive:
scan, top-k selection, fill, histogram, reduce, or block load/store policy. The
goal is usually a policy or dispatch idea rather than copying a full CUB
primitive into an application kernel.

```cuda
// Minimal policy probe shape for an application-specific top-k or scan helper.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
struct PrimitivePolicy {
  static constexpr int block_threads = BLOCK_THREADS;
  static constexpr int items_per_thread = ITEMS_PER_THREAD;
  static constexpr bool vectorized = ITEMS_PER_THREAD >= 4;
};
```

## Transfer Notes

- Check whether the PR tuned SM100 separately from SM90.
- Validate determinism and tie-breaking before using atomic or relaxed variants.
- For DSA TopK, keep radix/select cost separate from score-computation cost.
