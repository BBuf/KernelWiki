---
id: kernel-tensorrt-llm-blackwell-indexer
title: TensorRT-LLM Blackwell FP4 DSA Indexer
type: kernel
architectures:
- sm100
tags:
- attention
- gemm
- fp4
- kernel-fusion
- top-k-selection
- vectorized-loads
confidence: source-reported
reproducibility: snippet
kernel_types:
- attention
- gemm
- topk
languages:
- cuda-cpp
- python
related:
- kernel-fused-moe
- kernel-fp8-block-scale-gemm
- technique-vectorized-loads
- technique-fine-grained-quantization
sources:
- pr-TensorRT-LLM-13340
performance_claims: []
blackwell_relevance: TensorRT-LLM's Blackwell DSA indexer PR is a current upstream implementation reference for FP4/FP8 cache indexer paths and fused quantized gather/scatter kernels.
---

## Shape

TensorRT-LLM PR 13340 integrates an FP4 indexer path for DSA on Blackwell and
lands CUDA kernels for K-cache gather/scatter and fused FP4 concatenation. Use it
as implementation evidence for sparse indexer memory movement and quantized
cache layout, not as a drop-in answer for FlashInfer-Bench.

```cuda
// Evidence checklist before adapting the idea:
// 1. Inspect indexerKCacheGather.cu and indexerKCacheScatter.cu.
// 2. Check the scale and FP4 packing layout.
// 3. Benchmark gather/scatter separately from top-k selection.
__global__ void candidate_indexer_probe(const uint8_t* cache, int* indices) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  indices[tid] = static_cast<int>(cache[tid]);
}
```

## Transfer Notes

- Keep FP4 packing, scale placement, and invalid-token handling explicit in the
  correctness reference.
- Treat gather/scatter traffic as a separate NCU profile target.
- Avoid merging this with top-k selection until the memory path is understood.
