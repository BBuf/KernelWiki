---
id: kernel-flash-attention-sm100-mla-topk
title: FlashAttention SM100 MLA TopK Sparse Forward
type: kernel
architectures:
- sm100
tags:
- attention
- flash-attention
- mla
- sparse-attention
- tma
- tile-scheduling
- top-k-selection
confidence: source-reported
reproducibility: snippet
kernel_types:
- attention
- flash-attention
- mla
- sparse-attention
- topk
languages:
- cute-dsl
- python
related:
- kernel-flash-attention-4
- kernel-sparse-mla
- technique-tile-scheduling
- technique-external-source-map-research
sources:
- pr-flash-attention-2441
- pr-flash-attention-1236
performance_claims:
- gpu: B200
  dtype: bf16
  shape: batch=512, seqlen_q=1, seqlen_k=16384, nheads=128, topk=2048
  metric: latency_ms
  value: 0.3
  source_id: pr-flash-attention-2441
blackwell_relevance: PR-grade CuTe DSL SM100 MLA code is directly relevant to DSA sparse attention and top-k KV-gather routing on B200.
---

## Shape

FlashAttention PR 2441 adds an SM100 CuTe DSL forward path for MLA shapes with
top-k sparsity. It is useful when an attention candidate has to combine page/KV
layout handling, sparse top-k selection, and tiled forward scheduling.

```python
# Query pattern before borrowing implementation details:
# open PR page, then inspect the source snapshot or upstream files listed there.
from pathlib import Path

pr_page = Path("sources/prs/flash-attention/PR-2441.md")
text = pr_page.read_text()
assert "flash_fwd_mla_sm100.py" in text
assert "topk_gather_kv.py" in text
```

## Transfer Notes

- Treat top-k gather and tiled attention scheduling as separate evidence paths.
- Profile memory traffic separately from tensor-pipe utilization; sparse top-k
  routing can improve arithmetic work while worsening gather locality.
- Keep full-workload validation because the useful path is shape-specific.
