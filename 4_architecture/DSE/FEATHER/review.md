---
title: FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching
ref: ISCA2024 
date : 2026-06-29
PI : Tushar Krishna
level: review
status: draft
---

# TL;DR

# 1. Motivation
Changing dataflow by layer requires additional hardware overhead: 
- reconfiguring datapath in computation/distribution/reduction
- modifying data layout in on-chip buffers.
* especially, without effective data layout, the overhead outweigh the benefit from changing the dataflow layerwise.

# 2. Past Method 
- many of them considered reconfigurable datapath, but data layout is often ignored, However, it is critical.

- Insights
1. Disordance between dataflow and data layouy leads to bank conflict and results in performance degradation
2. (Layout, Dataflow) should be co-changed layerwise.
3. Also, these effect should be considered in DSE flow to reduce theory-practice gap.


# 3. Proposed Method 

## 3.1. Key idea
- First question: How to connect dataflow with appropriate, concordant layout? 
=> Dataflow-Layout interaction: if required data is packed into single cache line, it is advantageous.
- Second question: Then, different layer might require different data layout. How can we reorder data layout? 
=> Off-chip / On-chip(after reduction, **while reduction**)

## Neural engine: NEST
## Reduction-reorder network: BIRRD





# 5. Experiment

# 6. Conclusion

# 7. My perspective 
