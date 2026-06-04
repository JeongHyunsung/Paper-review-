---
title: "Centaur: A Chiplet-based, Hybrid Sparse-Dense Accelerator for Personalized Recommendations"
ref: ISCA 2020
date : 2026-06-04
Author : Ranggi Hwang
level: "Listing"
status: "Draft"
---

# Summary
Recommendation engines are typically deployed on CPU-based systems because of their large memeory requriements and latency-critical nature. This paper pinpoints that the major bottlenecks in recommendation models are sparse embedding table lookup, and dense MLP computation, which motivates acceleration on chiplet-based CPU+FPGA systems. Centaur directly accesses CPU-resident embedding tables and accelerates both sparse embeding operations and MLP computation.


# 1. What is the problem being solved?
- Personalized recommendation models are difficult to accelerate with conventional DNN accelerators because a large portion of their execution latency comes from "sparse" embedding table lookups rather than regular matrix operations.
- Because of large memory requirements, they are primarily deployed on CPU-only environments, however workload's heterogeneous nature(both sparse(memory bound) and dense(compute bound)) motivates its acceleration with FPGA.

# 2. What is unique about the suggested solution?
- Centaur uses a package-integrated CPU+FPGA substrate where the FPGA can directly access embedding tables stored in CPU memory, avoiding explicit movement of large embedding data between seperate CPU and accelerator memories.
- Matched with the workload's nature, Centaur is composed of EB-Streamer(handling embedding table lookup) and GeMM acceleration unit(handling feature interaction and GeMM computation).
- Sparse accelerator performs embedding gather and reduction through hardware-level address generation and streaming reduction, while the dense accelerator handles MLP computation through multicast-based PE array 

# 3. How is the idea evaluated?
- The authors implement Centaur on Intel HARPv2, a package-integrated CPU+FPGA platform with an Intel Xeon CPU and an Altera Arria 10 FPGA, using SystemVerilog RTL and FPGA synthesis.
- The evaluation uses DLRM-based recommendation model configurations that vary the number of embedding tables, gathers per table, embedding table size, and MLP size, allowing the study to cover both sparse-heavy and dense-heavy cases.
- Centaur is compared against CPU-only and CPU-GPU baselines using effective embedding memory throughput, energy efficiency, and end-to-end inference latency, achieving 1.7-17.2x speedup and 1.7-19.5x energy efficiency improvement over CPU-only execution.
