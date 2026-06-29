---
title: "TensorDIMM: A Practical Near-Memory Processing Architecture for Embeddings and Tensor Operations in Deep Learning"
ref: MICRO 2019
date : 2026-06-11
PI : Minsoo Rhu
level: "Listing"
status: "Draft"
---------------

# Summary

This paper proposes TensorDIMM, a practical near-memory processing architecture for embedding layers and tensor operations in deep learning recommender systems. Modern recommendation models are often limited by memory capacity and bandwidth because embedding tables can reach hundreds of GBs, exceeding GPU memory capacity. Existing CPU-only or hybrid CPU-GPU approaches suffers from slow CPU-memory embedding gathers, CPU computation bottlenecks, or expensive PCIe data transfers. TensorDIMM addresses this by adding lightweight near-memory processing cores inside buffered DIMMs and organizing them as a scalable remote memory pool called TensorNode.

# 1. What is the problem being solved?

- This paper identifies and addresses the memory wall problem in embedding-heavy recommender systems.
- Embedding tables can reach hundreds of GBs, which often exceeds GPU memory capacity.
- However, current system focuses accelerating compute-heavy kernels, existing CPU-only and hybrid CPU-GPU approaches suffer from low CPU-memory bandwidth, slow CPU computation, or PCIe communication overhead.
- The main problem is how to provide scalable memory capacity and bandwidth for large embedding tables while reducing unnecessary data movement.

# 2. What is unique about the suggested solution?

- TensorDIMM adds lightweight near-memory processing cores inside the buffer device of commodity DIMMs, while keeping DRAM chips unchanged.
- It accelerates key embedding-layer operations such as GATHER, REDUCE, and AVERAGE near the memory
- Instead of transferring many embeddings to the GPU, TensorDIMM performs gather and reduction near memory and sends only the reduced result.
- TensorNode connects multiple TensorDIMMs as a GPU-side remote memory pool, enabling both memory capacity and bandwidth to scale with the number of DIMMs.

# 3. How is the idea evaluated?

- The paper evaluates TensorDIMM on recommender system workloads including NCF, YouTube, Fox, and Facebook-style models.
- It compares TensorDIMM against CPU-only, CPU-GPU, and an ideal GPU-only oracle with unlimited GPU memory.
- The evaluation uses a hybrid methodology: Ramulator-based DRAM simulation, real GPU system emulation, and hardware overhead estimation.
- TensorDIMM achieves 6.2×–15.0× speedup over CPU-only and 8.9×–17.6× speedup over hybrid CPU-GPU, reaching about 84% of the ideal GPU-only performance.
