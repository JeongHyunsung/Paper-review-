---
title: "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices"
ref: JETCAS 2019
date : 2026-06-03
PI : Joel Emer, Vivienne Sze
level: "Review"
status: "Draft"
---

# TL;DR

# 1. Motivation
Efficient and flexible accelerator for sparse / compact DNNs.

# 2. Past Method 

## 2.1. Problem 
- Array utilization suffers from a. shape of input workload
- PE utilization suffers from a. bandwidth of broadcast NoC, b. hardware should adapt to workloads
- sparse workloads additionally suffers from a. workload imbalance, pe utilziation, b. Irregular access patterns.


# 3. Proposed Method 
## 3.1. Hierarchical mesh network (HM-NoC)
- can be flexible (high-bandwidth / high-reuse)
- reduce a implementation cost through limiting the a2a communication cost, circuit-switched routing 

### Motivation
- To adapt to divserse network, on-chip interconnect should be able to flexiblly adapt to dataflow pattern.
- To achieve this with limited hardware resources, implement diverse types of interconnect(unicast/boradcast/multicast/reduction) by composing inter-cluster mesh network and intra-cluster all2all network.

### Memory hierarchy 
GLB buffer => Router cluster => PE cluster
- Router cluster is connected with mesh network each other 
- Router cluster and PE cluster is connected with all2all network.


## 3.2. Sparse Network support

# 4. Implementation


# 5. Experiment

# 6. Conclusion
By proposing and implementing HM-NoC, multiple dataflow pattern can be effectively implemented through combination of 2-level NoC. 
As a result, eyeriss2 can adapt to the various workloads that contains dconv, pconv, sparse gemm, etc..

# 7. My perspective 
- The main point this paper wants to solve is **Adaptability and flexibility** of hardware.
- Eyeriss v2 achieves flexibility by using 2-level NoC and composing them into uni/multi/broadcast.
- However, if we have an information of workload, and if the interconnect can be fixed to single structure, 
- then we can consider fixing the hardware to support single dataflow.
- Even though the same hardware may not concurrently support both unicast and broadcast efficiently,
- we can modify the mapped dimensions and optimize the mapping so that it satisfies the specific dataflow.
