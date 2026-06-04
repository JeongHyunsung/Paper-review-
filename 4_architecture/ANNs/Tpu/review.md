---
title: "In-Datacenter Performance Analysis of a Tensor Processing Unit"
ref: ISCA 2017
date : 2026-06-04
PI : Google
level: "Listing"
status: "Draft"
---

# Summary
This paper analyzes the effectiveness of Google’s first-generation TPU in real datacenter inference workloads. Motivated by the difficulty of CPUs and GPUs in satisfying strict 99th-percentile latency constraints, TPU adopts a domain-specific and deterministic architecture based on a large systolic array. The paper shows that this design achieves lower latency, higher throughput, and better energy efficiency than contemporary CPU/GPU baselines across production neural network workloads.

# 1. What is the problem being solved?
- This paper addresses the challenge of running neural network(MLP, CNN, LSTM) inference efficiently in large-scale datacenter environments.
- More specifically, in production services, performance is constrained not only by throughput, but also by strict 99th-percentile latency requirements as it directly affects user experience.
- However, CPU and GPUs can provide high peak performnace, but their execution machanicsms and memory systems make it difficult to sustain predictable/deterministic latency inference.
- The main problem is, how to achieve high throughput and energy efficiency while satisfying real-world service latency constraints.

# 2. What is unique about the suggested solution?
- The paper evaluates Google's first generation TPU, a custom ASIC actually deployed in datacenters for network inference.
- TPU uses 256x256 systolic array with 65,536 8-bit MAC units, enabling high-throughput matrix multiplication. Systolic array is good for increasing the archimetic intensity and 8-bit design choice remains hardware power and area in reasonable range.
- Unlike CPUs and GPUs, TPU adopts a simpler and more deterministic execution model, using software managed memory instead of relying heavily on dynamic hardware mechanism, this sacrifies some general-purpose flexibility but improves predictability, efficiency, and performance per watt for inference workloads.

# 3. How is the idea evaluated?
- TPU is evaluated using real Google production inference workloads written in Tensorflow. 
- The benchmark includes MLPs, CNNs, LSTMs which represents most of Google's datacenter demand at that time(2015)
- The paper compares TPU against contemporary datacenter CPU and GPU baselines, including Intel Haswell CPU, and NVIDIA K80 GPU, this hardware seems outdated, but it is fair as practical options in that time.
- The evaluation focuses on latency, throughput, utilization, and energy efficiency. Compared with an Intel Haswell CPU and an NVIDIA K80 GPU, TPU achieves about 15×–30× higher performance and 30×–80× higher TOPS/Watt across production workloads.

