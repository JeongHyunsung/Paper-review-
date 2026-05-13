---
title: "FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models"
ref: ASPLOS 2025
date : 2026-05-05
PI : Xiaowen Chu
level: "Review"
status: "Draft"
---

# 1. Motivation
MoE efficient training 

# 2. Past Method 

## 2.1. Problem 
MoE training/inference framework scalability / efficiency.

## 2.2. Remaining Problem 
Lack of flexibility (gating/ordering) 
Motivation for 
1. optimizing network communication (pipelining oppeortunity)
2. optimizing forward and backward seperately 


# 3. Proposed Method 
## 3.1. Modularization 
Gate, Order, Dispatch, Expert, Combine, I-order modularization 

## 3.2. Scheduling optimization 
Overlapping, Gradient reduction partitioning.

Module latency Linear modeling => 4 heuristic-based scenario objective function => optimize r => use best scheduling strategies.

In runtime, CPU uses selcted scheduling strategies based on pre-execution profiling & calculation.

# 4. Implementation

# 5. Experiment
## 5.1. Linear regression fitted well.
## 5.2. Framework 
- for 1458 configured layers, faster 
- for end to end model training, faster 
- robust to Pipeline parallelism, sequence length, scale 
- flexible but efficient.

# 6. Conclusion