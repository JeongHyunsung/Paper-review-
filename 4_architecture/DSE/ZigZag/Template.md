---
title: "ZigZag: A Memory-Centric Rapid DNN Accelerator Design Space Exploration Framework"
ref: MICRO 2022
date : 2026-03-02
authors : Linyan Mei et al
level: "Listing"
status: "Draft"
---

# TL;DR

---

## 1. Problem Context

### 1.1 Main Problem
Architecture, Mapping design space exploration for DNN accelerators.

### 1.2 Previous Approaches and Their Limitations
Limited scope (Fixed dataflow / Fixed architecture)

## 2. Methodology

### 2.1 Core Idea
Hardware cost estimator + Temporal mapping generator + Arch generator 

### 2.2 How and Why It Works
### 2.3 Implementation Details (When Necessary)


## 3. Results and Discussion

### 3.1 Experimental Setup

### 3.2 Key Results

### 3.3 Conclusions and Implications

## 4. My Perspective (Optional)
1. Their statement is "They expanded the design space (architecture + mapping). 

- The architecture space is restricted to a predefined memory topology pool under area constraints. → Therefore, the explored space is a subset of memory-topology configurations, not the full architecture space.
- Arch-first exploration is unnatural, because microarch config can be derived from mapping. Arch space specification itself restrict mapping space.

=> By fixing architecture candidates early, the global design space is pruned before mapping exploration begins. Hence, the claimed “design space expansion” is effectively: An expansion of the mapping search space within a restricted memory-topology subspace.

2. Lack of formalization & reasoning (reuse factor <-> heuristic method)


