---
title: "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration"
ref: DAC 2021
date : 2026-02-09
authors : Hasan Genc et al
level: "Listing"
status: "Draft"
---

# TL;DR
## Systolic-array based DNN accelerator generator which provides full-stack integration (HW configs -> Multi-level SW -> Linux SoC)

---

## 1. Problem Context

### 1.1 Main Problem
DNN accelerator generation productivity matters.

### 1.2 Previous Approaches and Their Limitations
Existing DNN accelerator consider limited fraction of programming/computing stack, which results in discrepency between modeling vs realization

## 2. Methodology
Implemetation is major factor, methodology is not important.
### 2.1 Core Idea
- Accelerator templates (systolic-array based architecture) 

**systolic array throughput**
mk * kn => mn matrix mul : m-1 + n-1 + k cycles when requires 

- SoC integration(CPU + Accelerator + SW stack + perhepiary) Systemic architecture evaluation methodology 

### 2.2 How and Why It Works

### 2.3 Implementation Details (When Necessary)

## 3. Results and Discussion

### 3.1 Experimental Setup

### 3.2 Key Results

### 3.3 Conclusions and Implications

## 4. My Perspective (Optional)
- Rather than architectural generality, paper seems to keep "Implementation consistency" to systemically evaluate overall performance.
