---
title: LaZagna: An Open-Source Framework for Flexible 3D FPGA Architectural Exploration
ref: ICCAD2025 
date : 2026-06-18
PI : Callie Hao
level: review
status: listing
---

# TL;DR

# 1. Motivation
3D FPGA architecture exploration & bitstream generation

# 2. Past Method 
## 2.1. 3D FPGA architecture studies 
- homogeneous(all layer share an identical layout, logic blocks, routing components)
- Non-logic heterogeneous(logic at 1 layer)
- Logic Heterogeneous 

## FPGA tools 
- VTR(verilog-to-routing): 2D FPGA, 2-layer homomgeneous 
- OpenFPGA: support bitstream generation for user-defined architecture, lacks 3D

=> Motivates architecture-configurable 3D fpga fabric/benchmark PnR flow 

# 3. Proposed Method 
## 3.1. Archietcture specification, Fabric RTL generation
- Layer number, vertical connection type, switch block patterns, etc...
- Based on specification, generate 3D-Routing resource graph.
- Generate RTL using OpenFPGA 

## 3.2. Benchmark bitstream generation
- Yosys->VTR flow.


# 5. Experiment
- RTL/bitstream functional validation: RTL simulation
- physical design feasibility: Cadence Genus 
- architecture exploration: 

# 6. Conclusion

# 7. My perspective 
