# Paper Reading Log (Minimal)

## Reading Level Legend
- **Insight**: Core idea and problem framing understood
- **Review**: Full paper understood (methods, equations, experiments)
- **Implemented**: Implemented in code / hardware / tools
- **Improved / Paper**: Extended, improved, and published

---

## 2025 Summer

### Systems / LLM Serving

- **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention**  
  - Level: Review  
  - Note: Proposes a paging-based KV-cache management scheme that significantly improves memory efficiency and throughput in LLM serving.

### Systems + HW (PIM)

- **Decoupled Chunk Attention with Processing-in-Memory**  
  - Level: Review  
  - Note: Reorganizes attention computation into decoupled chunks to alleviate memory bottlenecks and enable PIM-friendly execution.

---

## 2025 Fall

### Hardware / EDA

- **SeGen: Automatic Topology Generator for Sequencing Elements (B-Flex)**  
  - Level: Improved / Paper  
  - Note: Automatically explores sequencing-element topology spaces and generates novel flip-flop designs, culminating in a published paper.

### ML Methods (Seminar — Insight Only)

- **Bayesian Code Diffusion for Efficient Automatic Deep Learning Program Optimization**  
  - Level: Insight  
  - Note: Uses Bayesian diffusion processes to automate optimization of deep learning programs.

- **BF-STVSR: B-Splines and Fourier – Best Friends for High-Fidelity Spatial-Temporal Video Super-Resolution**  
  - Level: Insight  
  - Note: Combines B-spline and Fourier representations to achieve high-fidelity spatial-temporal video super-resolution.

- **BIGS: Bimanual Category-agnostic Interaction Reconstruction**  
  - Level: Insight  
  - Note: Reconstructs bimanual interactions without relying on object category assumptions.

- **HUSH: Holistic Panoramic 3D Scene Understanding using Spherical Harmonics**  
  - Level: Insight  
  - Note: Uses spherical harmonics to enable holistic understanding of panoramic 3D scenes.

- **Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning**  
  - Level: Insight  
  - Note: Shows that flatter reward landscapes in policy parameter space lead to more robust reinforcement learning.

---

## 2025 Winter

### ML Methods

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (2)

### Hardware / EDA

- **INSTA: An Ultra-Fast, Differentiable, Statistical Static Timing Analysis Engine for Industrial Physical Design Applications**  (1)

### Neuromorphic / SNN

#### Chips / Platforms

- **TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip**
  - Level: Review
  - Note: Utilizing SNN's asynchronous nature, architecture / circuit / manufacture level departure from conventional von-Neumman architecture.

- **A Million Spiking-Neuron Integrated Circuit with a Scalable Communication Network and Interface**
- **SpiNNaker: A 1-Million Core Spiking-Neuron Integrated Circuit Platform for Real-Time Brain Simulation**
- **Loihi: A Neuromorphic Manycore Processor with On-Chip Learning**
- **Loihi 2: A Neuromorphic Manycore Processor with Programmable Learning**

#### Training / Pruning

- **Workload-Balanced Pruning for Sparse Spiking Neural Networks (U-ticket)**
  - Level: Review
  - Note: Iteratively modifying neuron connection while training the model, It resolves load imbalance problem while preserving sparsity and accuracy.

- **Prosperity: Accelerating Spiking Neural Networks via Product Sparsity**
  - Level : Review
  - Note : Utilizing product sparsity(reuse of axon accumulation) of SNN, Optimized hardware with algorithm-aware method, result in significantly improved power, latency, area when inference spiking CNN / spiking transformers.

- **AQUA: Activity- and Quantization-Aware Uniform Pruning for Spiking Neural Networks**
  - Level : Review
  - Note : Considering SNN's temporal quantization-friendly nature. which is distinguished from DNN, formalizing criteria for activation-aware pruning and quantization-aware rebalancing in LTH-based training-pruning proceedure.

#### Accelerators / Dataflow

- **Sata: Sparsity-aware training accelerator for spiking neural networks** 
- **GoSPA: An Energy-efficient High-performance Globally Optimized SParse Convolutional Neural Network Accelerator∗** 
- **LoAS: Fully Temporal-Parallel Dataflow for Dual-Sparse Spiking Neural Networks**
- **SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit Differentiation** 
- **Identifying Efficient Dataflows for Spiking Neural Networks**
- **SpikeExplorer: Hardware-Oriented Design Space Exploration for Spiking Neural Networks on FPGA**
- **Design Space Exploration of Sparsity-Aware Application-Specific Spiking Neural Network Accelerators**
- **SpikeX: Exploring Accelerator Architecture and Network Co-Design for Spiking Neural Networks** 
- **Spiker+: A Framework for Generating Efficient Spiking Neural Network Accelerators on FPGA**
- **SATA: Sparsity-Aware Training Accelerator for Spiking Neural Networks**
- **Gemmini**
  - Level : Listing 
  - Note : Systolic-array based DNN accelerator generator which provides full-stack integration (HW configs -> Multi-level SW -> Linux SoC)
- **Simba**

### Accelerators / DSE

#### Frameworks / Models

- **Timeloop: A Scalable and Accurate Architecture Modeling Framework for Deep Neural Network Accelerators** 
- **Accelergy: An Architecture-Level Energy Estimation Methodology for Accelerator Designs**
- **MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of Deep Neural Network Accelerators** 
- **MAGNet: A Modular Accelerator Generator for Neural Networks**
- **SMAUG: End-to-End Full-Stack Simulation Framework for Deep Neural Network Accelerators**
- **TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators**

#### Mapping / Dataflow

- **dMazeRunner: Executing Perfectly Nested Loops on Reconfigurable Accelerator Architectures**
- **Sparseloop: An Analytical, Iterative Design Space Exploration Framework for Sparse Tensor Accelerators** 
  - Level : Review 
  - Note : Generalizing sparse-aware accelerator techniques and integrate it into simulation flow, building Flexible, Fast, Accurate simulator for Tensor accelerators.
- **Ruby: Improving Hardware Efficiency for Tensor Algebra Accelerators Through Imperfect Factorization** 
  - Level : Review 
  - Note : Extending design space from Perfect factorization => Imperfect factorization space  mapping space, find more optimal solution space.

#### DSE Methods

- **GAMMA**
- **COSA**
- **Mind Mapping**
- **ZigZag: A Memory-Centric Rapid Deep Neural Network Accelerator Design Space Exploration Framework** (1)
- **ArchitectV2**
- **DOSA**
  - Level : Review 
  - Note : Mapping-first, Analytical model based Gradient discent ANN accelerator DSE within Gemmini architecture
- **GANDSE**
- **NAAS**
- **VAESA**
- **Ruby: Improving Hardware Efficiency for Tensor Algebra Accelerators Through Imperfect Factorization** 
  - Level : Review 
  - Note : Extending design space from Perfect factorization => Imperfect factorization space  mapping space, find more optimal solution space.

### Misc

- **Optimality of Gerver’s Sofa**







