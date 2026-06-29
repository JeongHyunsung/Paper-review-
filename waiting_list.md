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

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**

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

## 2026 SPRING 

### Berkley-sophia shao 
- **Stellar: An Automated Design Framework for Dense and Sparse Spatial Accelerators**

### Cornell-Jishen Zhao
- **SmoothE: Differentiable E-Graph Extraction**
  - Level : Review
  - Note : Relaxing the binary E-graph extraction problem into a differentiable continuous optimization framework, while explicitly modeling the coupling between local selection probabilities and global structural consistency, and enforcing acyclicity through differentiable constraints.

### UNIST-Ranggi Hwang
- **Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference**
  - Level : Review 
  - Note : Resolving expert selection-execution data dependency which causes CPU-GPU data transfer overhead, by pre-gating experts 1 layer before, improving small scale MoE deployment scalability & performance & efficiency.

### MoE expert parallelism 
- **MoEntwine** HPCA 2025
- **LAER-MoE** ASPLOS 2026
- **FSMoE** ASPLOS 2025
  - Level : Review 
  - Note : End-to-end MoE inference/training system tackling algorithmic flexibility and exploiting dynamic scheduling pipelining opportunity

### GTech-Callie hao
- **Faster-MoA** DAC2026
  - Level : Review
  - Note : Tackling redundancy & inefficiency in MoA dense inference, exploit A2A sparsity via tree-structured topology
- **Omnisim** 
  - Level : Review 
  - Note : Bridging the gap between HLS(c-level abstraction) to RTL level simulation, improved capability and scalability of performance/functional simulation by flexibly correlates them.
- **LaZagna** ICCAD2025
  - Level : Review
  - Note : 

### MIT-Joel Emer
- **Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices**
  - Level: Review
  - Note: By proposing and implementing 2-level hierarchical NoC, eyetiss V2 could adapt to various range(compact, sparse NN) of workloads while remain efficient. Diverse dataflow type is realized by composing inter-cluster mesh network and intra-cluster all2all netowrk.

### Course-driven (ACA)
- In-Datacenter Performance Analysis of a Tensor Processing Unit (Analysis of TPU performance and energy efficiecny based on real world workload)
- Centaur: A Chiplet-based, Hybrid Sparse-Dense Accelerator for Personalized Recommendations (Accelerating heterogeneous worklaod via package-integrated CPU+FPGA system and custom heterogeneous microarchitecture)

## Stanford-Priyanka Reina
- **Voyager: An End-to-End Framework for Design-Space Exploration and Generation of DNN Accelerators**
  - Level: Review
  - Note: (Broadly) Parametrized end-to-end DNN accelerator generator

## GTech-Callie Hao
- **LaZagna: An Open-Source Framework for Flexible 3D FPGA Architectural Exploration**
  - Level: Listing 
  - Note: 3D FPGA fabric & Benchmark P&R flow on top of open-sourced project
- **Escaping Flatland** 
  - Level: Listing 
  - Note: 3D-aware FPGA placement problem (algorithmic improvement)

## GTech-Tushar Krishna 
- **FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching**
  - Level: Review 
  - Note: (method 부터 다시 보기)



GROW: A Row-Stationary Sparse-Dense GEMM Accelerator for Memory-Efficient Graph Convolutional Neural Networks, Hwang et al., HPCA-2023

SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks, Parashar et al., ISCA-2017




