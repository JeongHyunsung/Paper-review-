# AI driven design automation for multi chip integration in chips
## Motivation 
- AI application is based on three factors : Data / Algorithm / HW 
- HW is bottleneck, multi-chip packaging is promising 
- 1) Efficiency from Domain specific application and 
- 2) Lack for supply of HW 
1), 2) makes every tech make their own chips..



## Semiconductor chip 
### 3 / 2.5 / 5.5 packaging 
3D packaging : stacks 
2.5D packaging : locate in 2D 
5.5D packaging : combination 3D/2.5D 

### GPU design / manufacture 
GPU(Nvidia/TSMC) / HBM(SM/SK) / Interposer(TSMC/TSMC)
- (Physical level) Design process 
GPU : Automated algorithms
HBM, Interposer : Heuristic

#### Physical design
Design specification + Architecture(RTL) => 3D / 2.5D layout AND Analyze 

### Research 
#### Replacing silicon to glass (is this work..?)
- Making sample or design costs to much 
- To Predict it with small cost, we need AI application
- design & simulation result data is expensive .

==> From design, EDA flow mapping will generate EDA <=> Performance/Realiability/Area dataset (2 month)

AI model is composed of 2 parts. (Performance predictor / Optimization actor)
1. Performcance predictor trains first (days) => supervised learning 
2. Optimization actor <=> Performance trains second (hours) => reinforcement learning 
3. Actor returns to original database 

#### 3D packaging Optimization (Placement, before routing) problem
- Design process of 3D packaging is Exponentially increased design space.

1. From the baseline placement, basically calculate few meaningful metrics from current position,
2. Refine overall placement using gradent, iterate.

