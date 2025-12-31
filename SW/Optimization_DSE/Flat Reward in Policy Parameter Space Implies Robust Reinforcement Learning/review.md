# Level : listing

# 1. Main Problem
Noise factor (action/reward perturbation) in Reinforcement learning 

# 2. Past Method 

# 3. Proposed Method 
## 1. Find FLAT maxima in reward!!!!!! 
Insight from 3d visualization of Reward: how to improve Robustness of reinforce learning model? 
- In loss function case, flat minima is more robust than sharp minima : it is good at small perturbation.
- then, How to use this insight from loss function in typical machine learning?

- there is two method to improve robustness of Reinforce learning 
- - 1. Add a perbutation in action, find worst case reward/loss, but it result in high complexity 
- - 2. (proposed method) Use the gradient vector in reward vector, 
**E-flat reward maxima WILL RESULT IN delta-action robustness** 

# 5. Implementation

# 6. Experiment 
- Improved robustness.
==> Hardware Implementation(expansion)

# 7. Conclusion