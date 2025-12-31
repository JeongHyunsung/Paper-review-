# Level : listing

# 1. Main Problem
- SW side (model architecture / compiler ) level Optimization of AI inference
- Find the code optimized parameter (in code search space)

# 2. Past Method 

# 3. Proposed Method 
## 1. Use Diffusion idea in exploration of code parameters.
- In multiple layer, code parameter theta has approximately similar value. (HARD assumption)
- To minimize search time, One code parameter from another layer is reused to another layer.

- Experimentally proved, assumption was true

- Insight from image diffusion moddel, noising step (foward step) in diffusion is implemented in searching process of code variables.


# 5. Implementation

# 6. Experiment


# 7. Conclusion
- Compiling time, Running time both improved.
