B : batch
L : sequence length 
M : embedding size 
k : top-k gating expert
f : load imbalance extra slot
H : heads 
h : hidden(ffn)

# Model flow 

1. Attention 
Input : B,L,M (attention input)
B,L,M=>B,L,3M=>3,B,L,M
Q,K,V : B,L,M (Query, Key, Values)
B,L,M=>B,L,H,m=>B,H,L,m
head-Q,K,V : B,H,L,m
Q*K^T/sqrt(m) : B,H,L,L
(if decoder => causal masking -inf)
softmax()*V : B,H,L,m
B,H,L,m=>B,L,M
concated head : B,L,M
(head*output)Output : B,L,M (attention output)

2. Gating
- GShard - topk,sigmoid expert activation
- BASE/StableMoE - Sigmoid expert activation
- XMoE - low dim space projection and cosine sim -> topk, sigmoide expert activation

Input : B,L,M (attention output)
Output : B,L,k,E (token-expert mapping)

3. Ordering 
Input : B,L,M + B,L,k,E
Output : E,T,M

T ~= B*L*k*f/E

- GShard : dense tensor operation
- TuTel : SimT-efficient ordering (CUDA optimization)

4. Expert FFN computation 
Input : E,T,M
Middle : E,T,h
Output : E,T,M

- weight split into shard, and distributed into multiple GPUs, and concatenated.

X. Reverse ordering 
Input : E,T,M
Output : B,L,M

- Contains reduction if k != 1.

# Parallelism 

1. Data parallelism 
spatial for b in inter-node GPUs
    forward 
    backward 
allreduce (Gradient reduction + update), typically Internode 

2. Model parallelism 
Tensor parallelism in MLA, (divide in H)
spatial for H in intra-node GPUs
    foward attention
    allreduce (Output reduction)
    foward expert 

    backward

3. Expert parallelism / Expert shard parallelism 


N_DP : Data parallel, N_MP : Model parallel 
N_EP : Expert parallel, N_ECP : Expert shard parallel

INPUT : B,L,M

spatial for dp in N_DP:
    - B/N_DP, L, M

    spatial for mp in N_MP:
        attention_foward_MP_slice() # B/N_DP, L, M => B/N_DP, L, M
        # even if it is sliced with H dimension, input and output dim is # same. Wq Wk Wv :M * M/N_MP, softmax(qkT)*V : B/N_DP,L,M/N_MP
        # Wo : M/N_MP * M

    MP_reduce()
    scatter()
    gate()
AlltoAll Dispatch() 

spatial for ep in N_EP:
    gather() 
    - E/N_EP,T,M
    spatial for esp in N_ESP:
        - E/N_EP,T,M/N_ESP
        
        expert_foward_ESP_slice() # E/N_EP,T,M => E/N_EP,T,M
        # same with MP case 
    ESP_reduce()
    scatter()
AlltoAll Gather()
    













