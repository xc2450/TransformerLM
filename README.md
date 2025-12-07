# TransformerLM

This project includes
* Implementation of a micro causal language model with modern architectural optimization, including RoPE with YaRN scaling, GQA, RMSNorm, KV Cache, and pre-norm attention block.
* Pretraining ablation studies on hyperparameters and architecture.
  
# Architecture 

<img width="1799" height="1043" alt="523407115-8232054f-5a2f-4b94-9ec4-187d3151eb1f" src="https://github.com/user-attachments/assets/7bda3b42-9dcd-4b29-9b73-69da6e195508" />

## Config
| params | len_vocab | rope_theta | n_layers | d_model | kv_heads | q_heads |
|--------|-----------|------------|----------|---------|----------|---------|
| 26M    | 6400      | 1e6        | 8        | 512     | 2        | 8       |

* Pretraining takes 20 minutes on a single H200, which costs approximately 1 dollar using RunPod.

# Experiments

## Pretraining 
### learning rate  
<img width="1999" height="1051" alt="image3" src="https://github.com/user-attachments/assets/8cafb201-ce39-4323-873b-121b286c0179" />

## batch size 
<img width="1999" height="1051" alt="image2" src="https://github.com/user-attachments/assets/943c7d02-e670-4042-ae40-1da20081747e" />

## RMSNorm
<img width="1999" height="1051" alt="image7" src="https://github.com/user-attachments/assets/0c713971-404d-437b-aecb-243f76cb99fe" />

* RMSNorm stabilizes the training process, and the model converges to a lower loss.

## Pre-Norm vs. Post-Norm
<img width="1999" height="1051" alt="image4" src="https://github.com/user-attachments/assets/a85c8c24-5969-4b3c-92b0-3133a18029d8" />

* For Post-Norm, the learning rate needs to be decreased 5 times to achieve stable training.
* The Pre-Norm setup simplifies the gradient flow, making training more stable and allowing for a higher learning rate.

## NoPE vs. RoPE
<img width="1999" height="1051" alt="image1" src="https://github.com/user-attachments/assets/bf232c81-f040-4636-8969-65f2f103ad6b" />

* Adding relative position embedding allows the model to learn better as RoPE converges to a lower loss. 
* Prompting the model with NoPE shows that it can infer position information without being explicitly provided with position embeddings.

## SwiGLU vs. SiLU
<img width="1999" height="1051" alt="image6" src="https://github.com/user-attachments/assets/196260c0-08a6-4242-a85e-e246e4b7135e" />

* For SiLU, d_ ff is set to 4 × d_model, to approximately match the parameter count of the SwiGLU feed-forward network.

## number of KV heads in GQA 
<img width="1999" height="1051" alt="image8" src="https://github.com/user-attachments/assets/c73f0a28-3536-41aa-8c13-2d7016dd7f12" />

* GQA can provide up to an 8x reduction in memory for the KV cache with no degradation in perplexity.
