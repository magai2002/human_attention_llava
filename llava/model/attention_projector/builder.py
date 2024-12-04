import torch.nn as nn

def build_attention_projector(config, delay_load=False, **kwargs):
    # return a single layer MLP with GELU activation
    return nn.Sequential(
        nn.Linear(config.mm_hidden_size, config.hidden_size),
        nn.GELU()
    )