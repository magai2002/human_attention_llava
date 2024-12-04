import os
from .attention_encoder import AttentionModule, ModifiedResNetVisualEncoder


def build_attention_tower():
    # Return the attention module with the weights
    return AttentionModule(weights_path="human_visual_encoder.pth", device='cuda' if torch.cuda.is_available() else 'cpu')