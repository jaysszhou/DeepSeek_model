import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math


class RoataryEmbedding(torch.nn.Module):
    """
    Roatary Embedding
    """
    def __init__(self, dim, scale_base=model_config.scale_base, use_xpos=True):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arrange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("scale", scale)

    def forward(self, seq_len, device=all_config.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** elt.Rearrange('n -> n 1')(power)# rearrange(power, )
        scale = torch.cat((scale, scale), dim=-1)
        
        return freqs, scale
    
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(pos, t , scale =1.):
        return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)

    if __name__ == '__main__':
        embedding = torch.randn(size=(1, 128, 512))
        print(rotate_half(embedding).shape)