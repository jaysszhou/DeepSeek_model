import torch
import torch.nn as nn
import math
import einops.layers.torch as elt

# 定义缩放点积注意力
class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

# define multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout= 0.1):
        super().__init__()
        assert d_model % h == 0
        
        # 假设d_v 始终等于d_k
        self.d_k = d_model // h 
        self.h = h
        
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.attention = Attention()
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linear_layers,(query, key, value))]
        
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x= x.transpose(1,2).contiguous().view(batch_size, -1, self.h* self.d_k)
        
        return self.output_linear(x)        

# 定义 Transformer 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = Attention()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x, mask=None):
        # 自注意力
        attn_output, _ = self.attention(x, x, x, mask, self.dropout)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接 + LayerNorm
        
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接 + LayerNorm
        
        return x

# 准备输入
vocab_size = 1024
embedding_dim = 312
hidden_dim = 256
batch_size = 5
seq_len = 80

# 定义输入 token
token = torch.randint(0, vocab_size, (batch_size, seq_len))

# 定义嵌入层
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# 将 token 转换为嵌入向量
input_embedding = embedding_layer(token)

# 创建 Transformer 编码器层实例
encoder_layer = TransformerEncoderLayer(embedding_dim, hidden_dim)

# 前向传播
output = encoder_layer(input_embedding)

print(output.shape)  # 输出: torch.Size([5, 80, 312])