import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

# encoder is consists of following five parts:
# input embedding
# positional encoding
# multi-head attention
# layer normalization
# feed forward network

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_length = 512):
        super().__init__() 
        
        #calculate positional embedding in log space for once
        pe = torch.zeros(max_length, d_model).float()
        pe.required_grad = False # do not need to set gradient, position embedding is set const ,do not need train
        
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout =0.1):
        super(PositionwiseFeedForward, self).__init__()
        # init first layer of full connect layers, input d_model, output d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # init second layer of full connect layers, input d_ff, output d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # init dropout layer, dropout stands for discard rate
        self.dropout = nn.Dropout(dropout)
        # use GELU as activation function
        self.activation = torch.nn.GELU()
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class Attention(nn.Module):
    """
        define Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    
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
    
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
         super(SublayerConnection, self).__init__()
         # init layer normalization
         self.norm = torch.nn.LayerNorm(size)
         # init dropout layer , discard rate
         self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
            x : input tensor
            sublayer: sublayer to use
        """
        return x+self.dropout(sublayer(self.norm(x)))
    
class TransformerBlock(nn.Module):
    """
        bidirectional encoder = Transformer (Self-Attention)
        Transformer = Multi-head Attention + feed forward network + sublayerConnection
    """
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
            :param hidden : size of transformer hidden layer
            :param attn_heads : num of multi-head attentions
            :param feed_forward_hidden : size of feed forward network's hidden layer, usually 4*hidden_size
            :param dropout : discard rate
        """
        super().__init__()
        # init multi head attention module
        self.attention = MultiHeadAttention(h= attn_heads, d_model= hidden)
        self.feed_forward = PositionwiseFeedForward(d_model= hidden, d_ff= feed_forward_hidden, dropout=dropout)
        # init input layer
        self.input_sublayer = SublayerConnection(size= hidden , dropout=dropout)
        # init output layer
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p= dropout)
        
    def forward(self,x , mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
        
        
              
        

encoder_vocab_size = 10
word_embedding_table = torch.nn.Embedding(num_embeddings=encoder_vocab_size, embedding_dim=768)
# encoder_embedding = word_embedding_table(inputs)

max_length = 128 
d_model = 512

pe = torch.zeros(max_length, d_model)
position = torch.arange(0., max_length).unsqueeze(1)

div_term = torch.exp(torch.arange(0., d_model, 2)* -(math.log(10000.0) / d_model))

pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

pe = pe.unsqueeze(0)

pe = pe.numpy()
pe = pe.squeeze()

plt.imshow(pe)
plt.colorbar()
plt.show()