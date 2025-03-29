import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 定义自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# 定义 Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# 定义 Transformer 模型
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        return out

# 定义文本生成函数
def generate_text(model, device, start_text, max_length, temperature=1.0):
    model.eval()
    tokens = [char_to_idx[char] for char in start_text]
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    for _ in range(max_length):
        # 动态调整位置编码
        positions = torch.arange(0, tokens.size(1)).expand(tokens.size(0), tokens.size(1)).to(device)
        mask = torch.tril(torch.ones((1, tokens.size(1), tokens.size(1)))).to(device)
        
        preds = model(tokens, mask)
        preds = preds[:, -1, :] / temperature
        probs = F.softmax(preds, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 如果生成的字符不在词汇表中，停止生成
        if next_token.item() >= vocab_size:
            print(f"Generated token {next_token.item()} is out of vocabulary. Stopping generation.")
            break
        
        tokens = torch.cat([tokens, next_token], dim=1)

    generated_text = "".join([idx_to_char[idx.item()] for idx in tokens[0]])
    return generated_text

# 示例数据
text = "hello world, this is a simple text generation example."
chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}
vocab_size = len(chars)

# 超参数
embed_size = 32
num_layers = 2
heads = 4
forward_expansion = 4
dropout = 0.1
max_length = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = Transformer(
    vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length
).to(device)

# 生成文本
start_text = "hello"
generated_text = generate_text(model, device, start_text, max_length=50, temperature=0.7)
print("Generated Text:", generated_text)