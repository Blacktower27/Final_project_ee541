import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinTransformer_Block(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, dropout=0.0):
        super(SwinTransformer_Block, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.attn(x)
        x = x.view(B, H * W, C)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, H, W, C = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1:3]
        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size * self.window_size, C)
        q = self.q_proj(x).reshape(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = x.view(-1, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        x = x[:, :H, :W, :]
        return self.proj(x)

if __name__ == "__main__":
    x = torch.randn(2, 64, 400, 400)
    block = SwinTransformer_Block(dim=64, num_heads=8, window_size=7)
    out = block(x)
    print(out.shape)
