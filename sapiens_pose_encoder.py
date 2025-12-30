## This file is adapted and reconstructed from "Sapiens: Foundation for Human Vision Models"

# https://arxiv.org/abs/2408.12569
# https://github.com/facebookresearch/sapiens

# @misc{khirodkar2024sapiensfoundationhumanvision,
#       title={Sapiens: Foundation for Human Vision Models}, 
#       author={Rawal Khirodkar and Timur Bagautdinov and Julieta Martinez and Su Zhaoen and Austin James and Peter Selednik and Stuart Anderson and Shunsuke Saito},
#       year={2024},
#       eprint={2408.12569},
#       archivePrefix={arXiv},
#       primaryClass={cs.CV},
#       url={https://arxiv.org/abs/2408.12569}, 
# }


## IMPORTS
import torch
import torch.nn as nn

class SapiensAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.out_drop = nn.Identity()
        self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return self.gamma1(self.out_drop(x))

class SapiensTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SapiensAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = SapiensFFN(dim, mlp_hidden)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), x)
        return x

class SapiensFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.0)
            ),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.0)
        )
        self.dropout_layer = nn.Identity() 
        self.gamma2 = nn.Identity()

    def forward(self, x, residual):
        return residual + self.gamma2(self.dropout_layer(self.layers(x)))

class SapiensPoseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=1536,
            kernel_size=(16, 16),
            stride=(16, 16),
            padding=(2, 2)
        )

        self.drop_after_pos = nn.Dropout(
            p=0.0, 
            inplace=False
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 3072, 1536))

        self.layers = nn.ModuleList([
            SapiensTransformerEncoderLayer(dim=1536, num_heads=24, mlp_hidden=6144) for _ in range(40)
        ])

        self.ln1 = nn.LayerNorm(1536, eps=1e-6)

    def forward(self, x):
        x = self.projection(x)
        h_int = x.size(2) 
        w_int = x.size(3)
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        for layer in self.layers:
            x = layer(x)
        
        x = self.ln1(x)

        B = x.size(0)
        
        x = x.reshape(B, h_int, w_int, -1)
        
        x = x.permute(0, 3, 1, 2)

        return x