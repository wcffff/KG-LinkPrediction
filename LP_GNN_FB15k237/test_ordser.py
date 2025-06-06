import torch
import torch.nn as nn

dim =3 # dim 为 channel数

qk = nn.Linear(dim, 2* dim, bias = True)
# 输入x->(batch, h*w, channel(dim))
x = torch.randn(1, 16, dim)
b, n, c = x.shape
tmp1 = qk(x).reshape(b, n, 2, c)

print(tmp1, tmp1.shape)
tmp2 = tmp1.permute(2, 0, 1, 3)
print(tmp2, tmp2.shape)

q, k, v = tmp2[0], tmp2[1], x
print(q, k, v, q.shape, k.shape, v.shape)

class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        print(channel_dims, feature_dim)
        k_max = feature_dim // (2 * len(channel_dims))
        
        print(k_max)

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

print("Calculate RoPE")
print(q.shape)
rope = RoPE(shape=(4, 4, 3))

 