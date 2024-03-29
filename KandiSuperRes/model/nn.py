import math

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from .utils import exist, set_default_layer


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def forward(x, *args, **kwargs):
        return x


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j').to(dtype=x.dtype)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ParallelLayerNorm(nn.Module):

    def __init__(self, normalized_shape, num_layers=1, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine) for _ in range(num_layers)
        ])

    def forward(self, x, layer_idx=None):
        if exist(layer_idx):
            batch_idx = torch.argsort(
                torch.sort(layer_idx, stable=True).indices
            )
            x = torch.cat([
                layer_norm(x[layer_idx == i]) for i, layer_norm in enumerate(self.layer_norms)
            ])[batch_idx]
        else:
            x = self.layer_norms[0](x)
        return x


class UpDownResolution(nn.Module):

    def __init__(self, num_channels, up_resolution, change_type='conv'):
        super().__init__()
        if change_type == 'pooling':
            self.change_resolution = set_default_layer(
                up_resolution,
                layer_1=nn.Upsample, kwargs_1={'scale_factor': 2., 'mode': 'nearest'},
                layer_2=nn.AvgPool2d, kwargs_2={'kernel_size': 2, 'stride': 2}
            )

        elif change_type == 'conv':
            self.change_resolution = set_default_layer(
                up_resolution,
                nn.ConvTranspose2d, (num_channels, num_channels), {'kernel_size': 4, 'stride': 2, 'padding': 1},
                nn.Conv2d, (num_channels, num_channels), {'kernel_size': 4, 'stride': 2, 'padding': 1},
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.change_resolution(x)
        return x
    
    
class ParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, num_layers=1, bias=True):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=bias) for _ in range(num_layers)
        ])

    def forward(self, x, layer_idx=None):
        if exist(layer_idx):
            batch_idx = torch.argsort(
                torch.sort(layer_idx, stable=True).indices
            )
            x = torch.cat([
                linear(x[layer_idx == i]) for i, linear in enumerate(self.linears)
            ])[batch_idx]
        else:
            x = self.linears[0](x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, context_dim, num_heads=8, num_conditions=1):
        super().__init__()
        assert dim % num_heads == 0
        self.scale = (dim // num_heads) ** -0.5
        self.num_heads = num_heads

        self.to_query = nn.Linear(dim, dim, bias=False)
        self.to_key = ParallelLinear(context_dim, dim, num_conditions, bias=False)
        self.to_value = ParallelLinear(context_dim, dim, num_conditions, bias=False)

        self.output_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context, context_mask=None, context_idx=None):
        query = rearrange(self.to_query(x), 'b n (h d) -> b h n d', h=self.num_heads)
        key = rearrange(self.to_key(context, context_idx), 'b n (h d) -> b h n d', h=self.num_heads)
        value = rearrange(self.to_value(context, context_idx), 'b n (h d) -> b h n d', h=self.num_heads)

        attention_matrix = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        if exist(context_mask):
            max_neg_value = -torch.finfo(attention_matrix.dtype).max
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            attention_matrix = attention_matrix.masked_fill(~context_mask.bool(), max_neg_value)
        attention_matrix = attention_matrix.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attention_matrix, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.output_layer(out)
        return out