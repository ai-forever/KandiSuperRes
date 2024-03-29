import torch
from torch import nn
from einops import rearrange
from .nn import Identity, Attention, SinusoidalPosEmb, UpDownResolution
from .utils import exist, set_default_item, set_default_layer
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, time_embed_dim=None, groups=32, activation=None, up_resolution=None, dropout=None):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, in_channels)
        self.activation = set_default_layer(
             exist(activation),
             nn.SiLU
         )
        self.change_resolution = set_default_layer(
             exist(up_resolution),
             UpDownResolution, (in_channels, up_resolution)
         )
        self.dropout = set_default_layer(
             exist(dropout),
             nn.Dropout, (), {'p': 0.1}
         )
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.group_norm(x)
        if exist(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)
        x = self.dropout(x)
        x = self.change_resolution(x)
        x = self.projection(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_embed_dim=None, groups=32, up_resolution=None):
        super().__init__()
        self.time_mlp = set_default_item(
            exist(time_embed_dim),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * out_channels)
            )
        )
        self.in_block = Block(in_channels, out_channels, time_embed_dim, groups, up_resolution=up_resolution)
        self.out_block = Block(out_channels, out_channels, time_embed_dim, groups, activation=True, up_resolution=None, dropout=True)

        self.change_resolution = set_default_layer(
            exist(up_resolution),
            UpDownResolution, (in_channels, up_resolution)
        )
        self.res_block = set_default_layer(
            in_channels != out_channels or exist(up_resolution),
            nn.Conv2d, (in_channels, out_channels), {'kernel_size': 1}
        )

    def forward(self, x, time_embed=None):
        scale_shift = None
        if exist(time_embed) and exist(self.time_mlp):
            time_embed = self.time_mlp(time_embed)
            time_embed = rearrange(time_embed, 'b c -> b c 1 1')
            scale_shift = time_embed.chunk(2, dim=1)
        out = self.in_block(x)
        out = self.out_block(out, scale_shift=scale_shift)
        x = self.change_resolution(x)
        out = out + self.res_block(x)
        return out


class AttentionBlock(nn.Module):

    def __init__(
            self, dim, context_dim=None, groups=32, num_heads=8, num_conditions=1, feed_forward_mult=2
    ):
        super().__init__()
        self.in_norm = nn.GroupNorm(groups, dim)
        self.attention = Attention(
            dim, context_dim or dim, num_heads, num_conditions=num_conditions
        )

        hidden_dim = feed_forward_mult * dim
        self.out_norm = nn.GroupNorm(groups, dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x, context=None, context_mask=None, context_idx=None):
        width = x.shape[-1]
        out = self.in_norm(x)
        out = rearrange(out, 'b c h w -> b (h w) c')
        context = set_default_item(exist(context), context, out)
        out = self.attention(out, context, context_mask, context_idx)
        out = rearrange(out, 'b (h w) c -> b c h w', w=width)
        x = x + out

        out = self.out_norm(x)
        out = self.feed_forward(out)
        x = x + out
        return x

    
class DownSampleBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_embed_dim,
            num_resnet_blocks=3, groups=32, down_sample=True, context_dim=None, self_attention=True, num_conditions=1):
        super().__init__()
        up_resolutions = [set_default_item(down_sample, False)] + [None] * (num_resnet_blocks - 1)
        hidden_channels = [(in_channels, out_channels)] + [(out_channels, out_channels)] * (num_resnet_blocks - 1)
        self.resnet_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(in_channel, out_channel, time_embed_dim, groups, up_resolution),
                set_default_layer(
                    exist(context_dim),
                    AttentionBlock, (out_channel, context_dim), {'num_conditions': num_conditions, 'groups': groups},
                    layer_2=Identity
                )
            ]) for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions)
        ])

        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock, (out_channels,), {'feed_forward_mult': 4, 'groups': groups},
            layer_2=Identity
        )

    def forward(self, x, time_embed, context=None, context_mask=None, context_idx=None):
        for resnet_block, attention in self.resnet_attn_blocks:
            x = resnet_block(x, time_embed)
            x = attention(x, context, context_mask, context_idx)
        x = self.self_attention_block(x)
        return x


class UpSampleBlock(nn.Module):

    def __init__(
            self, in_channels, cat_dim, out_channels, time_embed_dim,
            num_resnet_blocks=3, groups=32, up_sample=True, context_dim=None, self_attention=True, num_conditions=1):
        super().__init__()
        up_resolutions = [None] * (num_resnet_blocks - 1) + [set_default_item(up_sample, True)]
        hidden_channels = [(in_channels + cat_dim, in_channels)] + [(in_channels, in_channels)] * (num_resnet_blocks - 2) + [(in_channels, out_channels)]
        self.resnet_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(in_channel, out_channel, time_embed_dim, groups, up_resolution),
                set_default_layer(
                    exist(context_dim),
                    AttentionBlock, (out_channel, context_dim), {'num_conditions': num_conditions, 'groups': groups, 'feed_forward_mult': 4},
                    layer_2=Identity
                )
            ]) for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions)
        ])

        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock, (out_channels,), {'feed_forward_mult': 4, 'groups': groups},
            layer_2=Identity
        )

    def forward(self, x, time_embed, context=None, context_mask=None, context_idx=None):
        for resnet_block, attention in self.resnet_attn_blocks:
            x = resnet_block(x, time_embed)
            x = attention(x, context, context_mask, context_idx)
        x = self.self_attention_block(x)
        return x

    
class UNet(nn.Module):

    def __init__(self,
                 model_channels,
                 init_channels=128,
                 num_channels=3,
                 time_embed_dim=512,
                 context_dim=None,
                 groups=32,
                 feature_pooling_type='attention',
                 dim_mult=(1, 2, 4, 8),
                 num_resnet_blocks=(2, 4, 8, 8),
                 num_conditions=1,
                 skip_connect_scale=1.,
                 add_cross_attention=(False, False, False, False),
                 add_self_attention=(False, False, False, False),
                 lowres_cond=True,
                ):
        super().__init__()
        out_channels = num_channels
        num_channels = set_default_item(lowres_cond, num_channels * 2, num_channels)
        init_channels = init_channels or model_channels
        self.num_conditions = num_conditions
        self.skip_connect_scale = skip_connect_scale
        self.to_time_embed = nn.Sequential(
            SinusoidalPosEmb(init_channels),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.init_conv = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_resnet_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, res_block_num, groups, down_sample, text_dim, self_attention, num_conditions
                )
            )

        self.up_samples = nn.ModuleList([])
        for level, ((out_dim, in_dim), res_block_num, text_dim, self_attention) in enumerate(zip(reversed(in_out_dims), *rev_layer_params)):
            up_sample = level != 0
            self.up_samples.append(
                UpSampleBlock(
                    in_dim, cat_dims.pop(), out_dim, time_embed_dim, res_block_num, groups, up_sample, text_dim, self_attention, num_conditions
                )
            )
        
        self.norm = nn.GroupNorm(groups, init_channels)
        self.activation = nn.SiLU()
        self.out_conv = nn.Conv2d(init_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, time, context=None, context_mask=None, context_idx=None, lowres_img=None):
        if exist(lowres_img):
            _, _, new_height, new_width = x.shape
            upsampled = F.interpolate(lowres_img, (new_height, new_width), mode="bilinear")
            x = torch.cat([x, upsampled], dim=1)
        time_embed = self.to_time_embed(time)

        hidden_states = []
        x = self.init_conv(x)
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask, context_idx)
            if level != self.num_levels - 1:
                hidden_states.append(x)
        for level, up_sample in enumerate(self.up_samples):
            if level != 0:
                x = torch.cat([x, hidden_states.pop() / self.skip_connect_scale], dim=1)
            x = up_sample(x, time_embed, context, context_mask, context_idx)
        x = self.norm(x)
        x = self.activation(x)
        x = self.out_conv(x)
        return x


def get_unet(conf):
    unet = UNet(**conf)
    return unet