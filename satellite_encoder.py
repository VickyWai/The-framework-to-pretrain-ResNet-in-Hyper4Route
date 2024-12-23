from PIL import Image
import torchvision
import numpy as np
import torch
import collections
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., qkv_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, qkv_bias=qkv_bias))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., qkv_bias=False):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # nn.Linear(patch_dim, dim),
            # use conv2d to fit weight file
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qkv_bias)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

        self.emb_head = nn.Linear(dim, num_classes, bias=False)
        # w = torch.zeros((num_classes, dim), requires_grad=True)
        # self.emb_head.weight = nn.Parameter(w)

    def forward(self, img, mask=None):
        # img[b, c, img_h, img_h] > patches[b, p_h*p_w, dim]
        x = self.to_patch_embedding(img)
        x = x.flatten(2).transpose(1, 2)
        # ipdb.set_trace()
        b, n, _ = x.shape

        # cls_token[1, p_n*p_n*c] > cls_tokens[b, p_n*p_n*c]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # add(concat) cls_token to patch_embedding
        x = torch.cat((cls_tokens, x), dim=1)
        # add pos_embedding
        x = x + self.pos_embedding[:, :(n + 1)]
        # drop out
        x = self.dropout(x)

        # main structure of transformer
        x = self.transformer(x, mask)

        # use cls_token to get classification message
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.emb_head(x)
        return x
        # return self.mlp_head(x)


class SatelliteImageEncoderViT(nn.Module):
    def __init__(self, img_in_dim, img_out_dim, pretrained_weight_path, batch_size, mesh_num):
        super(SatelliteImageEncoderViT, self).__init__()
        self.input_dim = img_in_dim
        self.output_dim = img_out_dim
        self.batch_size = batch_size
        self.mesh_num = mesh_num
        self.weight_path = pretrained_weight_path
        self.model_ViT = ViT(image_size=self.input_dim,
                             patch_size=16,
                             num_classes=self.output_dim,
                             dim=768,
                             depth=12,
                             heads=12,
                             mlp_dim=3072,
                             dropout=0.1,
                             emb_dropout=0.1)
        state_dict = torch.load(pretrained_weight_path)
        self.model_ViT.load_state_dict(state_dict, strict=False)

    def forward(self, sudoku_img_list):
        sudoku_img_list = sudoku_img_list.reshape((self.batch_size, self.mesh_num, 3, self.input_dim, self.input_dim))
        vit_input = sudoku_img_list[0]
        for batch in range(1, self.batch_size):
            vit_input = torch.vstack((vit_input, sudoku_img_list[batch]))
        batch_img_emb_list = self.model_ViT(vit_input)
        # batch_img_emb_list = batch_img_emb_list.reshape(self.batch_size, self.mesh_num, -1)
        return batch_img_emb_list


def load_pretrain(model, pre_s_dict):
    s_dict = model.state_dict()
    # remove fc weights and bias
    pre_s_dict.pop('projector.0.weight')
    pre_s_dict.pop('projector.2.weight')
    # use new dict to store states, record missing keys
    missing_keys = []
    new_state_dict = collections.OrderedDict()
    for key in s_dict.keys():
        simclr_key = 'encoder.' + key
        if simclr_key in pre_s_dict.keys():
            new_state_dict[key] = pre_s_dict[simclr_key]
        else:
            new_state_dict[key] = s_dict[key]
            missing_keys.append(key)
    print('{} keys are not in the pretrain model:'.format(len(missing_keys)), missing_keys)
    # load new s_dict
    model.load_state_dict(new_state_dict)
    return model


class ImageEncoder(nn.Module):
    def __init__(self, img_in_dim, img_out_dim, pretrained_weight_path, batch_size, mesh_num):
        super(ImageEncoder, self).__init__()
        self.img_size = img_in_dim
        self.output_dim = img_out_dim
        self.batch_size = batch_size
        self.mesh_num = mesh_num
        self.simclr_resnet18_weight_path = pretrained_weight_path

        pretrain_simclr = torch.load(self.simclr_resnet18_weight_path)
        self.simclr_encoder = torchvision.models.resnet18(pretrained=False)
        self.simclr_encoder.fc = nn.Identity()
        self.simclr_encoder = load_pretrain(self.simclr_encoder, pretrain_simclr)

    def forward(self, img_group):
        output = self.simclr_encoder(img_group)
        return output
