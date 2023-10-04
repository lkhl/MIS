import math
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms


__all__ = [
    'DINOv1FeatureExtractor', 'DINOv2FeatureExtractor', 'MoCov3FeatureExtractor',
    'feature_extractor_registry', 'build_feature_extractor', 'build_transform'
]


class AlignedResize(object):

    def __init__(self, scale_factor=1.0, stride=1):
        self.scale_factor = scale_factor
        self.stride = stride

    def __call__(self, img):
        w, h = img.size
        tgt_w, tgt_h = round(w * self.scale_factor), round(h * self.scale_factor)
        if tgt_w % self.stride != 0:
            tgt_w = tgt_w + self.stride - tgt_w % self.stride
        if tgt_h % self.stride != 0:
            tgt_h = tgt_h + self.stride - tgt_h % self.stride
        return img.resize((tgt_w, tgt_h))


class VisionTransformerMoCo(VisionTransformer):

    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. /
                            float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

        del self.head

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w),
             torch.cos(out_w),
             torch.sin(out_h),
             torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        # x = self.patch_embed(x)  # patch linear embedding
        x = self.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.patch_embed.norm(x)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward_features(self, x):
        # x = self.patch_embed(x)
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x = self.pos_drop(x + self.pos_embed)
        x = self.prepare_tokens(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class ViTFeatureExtractor(nn.Module, metaclass=ABCMeta):

    available_model_size: list[str] = []
    available_patch_size: list[int] = []

    def __init__(self, model_size: str, patch_size: int):
        self.model_size = str(model_size)
        self.patch_size = int(patch_size)
        assert self.model_size in self.available_model_size, f'Available model size: {self.available_model_size}'
        assert self.patch_size in self.available_patch_size, f'Available patch size: {self.available_patch_size}'
        super(ViTFeatureExtractor, self).__init__()
        self.model = self.build_model()

    @abstractmethod
    def build_model(self) -> nn.Module:
        pass

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass


class DINOv1FeatureExtractor(ViTFeatureExtractor):

    available_model_size = ['small', 'base']
    available_patch_size = [8, 16]

    def build_model(self):
        models = {'small': f'vits{self.patch_size}', 'base': f'vitb{self.patch_size}'}
        repo = 'facebookresearch/dino:main'
        model = f'dino_{models[self.model_size]}'
        print(f'Loading model {model} from repo {repo}')
        return torch.hub.load(repo, model)

    def forward(self, x):
        return self.model.get_intermediate_layers(x)[0][:, 1:]


class DINOv2FeatureExtractor(ViTFeatureExtractor):

    available_model_size = ['small', 'base', 'large', 'giant']
    available_patch_size = [14]

    def build_model(self):
        models = {'small': 'vits14', 'base': 'vitb14', 'large': 'vitl14', 'giant': 'vitg14'}
        repo = 'facebookresearch/dinov2'
        model = f'dinov2_{models[self.model_size]}'
        print(f'Loading model {model} from repo {repo}')
        return torch.hub.load(repo, model)

    def forward(self, x):
        return self.model.get_intermediate_layers(x)[0]


class MoCov3FeatureExtractor(ViTFeatureExtractor):

    available_model_size = ['small', 'base']
    available_patch_size = [16]

    def build_model(self):
        models = {
            'small': {
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 12
            },
            'base': {
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12
            }
        }
        weights = {
            'small': 'https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar',
            'base': 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
        }
        model = VisionTransformerMoCo(
            patch_size=self.patch_size,
            **models[self.model_size],
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        state_dict = torch.hub.load_state_dict_from_url(
            weights[self.model_size], map_location='cpu')['state_dict']
        converted_state_dict = OrderedDict()
        for p, v in state_dict.items():
            if 'momentum_encoder' in p or 'head' in p or 'predictor' in p:
                continue
            converted_state_dict[p[20:]] = v
        model.load_state_dict(converted_state_dict)
        return model

    def forward(self, x):
        return self.model.forward_features(x)[:, 1:]


feature_extractor_registry = {
    'dino_v1': DINOv1FeatureExtractor,
    'dino_v2': DINOv2FeatureExtractor,
    'moco_v3': MoCov3FeatureExtractor
}


def build_feature_extractor(model_size, patch_size, pretrained='dino_v1'):
    assert pretrained in feature_extractor_registry
    return feature_extractor_registry[pretrained](model_size, patch_size)


def build_transform(size=None, scale_factor=1.0, patch_size=8):
    if size is None:
        resize = AlignedResize(scale_factor=scale_factor, stride=patch_size)
    else:
        resize = transforms.Resize(size)
    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform
