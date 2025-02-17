import torch
import torch.nn as nn
from functools import partial


from torch import Tensor
from typing import Optional

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf


from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_
import math

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.models.layers import DropPath

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class EncoderBlock(nn.Module):

    def __init__(self,
                 dim,
                 mixer_cls,
                 norm_cls=nn.LayerNorm,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 drop_path=0.0):

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"


    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):

        if not self.fused_add_norm:
          if residual is None:
            residual = hidden_states
          else:
            residual = residual + self.drop_path(hidden_states)

          hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
          if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        else:
          fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
          if residual is None:
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual = residual,
                prenorm=True,
                residual_in_fp32= self.residual_in_fp32,
                eps= self.norm.eps
            )
          else:
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual = residual,
                prenorm=True,
                residual_in_fp32= self.residual_in_fp32,
                eps= self.norm.eps
            )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        return hidden_states, residual
    
# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = EncoderBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block



class MyVisionMamba(nn.Module):
    def __init__(self,
                 embed_dim=192,
                 d_state=16,
                 depth=24,
                 use_cls_token=True,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 use_abs_pos_embed= True,
                 img_size=256,
                 patch_size=16,
                 stride=16,
                 channels=3,
                 ssm_cfg=None,
                 drop_rate=0.0,
                 drop_path_rate= 0.1,
                 norm_epsilon= 1e-5,
                 rms_norm: bool = True,
                 initializer_cfg = None,
                 residual_in_fp32= True,
                 fused_add_norm= True,
                 use_bidirectional= False,
                 if_bimamba = False,
                 bimamba_type= "v2",
                 if_divide_out= True,
                 init_layer_scale= None,
                 device=None,
                 dtype=None,
                 **kwargs
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super(MyVisionMamba, self).__init__()

        self.embed_dim = embed_dim
        self.d_state = d_state
        self.use_cls_token = use_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.use_abs_pos_embed = use_abs_pos_embed
        self.num_tokens = 1 if use_cls_token else 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.channels = channels
        self.ssm_cfg = None
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm


        self.patch_embed = PatchEmbed(
            img_size= self.img_size,
            patch_size= self.patch_size,
            stride= self.stride,
            in_chans= self.channels,
            embed_dim= self.embed_dim
        )
        self.use_bidirectional = use_bidirectional
        self.num_patches = self.patch_embed.num_patches


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()


        if use_cls_token:
            if use_double_cls_token:
                self.head_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.tail_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if self.use_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList(
          [
              create_block(
                  embed_dim,
                  d_state,
                  ssm_cfg = ssm_cfg,
                  norm_epsilon = norm_epsilon,
                  rms_norm = rms_norm,
                  residual_in_fp32=residual_in_fp32,
                  fused_add_norm=fused_add_norm,
                  layer_idx=i,
                  if_bimamba=if_bimamba,
                  bimamba_type=bimamba_type,
                  drop_path=inter_dpr[i],
                  if_divide_out=if_divide_out,
                  init_layer_scale=init_layer_scale,
                  **factory_kwargs,
              )

              for i in range(depth)
          ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(self.embed_dim, eps=norm_epsilon, **factory_kwargs)


        # original init
        self.patch_embed.apply(segm_init_weights)

        if use_abs_pos_embed:
          trunc_normal_(self.pos_embed, std= .02)
        if use_cls_token:
          if use_double_cls_token:
            trunc_normal_(self.cls_token_head, std = .02)
            trunc_normal_(self.cls_token_tail, std= .02)
          else:
            trunc_normal_(self.cls_token, std= .02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def forward_features(self, x, inference_params=None):
        x  = self.patch_embed(x)
        B, M, _ = x.shape

        if self.use_cls_token:
            if self.use_double_cls_token:
                head_cls_token = self.head_cls_token.expand(B, -1, -1)
                tail_cls_token = self.tail_cls_token.expand(B, -1, -1)
                self.num_tokens = 2
                x = torch.cat([head_cls_token, x, tail_cls_token], dim=1)
                M = x.shape[1] # update because just added new value nums patches

            elif self.use_middle_cls_token:
                cls_token = self.cls_token.expand(B, -1, -1)
                token_position = M // 2
                x = torch.cat([x[:, :token_position], cls_token, x[:, token_position:]], dim=1)
                M = x.shape[1] # update because just added new value nums patches

            else:
                cls_token = self.cls_token.expand(B, -1, -1)
                token_position = 0
                x = torch.cat([cls_token, x], dim=1)
                M = x.shape[1]

        if self.use_abs_pos_embed:
            x += self.pos_embed
            x = self.pos_drop(x)

        print(x.shape)

        # TODO: rope ???


        # Mamba impl
        residual = None
        hidden_states = x

        if not self.use_bidirectional:
          for layer in self.layers:
            hidden_states, residual = layer(hidden_states,
                                            residual,
                                            inference_params= inference_params)

        else:
          for i in range(len(self.layers) // 2):
            hidden_states_f, residual_f = self.layers[i * 2](hidden_states,
                                                             residual,
                                                             inference_params= inference_params)

            hidden_states_b, residual_b = self.layers[i * 2 + 1](hidden_states.flip([1]),
                                                                 None if residual == None else residual.flip([1]),
                                                                 inference_params= inference_params)


        if not self.fused_add_norm:
          if residual is None:
            residual = hidden_states
          else:
            residual = residual + self.drop_path(hidden_states)

        else:
          fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
          hidden_states = fused_add_norm_fn(
              self.drop_path(hidden_states),
              self.norm_f.weight,
              self.norm_f.bias,
              eps= self.norm_f.eps,
              residual = residual,
              prenorm=False,
              residual_in_fp32= self.residual_in_fp32
          )

        return hidden_states

    def forward(self, x, inference_params=None):
      x = self.forward_features(x, inference_params)
      print(x.shape)
      return x

images = torch.randn(10, 3, 256, 256)
model = MyVisionMamba()
out = model(images)
