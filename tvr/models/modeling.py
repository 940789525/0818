from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import numpy as np
import copy
allgather = AllGather.apply
allgather2 = AllGather2.apply

logger = logging.getLogger(__name__)

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class VTRModel(nn.Module):
    def __init__(self, config):
        super(VTRModel, self).__init__()
        
        self.config = config
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        self.lora_dim = config.lora_dim
        logger.info("v_LoRA: {} dim".format(self.lora_dim))
        
        assert backbone in _PT_NAME
        model_path = os.path.join(config.pretrained_path, _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        # Initialize CLIP (LoRA-only) without any token merging/TempMe-related parameters.
        self.clip = CLIP(
            embed_dim,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers,
            self.lora_dim,
        )

        self.loss_fct = CrossEn(config)
        self.clip.load_state_dict(state_dict, strict=False)

        # Note: merge-layer and TempMe/ToMe attributes removed.
        
    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)
        video_feat = self.get_lora_video_feat(video, video_mask)
        # video_feat = self.get_video_feat(video, video_mask)
        
        cls = allgather(cls, self.config)
        video_feat = allgather(video_feat, self.config)
        torch.distributed.barrier()
        
        logit_scale = self.clip.logit_scale.exp()
        loss = 0.
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True)
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])

        loss_t2v = self.loss_fct(t2v_logits * logit_scale)
        loss_v2t = self.loss_fct(t2v_logits.T * logit_scale)
        loss = (loss_t2v + loss_v2t) / 2
        
        return loss

    def stage1_eval(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)
        video = self.get_lora_video_feat(video, video_mask)
        # video = self.get_video_feat(video, video_mask)

        return cls, video

    def stage2_eval(self, cls, text_mask, video_feat, video_mask):
        logit_scale = self.clip.logit_scale.exp()
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True) 
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True) 

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])
        
        return t2v_logits * logit_scale

    def get_text_feat(self, text_ids, orig_mask):
        b = text_ids.size(0)
        x = self.clip.token_embedding(text_ids) 
        max_t_len = x.size(1)
        pos_emd = self.clip.positional_embedding[:max_t_len, :]
        x = x + pos_emd

        mask = orig_mask
        text_length = max_t_len
        attn_mask = self.clip.build_attention_mask(text_length).repeat(x.size(0), 1, 1).to(mask.device)
        inf = torch.zeros((text_length, text_length)).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        attn_mask = torch.where(mask>0, attn_mask, inf)
    
        x = self.clip.transformer(x, attn_mask)

        hidden = self.clip.ln_final(x) @ self.clip.text_projection
        cls = hidden[torch.arange(hidden.shape[0]), text_ids.argmax(dim=-1)]

        cls = cls.float()
        cls = cls.view(b, -1, cls.size(-1)).squeeze(1)
        return cls
    def get_lora_video_feat(self, video, video_mask):
        """
        video: (B*T, C, H, W)
        video_mask: (B, T)  1=有效帧, 0=padding
        return: (B, D) —— 视觉侧特征（带CLS跨帧注意力）
        """
        B, T = video_mask.shape
        BT, C, H, Wimg = video.shape
        assert BT == B * T, f"shape mismatch: video={video.shape}, mask={video_mask.shape}"

        # --- patch embedding + pos + ln_pre（保持 NLD）---
        x = self.clip.visual.conv1(video)                       # (B*T, Wd, g, g)
        x = x.reshape(BT, x.shape[1], -1).permute(0, 2, 1)      # (B*T, N, Wd)

        cls = (self.clip.visual.class_embedding.to(x.dtype) +
            torch.zeros(BT, 1, x.shape[-1], dtype=x.dtype, device=x.device))
        x = torch.cat([cls, x], dim=1)                          # (B*T, 1+N, Wd)

        gate_sp = torch.relu(self.clip.visual.spatial_pos_gate)
        x = x + gate_sp * self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        # ====== TimeRouter：筛帧 + 筛 patch（在 transformer 之前）======
        x_bt_l_w = x.view(B, T, x.size(1), x.size(2))           # [B,T,1+N,Wd]
        x_bt_l_w, video_mask, meta = self.clip.visual.time_router(x_bt_l_w, video_mask)
        # 现在：x_bt_l_w->[B,K,1+N',Wd]；video_mask->[B,K]；meta 里有 K、N' 等

        # 展平送入 Transformer
        B, K, L_red, Wd = x_bt_l_w.shape
        x = x_bt_l_w.view(B * K, L_red, Wd)                     # (B*K, 1+N', Wd)

        # --- 视觉 Transformer（LoRA 在工程里注入；主干可冻结）---
        x = self.clip.visual.transformer(x)                     # (B*K, 1+N', Wd)

        # --- 取 CLS → ln_post → proj ---
        x = x[:, 0, :]                                          # (B*K, Wd)
        x = self.clip.visual.ln_post(x)
        x = x @ self.clip.visual.proj                           # (B*K, D)

        # ===== 轻量 CLS 跨帧注意力（唯一的“时序编码”路径）=====
        x = x.view(B, K, -1)                                    # <<< 用 K，不是 T！(B,K,D)
        x = self.clip.visual.temporal_mix_cls(x, video_mask)    # (B,K,D)

        # --- 帧间池化 + L2 ---
        m = video_mask.to(x.dtype).unsqueeze(-1)                # (B, K, 1)
        summed = (x * m).sum(dim=1)                             # (B, D)
        if self.training:
            pooled = summed
        else:
            denom = m.sum(dim=1).clamp_min(1e-6)
            pooled = summed / denom

        x = F.normalize(pooled, p=2, dim=-1, eps=1e-6)          # (B, D)
        return x


    
    def get_video_feat(self, video, video_mask):
        """
        This method was used by the original TempMe/ToMe implementation to merge tokens across frames.
        For the pure LoRA version this functionality is removed. Use get_lora_video_feat instead.
        """
        raise NotImplementedError("get_video_feat has been removed in the pure LoRA model")

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
