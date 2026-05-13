from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn as nn



import torch
import numpy
from einops import rearrange
import torch.nn.functional as F



class PoolXOR(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.save_vis = False     # ← 新增
        self.diff_feat = None
        self.common_feat = None
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.pad)
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=self.pad)
        self.weight = nn.Parameter(torch.full((dim,), 0.5))
        self.relu = nn.ReLU()
    def forward(self, x):
        B, T, D = x.shape
        x_feat = x.transpose(1, 2)

        local_avg = self.avg_pool(x_feat)
        diff = x_feat - local_avg
        local_max = self.max_pool(x_feat)

        if self.save_vis:
            self.diff_feat = diff.detach().cpu()       # (B,C,T)
            self.common_feat = local_max.detach().cpu()

        weight = self.relu(self.weight).unsqueeze(0).unsqueeze(-1)
        xor_feat = diff * weight + local_max * (1 - weight)
        return xor_feat.transpose(1, 2)



class Local_Relational_Block(nn.Module):  # 原模块不变
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., kernel_size=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.xor_op = PoolXOR(dim=hidden_features, kernel_size=kernel_size)
        self.TC1 = nn.Conv1d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.relu = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act = act_layer()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop(x)
        x_xor = self.xor_op(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.TC1(x_conv)
        x_conv = x_conv.transpose(1, 2)
        w = self.relu(self.fusion_weight)
        x_fused = w * x_xor + (1 - w) * x_conv
        x = self.act(x_fused)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


import torch
import torch.nn as nn
class EMA(nn.Module):
    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha  # 固定alpha，保留原逻辑

    def forward(self, x):
        # x: (B,T,D) → 输出趋势特征：(B,T,D)
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double, device=x.device), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)


class Global_Positional_Relational_Block(nn.Module):
    def __init__(self, dim, num_heads=8, max_len = 256, relative_positional_embedig = True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. EMA模块（无修改）
        self.ema = EMA(alpha=0.1)

        # 2. LRSA核心：下采样（T→T//2）+ 上采样（T//2→T）（复用原模块，新增Q下采样逻辑）
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)  # 统一用于Q和EMA特征的下采样
        self.upsample = lambda x, target_T: F.interpolate(
            x, size=target_T, mode='linear', align_corners=False
        )

        # 3. 原始特征自注意力（无修改）
        self.q_self = nn.Linear(dim, dim)
        self.kv_self = nn.Linear(dim, dim * 2)

        # 4. 交叉注意力投影层（无修改，Q下采样在forward中处理）
        self.q_cross = nn.Linear(dim, dim)
        self.kv_ema = nn.Linear(dim, dim * 2)

        # 5. 输出投影与融合权重（无修改）
        self.proj = nn.Linear(dim, dim)
        self.fusion_weight = nn.Parameter(torch.tensor(0.2))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



    def to_low_res(self, feat):
        # (B,T,D) → (B,D,T) → 池化→(B,D,T//2) → (B,T//2,D)
        feat_permuted = feat.transpose(1, 2)
        feat_low_permuted = self.downsample(feat_permuted)
        return feat_low_permuted.transpose(1, 2)

    def forward(self, x):
        B, T, D = x.shape
        low_res_T = T // 2  # 低分辨率时间步
        target_T = T  # 上采样目标时间步（恢复原始T）

        # -------------------------- 步骤1：原始特征自注意力（无修改） --------------------------
        q_self = self.q_self(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_self = self.kv_self(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_self, v_self = kv_self[0], kv_self[1]
        attn_self = (q_self @ k_self.transpose(-2, -1)) * self.scale
        attn_self = attn_self.softmax(dim=-1)

        if getattr(self, "save_attn", False):
            # (B, H, T, T)
            self.self_attn_map = attn_self.detach().cpu()

        out_self = (attn_self @ v_self).transpose(1, 2).reshape(B, T, D)

        # -------------------------- 步骤2：交叉注意力（核心修改部分） --------------------------
        # 2.1 生成EMA趋势特征（无修改）
        ema_feat = self.ema(x)  # (B,T,D)
        q_trend = x - ema_feat  # (B,T,D)

        # 2. 趋势变化（ΔEMA）
        delta_ema = torch.zeros_like(ema_feat)
        delta_ema[:, 1:] = ema_feat[:, 1:] - ema_feat[:, :-1]

        # 3. Q / K / V 投影
        q = self.q_cross(q_trend)  # (B,T,D)
        kv = self.kv_ema(delta_ema)  # (B,T,2D)
        k, v = kv.chunk(2, dim=-1)

        # 4. Multi-head reshape
        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 5. Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if getattr(self, "save_attn", False):
            self.cross_attn_map = attn.detach().cpu()

        out_cross = (attn @ v).transpose(1, 2).reshape(B, T, D)

        # -------------------------- 步骤3：双路径融合（无修改） --------------------------
        w = self.sigmoid(self.fusion_weight)
        out = (1 - w) * out_self + w * out_cross
        out = self.proj(out)

        return out  # 输出：(B,T,D)，与输入格式一致



class Relative_Positional_Transformer_Block(nn.Module):
    """
    Global Local Relational Block
    """

    def __init__(self, dim, num_heads, max_len, mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, relative_positional_embedig = True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.Global__Positional_Relational_Block = Global_Positional_Relational_Block(
            dim,num_heads=num_heads, max_len = max_len, relative_positional_embedig = relative_positional_embedig)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Local_Relational_Block = Local_Relational_Block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.Global__Positional_Relational_Block(self.norm1(x))
        x = x + self.Local_Relational_Block(self.norm2(x))
        return x

class Temporal_Merging_Block(nn.Module):
    """
    Temporal_Merging_Block
    """

    def __init__(self, kernel_size=3, stride=1, in_chans=1024, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size// 2))
        # self.proj2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class Temporal_Merging_Block_F(nn.Module):  # 原模块不变
    def __init__(self, kernel_size=3, stride=1, in_chans=1024, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
        self.proj1 = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
        self.proj2 = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x) + self.proj1(x) + self.proj2(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class Fine_Detection_Module(nn.Module):
    def __init__(self, in_feat_dim=1024, embed_dims=[512, 512, 512, 512],
                 num_head=8, mlp_ratio=8, norm_layer=nn.LayerNorm,
                 num_block=3, num_clips = 256, relative_positional_embedig = True):
        super().__init__()

        # fine features
        self.Temporal_Merging_Block_f = Temporal_Merging_Block_F(kernel_size=3, stride=1, in_chans=in_feat_dim,
                                              embed_dim=embed_dims[0])
        self.fine = nn.ModuleList([Relative_Positional_Transformer_Block(
            dim=embed_dims[0], num_heads=num_head, max_len = num_clips, mlp_ratio=mlp_ratio,norm_layer=norm_layer, relative_positional_embedig=relative_positional_embedig)
            for i in range(num_block)])
        self.norm_f = norm_layer(embed_dims[0])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        fine_feats = self.Temporal_Merging_Block_f(inputs)
        # for i, blk in enumerate(self.fine):
        #     fine_feats = blk(fine_feats)
        # fine_feats = self.norm_f(fine_feats)
        fine_feats = fine_feats.permute(0, 2, 1).contiguous()
        return fine_feats

class Coarse_Detection_Module(nn.Module):
    def __init__(self, in_feat_dim=1024, embed_dims=[512, 512, 512, 512],
                 num_head=8, mlp_ratio=8, norm_layer=nn.LayerNorm,
                 num_block=3, num_clips = 256, relative_positional_embedig = True):
        super().__init__()
        # coarse features
        # level 2
        self.Temporal_Merging_Block_c1 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.coarse_1 = nn.ModuleList([Relative_Positional_Transformer_Block(
            dim=embed_dims[1], num_heads=num_head, max_len = num_clips // 2, mlp_ratio=mlp_ratio,norm_layer=norm_layer, relative_positional_embedig=relative_positional_embedig)
            for i in range(num_block)])
        self.norm_c1 = norm_layer(embed_dims[1])

        # level 3
        self.Temporal_Merging_Block_c2 = Temporal_Merging_Block(kernel_size=3, stride=4, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[2])
        self.Temporal_Merging_Block_c2_= Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[1],
                                                                embed_dim=embed_dims[2])
        self.linear2 = nn.Linear(embed_dims[2]*2, embed_dims[2])
        self.coarse_2 = nn.ModuleList([Relative_Positional_Transformer_Block(
            dim=embed_dims[2], num_heads=num_head, max_len = num_clips // 4, mlp_ratio=mlp_ratio,norm_layer=norm_layer, relative_positional_embedig=relative_positional_embedig)
            for i in range(num_block)])
        self.norm_c2 = norm_layer(embed_dims[2])

        # level 3
        self.Temporal_Merging_Block_c3 = Temporal_Merging_Block(kernel_size=3, stride=8, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[3])
        self.Temporal_Merging_Block_c3_ = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[2],
                                                                embed_dim=embed_dims[3])
        self.linear3 = nn.Linear(embed_dims[3] * 2, embed_dims[3])
        self.coarse_3 = nn.ModuleList([Relative_Positional_Transformer_Block(
            dim=embed_dims[3], num_heads=num_head, max_len = num_clips // 8, mlp_ratio=mlp_ratio,norm_layer=norm_layer, relative_positional_embedig=relative_positional_embedig)
            for i in range(num_block)])
        self.norm_c3 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, fine_feats):
        coarse_feats = []

        # coarse features (multi-scale, but non-hierarchical)
        coarse_feats_1 = self.Temporal_Merging_Block_c1(fine_feats)
        for i, blk in enumerate(self.coarse_1):
            coarse_feats_1 = blk(coarse_feats_1)
        coarse_feats_1 = self.norm_c1(coarse_feats_1)
        coarse_feats_1 = coarse_feats_1.permute(0, 2, 1).contiguous()
        coarse_feats.append(coarse_feats_1)

        coarse_feats_2 = self.Temporal_Merging_Block_c2(fine_feats)
        coarse_feats_2_ = self.Temporal_Merging_Block_c2_(coarse_feats_1)
        coarse_feats_2 = torch.cat((coarse_feats_2, coarse_feats_2_), -1)
        coarse_feats_2 = self.linear2(coarse_feats_2)
        for i, blk in enumerate(self.coarse_2):
            coarse_feats_2 = blk(coarse_feats_2)
        coarse_feats_2 = self.norm_c2(coarse_feats_2)
        coarse_feats_2 = coarse_feats_2.permute(0, 2, 1).contiguous()
        coarse_feats.append(coarse_feats_2)

        coarse_feats_3 = self.Temporal_Merging_Block_c3(fine_feats)
        coarse_feats_3_ = self.Temporal_Merging_Block_c3_(coarse_feats_2)
        coarse_feats_3 = torch.cat((coarse_feats_3, coarse_feats_3_), -1)
        coarse_feats_3 = self.linear2(coarse_feats_3)
        for i, blk in enumerate(self.coarse_3):
            coarse_feats_3 = blk(coarse_feats_3)
        coarse_feats_3 = self.norm_c3(coarse_feats_3)
        coarse_feats_3 = coarse_feats_3.permute(0, 2, 1).contiguous()
        coarse_feats.append(coarse_feats_3)

        return coarse_feats
