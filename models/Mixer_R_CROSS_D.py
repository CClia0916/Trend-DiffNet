import torch
import torch.nn as nn
import torch.nn.functional as F


class linear_layer(nn.Module):
    #
    def __init__(self, input_dim=2048, embed_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)
class PhaseCrossAttention(nn.Module):
    """
    Phase-aware Cross Attention Branch
    保留softmax，专属优化：query初始化+相位专属proj+位置编码
    Phases: [background, start, interior, end]
    """
    def __init__(self, dim, num_heads=4, temperature=0.07):
        super().__init__()
        self.num_phase = 4
        self.temperature = temperature

        # 【核心修改1：randn正态初始化+单位范数归一化，让4个query有初始区分】
        self.phase_queries = nn.Parameter(torch.randn(self.num_phase, dim))
        nn.init.normal_(self.phase_queries, std=0.02)  # 标准正态分布
        self.phase_queries.data = F.normalize(self.phase_queries.data, dim=-1)  # 单位范数，避免初始范数占优

        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )

        # 【核心修改2：替换共享proj为相位专属线性层，强化各相位特征区分】
        # 每个相位独立投影，避免共享层让start/end特征被bg/interior淹没
        self.phase_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(self.num_phase)
        ])

    def forward(self, x0):
        """
        x0: (B, T, D)
        return:
            phase_prob: (B, T, 4) softmax输出，和为1，保留互斥特性
            phase_token: (B, 4, D)
        """
        B, T, D = x0.shape

        # Cross-Attention：phase query → temporal features
        q = self.phase_queries.unsqueeze(0).expand(B, -1, -1)  # (B,4,D)
        phase_token, _ = self.attn(
            query=q,
            key=x0,
            value=x0,
            need_weights=False
        )  # (B,4,D)

        # 相位专属投影：每个相位token独立学习特征，避免共享层的特征淹没
        phase_token_list = []
        for i in range(self.num_phase):
            token = self.phase_projs[i](phase_token[:, i, :])  # (B,D)
            phase_token_list.append(token)
        phase_token = torch.stack(phase_token_list, dim=1)  # (B,4,D)

        # 反投影：frame ↔ phase，计算相似度logits
        x_norm = F.normalize(x0, dim=-1)
        p_norm = F.normalize(phase_token, dim=-1)
        logits = torch.einsum('btd,bkd->btk', x_norm, p_norm) / self.temperature

        # 【保留softmax，核心诉求】强制和为1，互斥相位判定
        phase_prob = F.softmax(logits, dim=-1)

        return phase_prob, phase_token

class Mixer(nn.Module):
    def __init__(self, inter_channels, embedding_dim):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = inter_channels

        self.linear_c3 = nn.Conv1d(c4_in_channels, embedding_dim, kernel_size=3, padding= 1)
        self.linear_c2 = nn.Conv1d(c3_in_channels, embedding_dim, kernel_size=3, padding= 1)
        self.linear_c1 = nn.Conv1d(c2_in_channels, embedding_dim, kernel_size=3, padding= 1)
        self.linear_fine = nn.Conv1d(c1_in_channels, embedding_dim, kernel_size=3, padding= 1)
        self.linear = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.phase_branch = PhaseCrossAttention(embedding_dim, num_heads=4, temperature=0.07)
    def forward(self, fine_feats, coarse_feats):
        coarse_f1, coarse_f2, coarse_f3 = coarse_feats

        coarse_f3 = self.linear_c3(coarse_f3)
        coarse_f3 = resize(coarse_f3, size=fine_feats.size()[2:],mode='linear',align_corners=False)

        coarse_f2 = self.linear_c2(coarse_f2)
        coarse_f2 = resize(coarse_f2, size=fine_feats.size()[2:],mode='linear',align_corners=False)

        coarse_f1 = self.linear_c1(coarse_f1)
        coarse_f1 = resize(coarse_f1, size=fine_feats.size()[2:],mode='linear',align_corners=False)

        fine_feats = self.linear_fine(fine_feats)
        phase_prob, phase_token = self.phase_branch(fine_feats.permute(0, 2, 1))
        phase_embed = torch.einsum(
            'btk,bkd->btd', phase_prob, phase_token
        )  # (B,T,D)

        phase_embed = phase_embed.permute(0, 2, 1)  # (B,D,T)
        phase_embed = self.linear(phase_embed)

        coarse_feats = coarse_f1 + coarse_f2 + coarse_f3 + phase_embed

        return  fine_feats, coarse_feats,phase_prob
