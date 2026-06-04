"""
Module 8: 3D 姿态解码头 v2 — 动作条件化

核心改动: Decoder 接收动作嵌入作为额外输入.
  - 训练时: 使用 GT 动作标签 (one-hot → embedding)
  - 推理时: 使用预测的动作概率 (softmax → weighted embedding)

这从架构上保证不同动作产生不同预测:
  decoder(z_global, action_A01) ≠ decoder(z_global, action_A15)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from taskprompt_decoder import TaskPromptCoarseHead

H36M_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

NUM_JOINTS = 17
NUM_BONES = len(H36M_BONES)


def build_adjacency_matrix(num_joints=17, bones=None, self_loop=True):
    if bones is None:
        bones = H36M_BONES
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in bones:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loop:
        A = A + np.eye(num_joints, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat


class CoarsePoseHead(nn.Module):
    """动作条件化粗姿态头.
    
    输入: z_global (B,T,C_g) + action_emb (B,D_a)
    输出: P_coarse (B,T,17,3)
    """

    def __init__(self, in_dim=256, hidden_dim=512, num_joints=17,
                 action_embed_dim=32):
        super().__init__()
        self.num_joints = num_joints
        # Input = z_global + action embedding
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + action_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_joints * 3),
        )

    def forward(self, z_global, action_emb):
        """
        Args:
            z_global: (B, T, C_g)
            action_emb: (B, D_a) — action embedding vector
        """
        B, T, _ = z_global.shape
        # Expand action embedding to all time steps
        act = action_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, D_a)
        z_cond = torch.cat([z_global, act], dim=-1)       # (B, T, C_g + D_a)
        out = self.mlp(z_cond)
        return out.reshape(B, T, self.num_joints, 3)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super().__init__()
        self.register_buffer('adj', torch.from_numpy(adj_matrix).float())
        self.W = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        support = self.W(x)
        out = torch.matmul(self.adj, support)
        BT, J, C = out.shape
        out = self.bn(out.reshape(BT * J, C)).reshape(BT, J, C)
        return out


class SkeletonRefiner(nn.Module):
    def __init__(self, in_features=3, hidden_dim=128, num_layers=3, num_joints=17):
        super().__init__()
        adj = build_adjacency_matrix(num_joints)
        self.input_proj = GraphConvLayer(in_features, hidden_dim, adj)
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim, adj))
        self.output_proj = nn.Linear(hidden_dim, 3)
        self.num_joints = num_joints

    def forward(self, p_coarse):
        B, T, J, _ = p_coarse.shape
        x = p_coarse.reshape(B * T, J, 3)
        x = F.gelu(self.input_proj(x))
        for gcn in self.gcn_layers:
            residual = x
            x = F.gelu(gcn(x)) + residual
        delta = self.output_proj(x)
        delta = delta.reshape(B, T, J, 3)
        return p_coarse + delta


class PoseDecoder(nn.Module):
    """动作条件化姿态解码器.
    
    输入: z_global (B,T,C_g), action_emb (B,D_a)
    输出: P_coarse, P_final (B,T,17,3)
    """

    def __init__(self, in_dim=256, hidden_dim=512, gcn_hidden=128,
                 num_gcn_layers=3, num_joints=17, action_embed_dim=32):
        super().__init__()
        # self.coarse_head = CoarsePoseHead(
        #     in_dim, hidden_dim, num_joints, action_embed_dim
        # )
        self.coarse_head = TaskPromptCoarseHead(
            in_dim, hidden_dim, num_joints, action_embed_dim
        )
        self.refiner = SkeletonRefiner(
            in_features=3, hidden_dim=gcn_hidden,
            num_layers=num_gcn_layers, num_joints=num_joints
        )

    def forward(self, z_global, action_emb):
        p_coarse = self.coarse_head(z_global, action_emb)
        p_final = self.refiner(p_coarse)
        return p_coarse, p_final


class ActionClassifier(nn.Module):
    """动作分类头 + 动作嵌入层."""

    def __init__(self, in_dim=128, num_actions=27, embed_dim=32):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, num_actions),
        )
        self.num_actions = num_actions
        self.embed_dim = embed_dim

    def forward(self, z_global):
        """Returns logits (B, num_actions)."""
        z_pooled = z_global.mean(dim=1)
        return self.classifier(z_pooled)

    def get_action_embedding(self, action_idx=None, action_probs=None):
        """Get action embedding vector.
        
        Args:
            action_idx: (B,) integer action indices (for training with GT)
            action_probs: (B, num_actions) soft probabilities (for inference)
        Returns:
            action_emb: (B, embed_dim)
        """
        if action_idx is not None:
            # Hard lookup (training with GT labels)
            return self.action_embed(action_idx)
        elif action_probs is not None:
            # Soft weighted average (inference with predicted probs)
            return action_probs @ self.action_embed.weight
        else:
            raise ValueError("Need either action_idx or action_probs")