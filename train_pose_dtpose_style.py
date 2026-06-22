"""
train_pose_dtpose_style.py  —  阶段二: 拓扑约束位姿解码 (DT-Pose 范式, 移植到你的 MM-Fi)
==========================================================================
冻结阶段一预训练 encoder, 接 task-prompt + GCN(骨架邻接) + Transformer 解码头,
直接回归【绝对】17关节 xyz, 裸 MPJPE 损失。评测口径 = 真·绝对 MPJPE / PA-MPJPE
(align=False), 与你的 eval_dtpose_faithful 及 DT-Pose 论文一致。

★ 关键 (照搬 DT-Pose train_pose.py / model.ViT_Pose_Decoder):
  - encoder 全冻结 (requires_grad=False), 只训 prompt+GCN+Transformer+fc
  - 解码: 全图(无掩码)特征 -> 取 patch 均值 -> 扩成17关节 -> +task_prompt
          -> GCN×3 (骨架邻接 A 归一化) -> Transformer×3 -> MLP -> (B,17,3)
  - 损失: mean(||pred - gt||_2)  直接绝对坐标, 无 anchor / 无 FK
  - 选点: 用 E04 (跨场景 val), 对齐 DT-Pose/你的 --select_on_e04 口径
  - 输入: 单帧 3通道纯幅度 (3,114,10), 同阶段一

★ MM-Fi 17关节骨架邻接 (同 DT-Pose model.py generate_adjacency_matrix mmfi-csi)

用法 (S3 跨场景):
  python train_pose_dtpose_style.py \
      --data_root /home/a123456/PerceptAlign/MMFi \
      --train_envs E01 E02 E03 --test_env E04 \
      --pretrained pretrain_dtpose.pt \
      --epochs 50 --batch_size 32 --lr 1e-3
"""
import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from einops import rearrange

# 复用阶段一的数据读取、MAE 定义、以及内置 trunc_normal_ (无 timm)
import pretrain_dtpose_style as _P1
from pretrain_dtpose_style import _read_amp, ENV_SUBJECTS, MAE_ViT, trunc_normal_

# 阶段一用 torch.save(整个model对象) 存盘, pickle 把类绑到了当时的 __main__。
# 加载时把阶段一的所有类注入本脚本的 __main__, 让 torch.load 能找到。
import sys as _sys
_main = _sys.modules['__main__']
for _cls in ['MAE_ViT', 'MAE_Encoder', 'MAE_Decoder', 'PatchShuffle',
             'Block', '_Mlp', '_Attention']:
    if hasattr(_P1, _cls):
        setattr(_main, _cls, getattr(_P1, _cls))


# ----------------------------------------------------------------------
# 数据: 逐帧 (amp, gt17x3)。绝对坐标, 米。
# ----------------------------------------------------------------------
class PoseFrameDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, envs):
        self.items = []
        for env in envs:
            for sid in ENV_SUBJECTS[env]:
                subj = f'S{sid:02d}'
                for aid in range(1, 28):
                    act = f'A{aid:02d}'
                    csi_dir = os.path.join(data_root, env, subj, act, 'wifi-csi')
                    gt_path = os.path.join(data_root, env, subj, act, 'ground_truth.npy')
                    if not os.path.isdir(csi_dir) or not os.path.exists(gt_path):
                        continue
                    frames = sorted(glob.glob(os.path.join(csi_dir, 'frame*.mat')))
                    gt = np.load(gt_path)
                    n = min(len(frames), gt.shape[0])
                    for i in range(n):
                        self.items.append((frames[i], gt_path, i))
        print(f'  PoseFrameDataset: {len(self.items)} 帧 (envs={envs})')
        self._gt_cache = {}

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fp, gt_path, i = self.items[idx]
        amp = torch.from_numpy(_read_amp(fp))                # (3,114,10)
        if gt_path not in self._gt_cache:
            self._gt_cache[gt_path] = np.load(gt_path).astype(np.float32)
        gt = torch.from_numpy(self._gt_cache[gt_path][i])    # (17,3)
        return amp, gt


# ----------------------------------------------------------------------
# 解码器 (照搬 DT-Pose model.ViT_Pose_Decoder, mmfi 分支)
# ----------------------------------------------------------------------
class GraphConvLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__(); self.fc = nn.Linear(in_f, out_f); self.relu = nn.ReLU()
    def forward(self, x, adj):
        return self.relu(self.fc(torch.matmul(adj, x)))

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model); self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1); self.dropout2 = nn.Dropout(0.1)
    def forward(self, src):
        src2, _ = self.self_attn(src, src, src, need_weights=False)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(F.relu(self.linear1(src)))
        return self.norm2(src + self.dropout2(src2))

def mmfi_adjacency(num_joints=17):
    """与 DT-Pose model.py generate_adjacency_matrix 完全一致:
    原始 0/1 对称邻接, 【不加自环、不做归一化】。
    DT-Pose 的 GraphConvLayer = relu(fc(A @ x)), 直接用此原始邻接。
    (之前误加了自环+对称归一化, 改变了 GCN 聚合方式, 导致 PA 退化。)"""
    adj = torch.zeros(num_joints, num_joints)
    conn = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],
            [8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    for i,j in conn:
        adj[i,j] = 1; adj[j,i] = 1
    return adj

class ViT_Pose_Decoder(nn.Module):
    def __init__(self, encoder, keypoints=17, coor_num=3):
        super().__init__()
        self.keypoints = keypoints; self.coor_num = coor_num
        # 冻结的预训练件
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.emb_dim = self.cls_token.size()[2]
        for p in self.parameters():
            p.requires_grad = False
        # 可训练: task prompt
        self.pose_prompt = nn.Parameter(torch.zeros(self.emb_dim, keypoints))
        trunc_normal_(self.pose_prompt, std=.02)
        # GCN x3
        self.g1 = GraphConvLayer(self.emb_dim, self.emb_dim)
        self.g2 = GraphConvLayer(self.emb_dim, self.emb_dim)
        self.g3 = GraphConvLayer(self.emb_dim, self.emb_dim)
        self.register_buffer('adj', mmfi_adjacency(keypoints))
        # Transformer x3
        self.t1 = CustomTransformerEncoderLayer(self.emb_dim, 4)
        self.t2 = CustomTransformerEncoderLayer(self.emb_dim, 4)
        self.t3 = CustomTransformerEncoderLayer(self.emb_dim, 4)
        self.fc = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim//4),
                                nn.ReLU(), nn.Linear(self.emb_dim//4, coor_num))

    def forward(self, img):
        B = img.shape[0]
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c') + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], 0)
        patches = rearrange(patches, 't b c -> b t c')
        feat = self.layer_norm(self.transformer(patches))
        feat = feat[:,1:,:].mean(1)                          # 全图 patch 均值 (B, emb)
        x = feat.unsqueeze(2).expand(B, self.emb_dim, self.keypoints)
        x = x + self.pose_prompt.unsqueeze(0).expand(B, -1, -1)
        x = x.permute(0, 2, 1)                               # (B,17,emb)
        x = self.g1(x, self.adj); x = self.g2(x, self.adj); x = self.g3(x, self.adj)
        x = self.t1(x); x = self.t2(x); x = self.t3(x)
        return self.fc(x)                                    # (B,17,3)


# ----------------------------------------------------------------------
# 评测 (绝对 MPJPE + PA-MPJPE, align=False, 同 DT-Pose calulate_error)
# ----------------------------------------------------------------------
def compute_similarity_transform(X, Y, compute_optimal_scale=True):
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X-muX, Y-muY
    nX = np.sqrt((X0**2).sum()); nY = np.sqrt((Y0**2).sum())
    X0 /= (nX+1e-8); Y0 /= (nY+1e-8)
    U,s,Vt = np.linalg.svd(X0.T @ Y0)
    V = Vt.T; R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:,-1] *= -1; s[-1] *= -1; R = V @ U.T
    if compute_optimal_scale:
        b = s.sum()*nX/(nY+1e-8)
    else:
        b = 1.0
    T = R; c = muX - b*muY@R
    return None, None, T, b, c

def calc_error(preds, gts):
    N = preds.shape[0]
    mpjpe = np.sqrt(((preds-gts)**2).sum(2)).mean(1)
    pampjpe = np.zeros(N)
    for n in range(N):
        _,_,T,b,c = compute_similarity_transform(gts[n], preds[n], True)
        fp = b*preds[n]@T + c
        pampjpe[n] = np.sqrt(((fp-gts[n])**2).sum(1)).mean()
    hip = np.sqrt(((preds[:,0]-gts[:,0])**2).sum(1)).mean()*1000
    return mpjpe.mean()*1000, pampjpe.mean()*1000, hip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--train_envs', nargs='+', default=['E01','E02','E03'])
    ap.add_argument('--test_env', default='E04')
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_path', default='pose_dtpose.pt')
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('='*70)
    print(f'  阶段二 DT-Pose式拓扑解码  (冻结encoder + prompt+GCN+Transformer)')
    print(f'  预训练权重: {a.pretrained}')
    print('='*70)

    train_set = PoseFrameDataset(a.data_root, a.train_envs)
    test_set = PoseFrameDataset(a.data_root, [a.test_env])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=a.batch_size, shuffle=True,
                                               num_workers=a.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=a.batch_size, shuffle=False,
                                              num_workers=a.num_workers, pin_memory=True)

    try:
        mae = torch.load(a.pretrained, map_location='cpu', weights_only=False)
    except TypeError:
        # torch < 2.0 没有 weights_only 参数
        mae = torch.load(a.pretrained, map_location='cpu')
    model = ViT_Pose_Decoder(mae.encoder, keypoints=17, coor_num=3).to(dev)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    nfroz = sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e6
    print(f'  可训练={ntrain:.2f}M  冻结={nfroz:.2f}M')

    # DT-Pose mmfi 分支: SGD lr=1e-3 wd=0.01, 无 scheduler (固定lr)
    optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=a.lr, weight_decay=a.weight_decay, momentum=0.9)

    best = {'mpjpe': 1e9, 'pa': 1e9, 'hip': 1e9, 'epoch': -1}
    for epoch in range(a.epochs):
        model.train()
        tl = []
        for amp, gt in train_loader:
            amp = amp.to(dev); gt = gt.to(dev)
            pred = model(amp)
            loss = torch.mean(torch.norm(pred-gt, dim=-1))   # 裸 MPJPE
            optim.zero_grad(); loss.backward(); optim.step()
            tl.append(loss.item())
        # DT-Pose mmfi 无 lr scheduler, 固定 lr
        # eval E04
        model.eval(); P, G = [], []
        with torch.no_grad():
            for amp, gt in test_loader:
                P.append(model(amp.to(dev)).cpu().numpy()); G.append(gt.numpy())
        P = np.concatenate(P); G = np.concatenate(G)
        mpjpe, pa, hip = calc_error(P, G)
        if mpjpe < best['mpjpe']:
            best = {'mpjpe':mpjpe, 'pa':pa, 'hip':hip, 'epoch':epoch}
            torch.save(model.state_dict(), a.save_path)
        print(f'  ep{epoch:02d} train_loss={np.mean(tl):.4f}  '
              f'E04 MPJPE={mpjpe:.2f}  PA={pa:.2f}  hip={hip:.1f}'
              f'{"  <-best" if epoch==best["epoch"] else ""}')

    print('\n'+'='*70)
    print(f'  [最优 E04] epoch{best["epoch"]}  MPJPE={best["mpjpe"]:.2f}  '
          f'PA-MPJPE={best["pa"]:.2f}  hip={best["hip"]:.1f}')
    print(f'  对标: DT-Pose S3/P3 MPJPE 316.8 / PA 104.2 | 你之前 ~358 / 103.24')
    print('='*70)


if __name__ == '__main__':
    main()