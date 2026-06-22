"""
pretrain_dtpose_style.py  —  阶段一: 域一致表征预训练 (DT-Pose 范式, 移植到你的 MM-Fi)
==========================================================================
复刻 DT-Pose (github.com/cseeyangchen/DT-Pose, TPAMI'26) 的预训练机制, 它在
【同硬件同数据同口径】的 MM-Fi S3/P3 上做到 MPJPE 316.8 (你当前 ~358)。
我们的探针已证伪"换 Transformer 前端"这条路; DT-Pose 的增益来自这套自监督
预训练范式 (论文 Table 5: 预训练把 MPJPE 198->165), 而非架构。

★ 与你当年"MAE 负结果"的关键差异 (这就是当年失败的原因):
  1. 输入: 单帧、3通道【纯幅度】(3,114,10) 当图像。不含相位、不做时序展开。
     (DT-Pose feeder L581: 只取 CSIamp, 全局 min-max)
  2. 掩码: unstructured 随机 patch, ratio=0.80
  3. 时序对比 TC-CL: 相邻帧(frame t, t+1)做正对, batch内其他为负对, InfoNCE
     —— 论文说贡献最大, 你当年大概率没有
  4. uniformity 正则: 防 WiFi 稀疏信号维度坍缩 (权重0.01) —— 你当年没有
  5. 编码器极小: 4层 ViT, emb256, patch(2,2)

输出: 一个预训练好的 encoder, 供阶段二 (train_pose_dtpose_style.py) 冻结使用。

★ 数据布局 (你的): /home/a123456/PerceptAlign/MMFi/<E>/<S>/<A>/wifi-csi/frame*.mat
  每 .mat 有 CSIamp (3,114,10)。本脚本逐帧训练(像 DT-Pose), 不切64窗。

用法 (S3 跨场景预训练, 对齐论文):
  python pretrain_dtpose_style.py \
      --data_root /home/a123456/PerceptAlign/MMFi \
      --train_envs E01 E02 E03 \
      --mask_ratio 0.80 --emb_dim 256 --encoder_layer 4 \
      --batch_size 4096 --max_device_batch 256 \
      --total_epoch 400 --warmup_epoch 40 --base_lr 1.5e-4 \
      --save_path pretrain_dtpose.pt

  ★ batch_size 4096 靠梯度累积 (max_device_batch 256 => 累积16步), 对齐 DT-Pose。
    4080 显存不够可调 --max_device_batch 128。400 epoch 较久, 这是它"算力堆域不变
    表征"的代价; 想先验证可 --total_epoch 100 看 val 重建 loss 趋势。
"""
import os, glob, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

# ----------------------------------------------------------------------
# 内置 ViT Block + trunc_normal_ (无 timm 依赖, 等价 timm 默认 ViT Block:
# pre-norm, qkv_bias=True, GELU, mlp_ratio=4, 残差 —— 预训练权重语义不变)
# ----------------------------------------------------------------------
import math as _math
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def _ncdf(x): return (1. + _math.erf(x / _math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = _ncdf((a-mean)/std); u = _ncdf((b-mean)/std)
        tensor.uniform_(2*l-1, 2*u-1); tensor.erfinv_()
        tensor.mul_(std*_math.sqrt(2.)); tensor.add_(mean); tensor.clamp_(min=a, max=b)
    return tensor

class _Mlp(nn.Module):
    def __init__(self, dim, hidden=None, drop=0.):
        super().__init__(); hidden = hidden or dim*4
        self.fc1 = nn.Linear(dim, hidden); self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim); self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class _Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads; self.scale = (dim//num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = ((q @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim*mlp_ratio), drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

ENV_SUBJECTS = {'E01': range(1, 11), 'E02': range(11, 21),
                'E03': range(21, 31), 'E04': range(31, 41)}


# ----------------------------------------------------------------------
# 数据: 逐帧, 3通道纯幅度, 全局 min-max (口径同 DT-Pose feeder.read_frame)
# 每个样本额外取下一帧 (相邻帧对比正对)
# ----------------------------------------------------------------------
def _read_amp(mat_path):
    """读单帧 CSIamp (3,114,10), 与 DT-Pose feeder.read_frame 完全一致:
    inf->nan; 逐 packet(最后一维)用该 packet 整体非nan均值填充; 全局 min-max。"""
    d = loadmat(mat_path)['CSIamp'].astype(np.float32)   # (3,114,10)
    d[np.isinf(d)] = np.nan
    for i in range(d.shape[-1]):                         # 逐 packet
        temp = d[:, :, i]
        if np.isnan(temp).any():
            m = temp[~np.isnan(temp)].mean() if (~np.isnan(temp)).any() else 0.0
            temp[np.isnan(temp)] = m
            d[:, :, i] = temp
    dmin, dmax = d.min(), d.max()
    d = (d - dmin) / (dmax - dmin + 1e-8)
    return d.astype(np.float32)


class FramePairDataset(torch.utils.data.Dataset):
    """逐帧样本; training 时附带下一帧。返回 (amp_t, amp_t1)。"""
    def __init__(self, data_root, envs, training=True):
        self.training = training
        self.items = []          # (cur_path, next_path or None)
        for env in envs:
            for sid in ENV_SUBJECTS[env]:
                subj = f'S{sid:02d}'
                for aid in range(1, 28):
                    act = f'A{aid:02d}'
                    csi_dir = os.path.join(data_root, env, subj, act, 'wifi-csi')
                    if not os.path.isdir(csi_dir):
                        continue
                    frames = sorted(glob.glob(os.path.join(csi_dir, 'frame*.mat')))
                    n = len(frames)
                    if n < 2:
                        continue
                    for i in range(n - 1):          # 留最后一帧没有 next
                        nxt = frames[i + 1] if training else None
                        self.items.append((frames[i], nxt))
        print(f'  FramePairDataset: {len(self.items)} 帧 (envs={envs}, training={training})')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cur, nxt = self.items[idx]
        a = torch.from_numpy(_read_amp(cur))         # (3,114,10)
        if self.training and nxt is not None:
            a1 = torch.from_numpy(_read_amp(nxt))
            return a, a1
        return a, a


# ----------------------------------------------------------------------
# MAE (照搬 DT-Pose model.py 的 MAE_Encoder/Decoder/ViT, 仅整理)
# ----------------------------------------------------------------------
def random_indexes(size):
    fwd = np.arange(size); np.random.shuffle(fwd)
    bwd = np.argsort(fwd)
    return fwd, bwd

def take_indexes(seq, idx):
    return torch.gather(seq, 0, repeat(idx, 't b -> t b c', c=seq.shape[-1]))

class PatchShuffle(nn.Module):
    def __init__(self, ratio): super().__init__(); self.ratio = ratio
    def forward(self, patches):
        T, B, C = patches.shape
        remain = int(T * (1 - self.ratio))
        idx = [random_indexes(T) for _ in range(B)]
        fwd = torch.as_tensor(np.stack([i[0] for i in idx], -1), dtype=torch.long, device=patches.device)
        bwd = torch.as_tensor(np.stack([i[1] for i in idx], -1), dtype=torch.long, device=patches.device)
        patches = take_indexes(patches, fwd)[:remain]
        return patches, fwd, bwd

class MAE_Encoder(nn.Module):
    def __init__(self, image_size=(114,10), patch_size=(2,2), emb_dim=256,
                 num_layer=4, num_head=4, input_dim=3, mask_ratio=0.80):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1,1,emb_dim))
        np_h = image_size[0]//patch_size[0]; np_w = image_size[1]//patch_size[1]
        self.pos_embedding = nn.Parameter(torch.zeros(np_h*np_w, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.patchify = nn.Conv2d(input_dim, emb_dim, patch_size, patch_size)
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)
        trunc_normal_(self.cls_token, std=.02); trunc_normal_(self.pos_embedding, std=.02)

    def feature_extract(self, img):
        """无掩码全图 -> cls 特征 (阶段二/对比/uniformity 用)。"""
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c') + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], 0)
        patches = rearrange(patches, 't b c -> b t c')
        return self.layer_norm(self.transformer(patches))[:,0,:]

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c') + self.pos_embedding
        patches, fwd, bwd = self.shuffle(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], 0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        return rearrange(features, 'b t c -> t b c'), bwd

class MAE_Decoder(nn.Module):
    def __init__(self, image_size=(114,10), patch_size=(2,2), emb_dim=256,
                 num_layer=2, num_head=4, output_dim=3):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1,1,emb_dim))
        np_h = image_size[0]//patch_size[0]; np_w = image_size[1]//patch_size[1]
        self.pos_embedding = nn.Parameter(torch.zeros(np_h*np_w+1, 1, emb_dim))
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = nn.Linear(emb_dim, output_dim*patch_size[0]*patch_size[1])
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)',
                                   p1=patch_size[0], p2=patch_size[1], h=np_h)
        trunc_normal_(self.mask_token, std=.02); trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, bwd):
        T = features.shape[0]
        bwd = torch.cat([torch.zeros(1, bwd.shape[1]).to(bwd), bwd+1], 0)
        features = torch.cat([features, self.mask_token.expand(bwd.shape[0]-features.shape[0], features.shape[1], -1)], 0)
        features = take_indexes(features, bwd) + self.pos_embedding
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')[1:]
        patches = self.head(features)
        mask = torch.zeros_like(patches); mask[T-1:] = 1
        mask = take_indexes(mask, bwd[1:]-1)
        return self.patch2img(patches), self.patch2img(mask)

class MAE_ViT(nn.Module):
    def __init__(self, image_size=(114,10), patch_size=(2,2), emb_dim=256,
                 encoder_layer=4, encoder_head=4, decoder_layer=2, decoder_head=4,
                 input_dim=3, mask_ratio=0.80):
        super().__init__()
        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, input_dim, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, input_dim)
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU())

    def forward(self, img, flag='train'):
        features, bwd = self.encoder(img)
        if flag == 'test':
            unmasked = self.encoder.feature_extract(img)
            pred, mask = self.decoder(features, bwd)
            return pred, mask, unmasked
        pred, mask = self.decoder(features, bwd)
        cl_feature = self.predictor(features[1:,:,:].mean(0))     # 投影头, 对比用
        return pred, mask, features[0], cl_feature                # features[0]=cls, uniformity用


# ----------------------------------------------------------------------
# 两个正则 (照搬 DT-Pose utils.py)
# ----------------------------------------------------------------------
def uniformity_loss(features):
    features = F.normalize(features, dim=1)
    sim = torch.mm(features, features.T)
    n = features.shape[0]
    sim = sim.masked_fill(torch.eye(n, device=features.device).bool(), 0)
    return torch.mean(sim**2)

def infonce_loss(batch_data, temperature=0.5):
    n = batch_data.shape[0]//2
    f1, f2 = batch_data[:n], batch_data[n:]
    sim = torch.mm(f1, f2.T)/temperature
    pos = torch.diag(sim).view(n,1)
    mask = torch.eye(n, device=batch_data.device).bool()
    neg = sim[~mask].view(n,-1)
    logits = torch.cat([pos, neg], 1)
    labels = torch.zeros(n, dtype=torch.long, device=batch_data.device)
    return F.cross_entropy(logits, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--train_envs', nargs='+', default=['E01','E02','E03'])
    ap.add_argument('--val_env', default='E04')
    ap.add_argument('--mask_ratio', type=float, default=0.80)
    ap.add_argument('--emb_dim', type=int, default=256)
    ap.add_argument('--encoder_layer', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=4096)
    ap.add_argument('--max_device_batch', type=int, default=256)
    ap.add_argument('--total_epoch', type=int, default=400)
    ap.add_argument('--warmup_epoch', type=int, default=40)
    ap.add_argument('--base_lr', type=float, default=1.5e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_path', default='pretrain_dtpose.pt')
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_bs = min(a.max_device_batch, a.batch_size)
    assert a.batch_size % load_bs == 0
    accum = a.batch_size // load_bs
    print('='*70)
    print(f'  阶段一 DT-Pose式预训练  mask={a.mask_ratio} emb={a.emb_dim} enc_layer={a.encoder_layer}')
    print(f'  有效batch={a.batch_size} (device_bs={load_bs} x accum={accum})  epoch={a.total_epoch}')
    print('='*70)

    train_set = FramePairDataset(a.data_root, a.train_envs, training=True)
    val_set = FramePairDataset(a.data_root, [a.val_env], training=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=load_bs, shuffle=True,
                                               num_workers=a.num_workers, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=load_bs, shuffle=False,
                                             num_workers=a.num_workers, pin_memory=True)

    model = MAE_ViT(image_size=(114,10), patch_size=(2,2), emb_dim=a.emb_dim,
                    encoder_layer=a.encoder_layer, encoder_head=4,
                    decoder_layer=2, decoder_head=4, input_dim=3, mask_ratio=a.mask_ratio).to(dev)
    nparam = sum(p.numel() for p in model.parameters())/1e6
    print(f'  MAE params={nparam:.2f}M')

    optim = torch.optim.AdamW(model.parameters(), lr=a.base_lr*a.batch_size/256,
                              betas=(0.9,0.95), weight_decay=a.weight_decay)
    lr_func = lambda ep: min((ep+1)/(a.warmup_epoch+1e-8),
                             0.5*(math.cos(ep/a.total_epoch*math.pi)+1))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_func)

    step = 0; optim.zero_grad()
    for epoch in range(a.total_epoch):
        model.train()
        L, Lm, Lu, Lc = [], [], [], []
        for amp_t, amp_t1 in train_loader:
            step += 1
            amp_t = amp_t.to(dev); amp_t1 = amp_t1.to(dev)
            n = amp_t.shape[0]
            csi = torch.cat([amp_t, amp_t1], 0)                  # 相邻帧拼一起
            pred, mask, feat_cls, cl_feature = model(csi, 'train')
            loss_mse = torch.mean((pred[:n] - amp_t)**2 * mask[:n]) / a.mask_ratio
            loss_unif = uniformity_loss(feat_cls[:n])
            loss_cl = infonce_loss(cl_feature, temperature=0.5)
            cl_lambda = min(1e-4 + (1e-2 - 1e-4)*(epoch/a.total_epoch), 1e-2)
            loss = loss_mse + 0.01*loss_unif + cl_lambda*loss_cl
            loss.backward()
            if step % accum == 0:
                optim.step(); optim.zero_grad()
            L.append(loss.item()); Lm.append(loss_mse.item())
            Lu.append(0.01*loss_unif.item()); Lc.append(cl_lambda*loss_cl.item())
        sched.step()
        if epoch % 5 == 0 or epoch == a.total_epoch-1:
            print(f'  ep{epoch:03d} lr={optim.param_groups[0]["lr"]:.2e}  '
                  f'loss={np.mean(L):.4f} (mse={np.mean(Lm):.4f} unif={np.mean(Lu):.5f} cl={np.mean(Lc):.5f})')
        # val 重建
        if (epoch+1) % 50 == 0 or epoch == a.total_epoch-1:
            model.eval(); vl = []
            with torch.no_grad():
                for amp_t, _ in val_loader:
                    amp_t = amp_t.to(dev)
                    pred, mask, _ = model(amp_t, 'test')
                    pred = pred*mask + amp_t*(1-mask)
                    vl.append((torch.mean((pred-amp_t)**2*mask)/a.mask_ratio).item())
            print(f'    [val E04 重建] epoch{epoch}  recon_loss={np.mean(vl):.4f}')
        torch.save(model, a.save_path)
    print(f'\n预训练完成, encoder 已存入 {a.save_path}  (阶段二用 --pretrained {a.save_path})')


if __name__ == '__main__':
    main()