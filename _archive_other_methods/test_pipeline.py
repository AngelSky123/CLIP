"""Pipeline验证脚本"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import get_config
from models import CSIRSCPoseDG
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator

args = get_config()
args.seq_len = 4
args.batch_size = 1
args.encoder_hidden_dim = 16
args.encoder_out_dim = 16
args.local_hidden_dim = 16
args.local_out_dim = 16
args.global_dim = 32
args.num_transformer_layers = 1
args.num_heads = 2
args.tcn_channels = [32]
args.coarse_hidden_dim = 64
args.gcn_hidden_dim = 32
args.num_res3d_blocks = 1

model = CSIRSCPoseDG(args)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model params: {n:,}')

csi = torch.randn(1, 4, 9, 114, 10)
pose = torch.randn(1, 4, 17, 3) * 0.5

# Inference
model.eval()
with torch.no_grad():
    out = model(csi)
print(f'Inference: p_final={out["p_final"].shape} z_local={out["z_local"].shape} z_global={out["z_global"].shape}')

# RSC forward
model.train()
ploss = PoseLoss()
out_rsc = model.forward_rsc(csi, pose, loss_fn=lambda p, g: ploss(p, g)[0])
print(f'RSC: masked={out_rsc["p_final_masked"].shape}')

# Loss
tfn = TotalLoss()
loss, ld = tfn(out_rsc, pose, training=True)
print(f'Loss: total={ld["l_total"]:.4f}')

# Backward
loss.backward()
print('Backward OK')

# Metrics
model.eval()
with torch.no_grad():
    o2 = model(csi)
ev = PoseEvaluator('meter')
m = ev.evaluate(o2['p_final'], pose)
for k, v in m.items():
    print(f'  {k}: {v:.2f}')

print('\nALL TESTS PASSED')