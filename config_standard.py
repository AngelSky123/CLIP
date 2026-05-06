"""
CSI-RSC-PoseDG 配置文件 — 标准 8:2 划分模式
整体数据集按 8:2 划分, 不区分环境, 测试模型整体性能.
"""
import argparse
import os


def get_config(inference_only=False):
    parser = argparse.ArgumentParser(description='CSI-RSC-PoseDG Standard Split')

    # ======================== Dataset ========================
    parser.add_argument('--data_root', type=str,
                        default='/home/a123456/PerceptAlign/MMFi')
    parser.add_argument('--all_envs', nargs='+', default=['E01', 'E02', 'E03', 'E04'])
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--split_mode', type=str, default='standard',
                        help='standard=8:2 random split across all data')
    parser.add_argument('--num_subjects_per_env', type=int, default=10)
    parser.add_argument('--num_actions', type=int, default=27)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--num_joints', type=int, default=17)

    # ======================== CSI Input ========================
    parser.add_argument('--num_rx_antennas', type=int, default=3)
    parser.add_argument('--num_subcarriers', type=int, default=114)
    parser.add_argument('--num_packets', type=int, default=10)
    parser.add_argument('--csi_channels', type=int, default=9)

    # ======================== Model ========================
    parser.add_argument('--amp_channels', type=int, default=3)
    parser.add_argument('--phase_channels', type=int, default=6)
    parser.add_argument('--encoder_hidden_dim', type=int, default=32)
    parser.add_argument('--encoder_out_dim', type=int, default=64)
    parser.add_argument('--local_hidden_dim', type=int, default=64)
    parser.add_argument('--local_out_dim', type=int, default=64)
    parser.add_argument('--num_res3d_blocks', type=int, default=2)
    parser.add_argument('--global_dim', type=int, default=128)
    parser.add_argument('--num_transformer_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--tcn_channels', type=list, default=[128, 128])
    parser.add_argument('--tcn_kernel_size', type=int, default=3)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--coarse_hidden_dim', type=int, default=256)
    parser.add_argument('--gcn_hidden_dim', type=int, default=128)
    parser.add_argument('--num_gcn_layers', type=int, default=3)

    # ======================== RSC (disabled for standard mode) ========================
    parser.add_argument('--rsc1_spatial_drop_pct', type=float, default=0.5)
    parser.add_argument('--rsc1_channel_drop_pct', type=float, default=0.5)
    parser.add_argument('--rsc1_batch_pct', type=float, default=0.5)
    parser.add_argument('--rsc2_time_drop_pct', type=float, default=0.5)
    parser.add_argument('--rsc2_channel_drop_pct', type=float, default=0.5)
    parser.add_argument('--rsc2_batch_pct', type=float, default=0.5)
    parser.add_argument('--use_rsc', action='store_true', default=False,
                        help='Enable RSC (default off for standard split)')

    # ======================== Loss ========================
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Anti-collapse (0 for standard)')
    parser.add_argument('--delta', type=float, default=0.5,
                        help='Action classification (0 for standard)')

    # ======================== Training ========================
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--accumulate_grad', type=int, default=4)
    parser.add_argument('--patience', type=int, default=15)

    # ======================== Logging ========================
    parser.add_argument('--save_dir', type=str, default='./checkpoints_standard')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=3)

    args = parser.parse_args([])

    if not inference_only:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(args.save_dir, exist_ok=True)
    return args