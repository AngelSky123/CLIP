"""
CSI-RSC-PoseDG 配置文件 v2.1
- 修复: parse_args([]) → parse_args() 以接受命令行参数
- Root-relative coordinates
- Stronger RSC
- Early stopping patience
"""
import argparse
import os


def get_config(inference_only=False):
    parser = argparse.ArgumentParser(description='CSI-RSC-PoseDG')

    # ======================== Dataset ========================
    parser.add_argument('--data_root', type=str,
                        default='/home/a123456/PerceptAlign/MMFi')
    parser.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    parser.add_argument('--test_env', type=str, default='E04')
    parser.add_argument('--num_subjects_per_env', type=int, default=10)
    parser.add_argument('--num_actions', type=int, default=27)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--num_joints', type=int, default=17)

    # ======================== CSI Input ========================
    parser.add_argument('--num_rx_antennas', type=int, default=3)
    parser.add_argument('--num_subcarriers', type=int, default=114)
    parser.add_argument('--num_packets', type=int, default=10)
    parser.add_argument('--csi_channels', type=int, default=9)

    # ======================== Dual-Branch Encoder ========================
    parser.add_argument('--amp_channels', type=int, default=3)
    parser.add_argument('--phase_channels', type=int, default=6)
    parser.add_argument('--encoder_hidden_dim', type=int, default=32)
    parser.add_argument('--encoder_out_dim', type=int, default=64)

    # ======================== Local Spatio-Temporal Encoder ========================
    parser.add_argument('--local_hidden_dim', type=int, default=64)
    parser.add_argument('--local_out_dim', type=int, default=64)
    parser.add_argument('--num_res3d_blocks', type=int, default=2)

    # ======================== RSC-I (stronger) ========================
    parser.add_argument('--rsc1_spatial_drop_pct', type=float, default=0.5)
    parser.add_argument('--rsc1_channel_drop_pct', type=float, default=0.5)
    parser.add_argument('--rsc1_batch_pct', type=float, default=0.5)

    # ======================== Global Temporal Modeler ========================
    parser.add_argument('--global_dim', type=int, default=128)
    parser.add_argument('--num_transformer_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--tcn_channels', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--tcn_kernel_size', type=int, default=3)
    parser.add_argument('--transformer_dropout', type=float, default=0.3)

    # ======================== RSC-II (stronger) ========================
    parser.add_argument('--rsc2_time_drop_pct', type=float, default=0.5)
    parser.add_argument('--rsc2_channel_drop_pct', type=float, default=0.5)
    parser.add_argument('--rsc2_batch_pct', type=float, default=0.5)

    # ======================== 3D Pose Decoder ========================
    parser.add_argument('--coarse_hidden_dim', type=int, default=256)
    parser.add_argument('--gcn_hidden_dim', type=int, default=128)
    parser.add_argument('--num_gcn_layers', type=int, default=3)

    # ======================== Loss Weights ========================
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--lambda3', type=float, default=2.0,
                        help='Weight for MotionGuidanceLoss (anti temporal collapse)')
    parser.add_argument('--lambda_hip', type=float, default=1.0,
                        help='Weight for HipPositionLoss (improve global hip prediction)')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=2.0,
                        help='Increased consistency weight for stronger RSC')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Weight for anti-collapse losses (diversity + temporal)')
    parser.add_argument('--delta', type=float, default=0.5,
                        help='Weight for action classification auxiliary loss')

    # ======================== Training ========================
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--accumulate_grad', '--accum', type=int, default=4)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience on eval MPJPE')

    # ======================== Logging ========================
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=3)

    # ======================== Parse ========================
    # v2.1 修复: parse_args([]) → parse_known_args() 以接受命令行参数
    # parse_args([]) 会忽略所有命令行参数, 导致 --alpha/--beta 等不生效
    # parse_known_args() 解析已知参数, 未知参数 (如 --protocol) 会被忽略并警告
    if inference_only:
        args = parser.parse_args([])  # 推理模式不需要命令行参数
    else:
        args, unknown = parser.parse_known_args()
        if unknown:
            import warnings
            warnings.warn(f"Ignoring unrecognized arguments: {unknown}")

    # Auto-append timestamp so each run has its own directory
    if not getattr(args, 'no_timestamp', False) and not inference_only:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(args.save_dir, exist_ok=True)
    return args