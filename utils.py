"""
Utility functions for training and evaluation.
"""
import os
import sys
import json
import socket
import random
import logging
import time
import subprocess
from datetime import datetime
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


def _git_info():
    """Best-effort git commit/branch/dirty. Silent in non-git dirs."""
    info = {"commit": None, "branch": None, "dirty": None}
    try:
        root = os.path.dirname(os.path.abspath(sys.argv[0])) or "."

        def _run(cmd):
            return subprocess.check_output(
                cmd, cwd=root, stderr=subprocess.DEVNULL).decode().strip()

        info["commit"] = _run(["git", "rev-parse", "HEAD"])
        info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["dirty"] = bool(_run(["git", "status", "--porcelain"]))
    except Exception:
        pass
    return info


def _env_info():
    """Best-effort runtime/env info."""
    info = {"python": sys.version.split()[0], "hostname": socket.gethostname()}
    try:
        info["torch"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None)
        info["cudnn"] = (torch.backends.cudnn.version()
                         if torch.backends.cudnn.is_available() else None)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return info


def save_run_config(args, save_dir, extra=None, filename="run_config.json"):
    """Dump a reproducibility snapshot to <save_dir>/<filename>.

    Records full args, git commit/branch/dirty flag, runtime env, the exact
    command line, and a timestamp. Designed to NEVER raise: config archiving
    must not be able to crash a training run.

    Args:
        args:     argparse.Namespace, dict, or any object with __dict__.
        save_dir: directory to write into (created if missing).
        extra:    optional dict of extra fields (e.g. {"stage": "1A"}).
        filename: output filename (default run_config.json).
    Returns:
        Path written, or None on failure (failure is logged, not raised).
    """
    try:
        if hasattr(args, "__dict__"):
            args_dict = dict(vars(args))
        elif isinstance(args, dict):
            args_dict = dict(args)
        else:
            args_dict = {"_repr": repr(args)}

        def _safe(v):
            try:
                json.dumps(v)
                return v
            except (TypeError, ValueError):
                return str(v)

        args_dict = {k: _safe(v) for k, v in args_dict.items()}

        snapshot = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(sys.argv),
            "args": args_dict,
            "git": _git_info(),
            "env": _env_info(),
        }
        if extra:
            snapshot["extra"] = {k: _safe(v) for k, v in extra.items()}

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        return path
    except Exception as e:
        print(f"[save_run_config] warning: failed to save config ({e})")
        return None


def load_checkpoint(model, optimizer, path, device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('metrics', {})


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time

    def elapsed_str(self):
        elapsed = self.elapsed()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'