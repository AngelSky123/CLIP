"""
domain_balanced_sampler.py — 让每个 batch 都含「同动作、跨环境」样本对。

== 为什么需要 ==
domain_invariant_loss 的正样本对 = 同 action 但不同 env 的窗口。
随机采样下, batch_size=8 时一个 batch 凑齐「同动作跨环境对」的概率很低
(27 动作 × 3 环境), 导致该项常返回 0 —— 三个 DCL 组件里唯一直接针对
cross-env 瓶颈的那个就失效了。

== 做法 ==
每个 batch 这样构造: 随机选 (batch_size // group_size) 个 action,
每个选中的 action 从【不同环境】各取 group_size 个窗口。
group_size=2 时, 每个 action 贡献 2 个跨环境窗口 -> 正样本对天然存在。
batch_size=8 -> 4 个 action × 2 个跨环境窗口。

兼容性: 只依赖 dataset.samples[i] 里的 'env' 和 'action' 字段
(你的 MMFiDataset 建索引时存了这两个)。不改 dataset、不改模型。
"""
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler):
    """每个 batch 含若干 action, 每个 action 取来自不同 env 的 group_size 个窗口。

    Args:
        dataset: 需有 .samples, 每项是 dict 含 'env' 和 'action'。
        batch_size: 必须能被 group_size 整除。
        group_size: 每个 action 在一个 batch 里贡献几个 (跨环境) 窗口。默认 2。
        seed, drop_last 同常规。
    """
    def __init__(self, dataset, batch_size, group_size=2, seed=42, drop_last=True):
        if batch_size % group_size != 0:
            raise ValueError(f"batch_size({batch_size}) 必须能被 group_size({group_size}) 整除")
        self.batch_size = batch_size
        self.group_size = group_size
        self.n_actions_per_batch = batch_size // group_size
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed

        # 建立 action -> {env -> [sample indices]} 索引
        # action/env 同时记录, 便于「同 action 取不同 env」
        self.act_env_idx = defaultdict(lambda: defaultdict(list))
        for i in range(len(dataset)):
            s = dataset.samples[i]
            self.act_env_idx[s['action']][s['env']].append(i)
        # 只保留「至少跨 2 个环境」的 action (否则无法构造跨环境对)
        self.valid_actions = [
            a for a, envs in self.act_env_idx.items() if len(envs) >= 2
        ]
        if not self.valid_actions:
            raise RuntimeError("没有任何 action 跨>=2个环境, 无法做域平衡采样")

        self.num_samples = len(dataset)
        self._num_batches = self.num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        for _ in range(self._num_batches):
            batch = []
            # 随机选 n_actions_per_batch 个跨环境 action (可重复抽, 数据量足够)
            chosen_actions = rng.choice(
                self.valid_actions,
                size=self.n_actions_per_batch,
                replace=len(self.valid_actions) < self.n_actions_per_batch,
            )
            for act in chosen_actions:
                envs = list(self.act_env_idx[act].keys())
                # 从该 action 的不同环境里挑 group_size 个 (尽量来自不同 env)
                if len(envs) >= self.group_size:
                    pick_envs = rng.choice(envs, size=self.group_size, replace=False)
                else:
                    pick_envs = rng.choice(envs, size=self.group_size, replace=True)
                for e in pick_envs:
                    pool = self.act_env_idx[act][e]
                    batch.append(int(rng.choice(pool)))
            rng.shuffle(batch)
            yield batch


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 用 stub dataset 验证: 每个 batch 是否真的含同动作跨环境对
    class StubDS:
        def __init__(self):
            self.samples = []
            envs = ['E01', 'E02', 'E03']
            for a in range(1, 28):
                for e in envs:
                    for _ in range(10):
                        self.samples.append({'action': f'A{a:02d}', 'env': e})
        def __len__(self): return len(self.samples)

    ds = StubDS()
    sampler = DomainBalancedBatchSampler(ds, batch_size=8, group_size=2, seed=0)
    print(f"dataset size={len(ds)}  num_batches={len(sampler)}")
    n_check, n_has_cross = 0, 0
    for batch in sampler:
        assert len(batch) == 8
        # 检查这个 batch 有没有「同动作跨环境」对
        by_act = defaultdict(set)
        for i in batch:
            s = ds.samples[i]
            by_act[s['action']].add(s['env'])
        has_cross = any(len(envs) >= 2 for envs in by_act.values())
        n_has_cross += int(has_cross)
        n_check += 1
        if n_check >= 100:
            break
    print(f"前 {n_check} 个 batch 中, 含同动作跨环境对的比例: {n_has_cross}/{n_check}")
    assert n_has_cross == n_check, "应该每个 batch 都有跨环境对"
    print("[OK] 每个 batch 都保证含同动作跨环境对")


# ======================================================================
# 接入 train_mae.py (改 DataLoader 构建)
# ======================================================================
#
# 原代码:
#     train_loader = DataLoader(
#         train_set, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.num_workers, pin_memory=True, drop_last=True,
#     )
#
# 改为:
#     from domain_balanced_sampler import DomainBalancedBatchSampler
#     batch_sampler = DomainBalancedBatchSampler(
#         train_set, batch_size=args.batch_size, group_size=2, seed=args.seed)
#     train_loader = DataLoader(
#         train_set, batch_sampler=batch_sampler,
#         num_workers=args.num_workers, pin_memory=True,
#     )
#     # 注意: 用 batch_sampler 时不能再传 batch_size/shuffle/drop_last
#
# 并在每个 epoch 开头 (for epoch in ...: 之后) 加:
#     batch_sampler.set_epoch(epoch)   # 保证每个 epoch 采样不同
#
# 验证生效: 接入后烟雾测试时, 总 loss 应比纯 MAE 明显高 (domain_invariant
# 不再是 0)。若想确认, 可临时在 train_mae.py 里把三个 DCL 项分别打印出来。