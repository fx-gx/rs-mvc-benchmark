"""PMIMC 适配器 —— 通用不完整多视图聚类（TIP 2025），**非遥感设计**。

PMIMC = Prototype Matching Learning for Incomplete Multi-view Clustering。
它是本平台「通用算法」那一侧的第三个方法。

⚠️ 为什么必须自己重跑读出，而不能用它的 ACC
------------------------------------------------
`main.py` 末尾：
    estimator = KMeans(n_clusters=args.K); estimator.fit(fea_cluster)
    pred_final = estimator.labels_
    acc, ... = evaluate(Labels, pred_final)      # Nmetrics.evaluate

`Nmetrics.evaluate` 内部同样用**混淆矩阵做最优匹配**算 ACC——平台明令禁止的
「用真值挑结果」。所以本适配器只取 `pred_final`，换成平台 `common/metrics.py`。

⚠️ 三处必须处理的地方
------------------------------------------------
1. **`main.py` 是模块级 `args = parser.parse_args()`**，导入时就会解析命令行。
   且它在导入期强制查 `ALL_data[i_d[args.i_d]]`，我们的数据集不在表里。
   做法：先传一个合法 `--i_d` 让导入通过，再直接改写 `main.args` 的字段。
2. **`loader.load_data` 是硬编码 h5py 路径的**，且内部 `Form_Incomplete_Data`
   会 **`np.random.shuffle` 打乱样本顺序**（且不返回该顺序），标签就对不上了。
   做法：monkeypatch 掉 `load_data`，用平台数据**不打乱**地构造它期望的七个返回值。
3. **`main()` 不返回标签**。做法：给 `Nmetrics.evaluate` 装探针拦下最后一次调用，
   拿到 `pred_final`（不改它一行代码）。

数据顺序：用 `flatten_view(v,"F")`，索引与平台 F 序逐位对应，故标签无需置换。

依赖：`D:\\cc\\OSMAGC\\baselines\\PMIMC`（原论文代码，未改动）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PMIMC_DIR = Path(__file__).resolve().parents[2] / "OSMAGC" / "baselines" / "PMIMC"


@dataclass(frozen=True)
class PMIMCParams:
    """默认值取自 PMIMC 官方 argparse 默认。"""

    pretrain_epochs: int = 200
    proto_epochs: int = 100
    batch_size: int = 256
    batch_rob: int = 256
    # ⚠️ final_batch 官方默认 256，但那个值是为 N≈2000~4500 的数据集定的。
    # `get_Similarity` 会开一个 (原型数, batch, feature_dim) 的三维中间量，
    # 而原型数 ≈ protorate×N。我们的 N 是 7 万~16 万（大 20~40 倍），
    # 256 时该中间量达 7.2 GiB 直接 CUDA OOM。降到 64 后方可运行。
    final_batch: int = 64
    lr_pre: float = 5e-4
    lr_align: float = 1e-4
    feature_dim: int = 256
    protorate: float = 0.3


DEFAULT_PARAMS: dict[str, PMIMCParams] = {
    "Augsburg": PMIMCParams(),
    "MUUFL":    PMIMCParams(),
    "Trento":   PMIMCParams(),
    "Salinas":  PMIMCParams(),
}


def pmimc_applicable(ds) -> tuple[bool, str]:
    """PMIMC 在这份数据上能不能跑。"""
    if ds.n_views < 2:
        return False, f"PMIMC 需要至少 2 个视图，{ds.name} 只有 {ds.n_views} 个"
    return True, ""


def _build_loader(ds):
    """复刻 `Form_Incomplete_Data` 的处理，但**不做样本随机打乱**。

    原函数会 `np.random.shuffle(index)` 把 X/Y 重排却不返回该顺序，
    导致输出标签无法对回平台像素序。这里保持原序。
    """
    import torch

    v_num = ds.n_views
    n = ds.n_pixels
    x_list = [np.asarray(ds.flatten_view(v, "F"), dtype=np.float32) for v in range(v_num)]
    labels = ds.flatten_gt("F").astype(np.int64)
    y_list = [labels.copy() for _ in range(v_num)]

    # 缺失掩码：用平台记录的真实可用性（MUUFL 有 25%×25% 像素级缺失）；其余全 1
    if ds.index is not None:
        miss = np.zeros((n, v_num), dtype=np.float64)
        for v in range(v_num):
            miss[np.asarray(ds.index[v], dtype=np.int64), v] = 1.0
    else:
        miss = np.ones((n, v_num), dtype=np.float64)

    # 逐视图拆出「完整」与「不完整」样本下标（原逻辑）
    idx_complete = [[] for _ in range(v_num)]
    idx_partial = [[] for _ in range(v_num)]
    for i in range(n):
        for v in range(v_num):
            (idx_complete[v] if miss[i, v] == 1 else idx_partial[v]).append(i)

    # 把各视图的完整样本补齐到同一长度（原逻辑：有放回地重采样差值部分）
    rng = np.random.RandomState(0)
    max_len = max(len(c) for c in idx_complete) if idx_complete else 0
    filled = []
    for v in range(v_num):
        cur = idx_complete[v]
        if len(cur) < max_len:
            extra = rng.choice(cur, size=max_len - len(cur), replace=True).tolist()
            filled.append(cur + extra)
        else:
            filled.append(cur)

    x_com = [torch.from_numpy(x_list[v][filled[v]]) for v in range(v_num)]
    y_com = [torch.from_numpy(y_list[v][filled[v]]) for v in range(v_num)]
    x_t = [torch.from_numpy(x_list[v]) for v in range(v_num)]
    y_t = [torch.from_numpy(y_list[v]) for v in range(v_num)]

    return x_t, y_t, miss, x_com, y_com, idx_complete, idx_partial


def run_pmimc(ds, params: PMIMCParams | None = None, seed: int = 0, verbose: bool = False):
    """在 `RSDataset` 上跑一次 PMIMC。返回平台 F 序的簇标签。"""
    if str(PMIMC_DIR) not in sys.path:
        sys.path.insert(0, str(PMIMC_DIR))

    p = params or DEFAULT_PARAMS.get(ds.name, PMIMCParams())

    # --- 1. 导入：给它一个合法的 --i_d 让模块级 parse_args 通过 ---
    import main as pmimc_main                       # noqa: E402

    # --- 2. 改写 args（模块级对象，直接改字段） ---
    a = pmimc_main.args
    a.V = ds.n_views
    a.K = int(ds.n_classes)
    a.N = int(ds.n_pixels)
    a.view_dims = [int(c.shape[2]) for c in ds.cubes]
    a.dataset = ds.name
    a.missrate = 0.0                                 # 掩码由我们注入，不让它随机生成
    a.pretrain_epochs = p.pretrain_epochs
    a.ProtoRobs_epochs = p.proto_epochs
    a.batch_size = p.batch_size
    a.Batch_Rob = p.batch_rob
    a.final_batch = p.final_batch
    a.lr_pre = p.lr_pre
    a.lr_align = p.lr_align
    a.feature_dim = p.feature_dim
    a.protorate = p.protorate

    # --- 3. monkeypatch load_data（保持原序，用平台数据） ---
    pmimc_main.loader.load_data = lambda dataset, missrate: _build_loader(ds)

    # --- 4. 给 evaluate 装探针，拦下最后一次 pred_final（不改它代码） ---
    original_eval = pmimc_main.evaluate
    captured: dict[str, np.ndarray] = {}

    def spy(labels, pred):
        pred = np.asarray(pred).reshape(-1)
        if pred.size == ds.n_pixels:
            captured["labels"] = pred.copy()
        return original_eval(labels, pred)

    pmimc_main.evaluate = spy

    # --- 5. 固定随机种子并跑 ---
    pmimc_main.seed_everything(seed)
    try:
        pmimc_main.main()
    finally:
        pmimc_main.evaluate = original_eval

    if "labels" not in captured:
        raise RuntimeError("未能从 PMIMC 捕获预测标签（evaluate 未被以 N 长度调用）")

    labels = captured["labels"].astype(np.int32)
    detail = {
        "pretrain_epochs": p.pretrain_epochs,
        "proto_epochs": p.proto_epochs,
        "feature_dim": p.feature_dim,
        "view_dims": a.view_dims,
    }
    return labels, detail
