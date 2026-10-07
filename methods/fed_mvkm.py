"""Fed-MVKM 适配器 —— 「普通联邦多视图聚类」（非遥感设计）。

论文：Yang & Sinaga, "Federated Multi-View K-Means Clustering",
      IEEE TPAMI 47(4):2446-2459, 2025。
本地代码：`D:\\cc\\Fed-MVKM\\fmvkm\\`（该项目自己的忠实复现，非官方包）。

⚠️ 联邦设定是【水平联邦】
   原文 p2：*"A horizontal federated learning (HFL) is adopted"*。
   即：**客户端 = 样本子集，每个客户端持有全部视图**。
   因此它只能跑**区域划分（范式 A）**，不能跑视图划分。
   在视图划分下，`FedMVKM.__init__` 会因为「每个客户端必须持有全部视图」
   直接 `raise ValueError` —— 这个失败本身就是结论，要如实记录。

它为什么可能在高光谱上掉点（本平台要验证的假设）
------------------------------------------------
1. **完全没有空间概念**：每个像素被当成独立的特征向量，
   邻居关系、"同块地物连片"这些遥感基本先验一概用不上。
2. **指数距离核对高维不友好**：成员度用 `1-exp(-β‖x-a‖²)`，
   β 按各视图自身的距离尺度估计。高光谱 180 维下距离集中（维度灾难），
   β 估计不稳。该项目自己的笔记也记录了「HD/Caltech-20 的最优 β_k 落在
   β→0 的欧氏极限」——即指数核在高维上退化了。
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

CC = Path(__file__).resolve().parents[2]          # D:\cc
if str(CC / "Fed-MVKM") not in sys.path:
    sys.path.insert(0, str(CC / "Fed-MVKM"))

from common.data import RSDataset                  # noqa: E402
from common.partition import Partition             # noqa: E402


def run_fed_mvkm(
    ds: RSDataset,
    part: Partition,
    views: list[np.ndarray],
    seed: int = 0,
    beta_k: float = 1.0,
    alpha: float = 15.0,
    max_iterations: int = 100,
    **overrides,
):
    """在区域划分上跑 Fed-MVKM。

    Parameters
    ----------
    ds : RSDataset
    part : Partition
        必须是 region / random 模式（水平联邦）。
    views : list[(N, B_v)]
        已归一化的各视图特征。
    beta_k : float
        「scale_free」模式下 β_h = beta_k / mean(距离²)。
        原复现在各数据集上单独调过此值（`scripts/run_protocol.py:DATASET_BETA_K`）；
        遥感数据没有先验最优值，默认 1.0，可扫。

    Returns
    -------
    (labels, detail)
        `labels` : (N,) int32，**已还原到原始像素顺序**（靠 client_indices）
    """
    from fmvkm.core.fed_mvkm import FedMVKM, FedMVKMParams

    if part.mode not in ("region", "random"):
        raise ValueError(
            f"Fed-MVKM 是水平联邦，只接受 region/random 划分；收到 mode={part.mode!r}。"
            f"视图划分下它要求每个客户端持有全部视图，无法运行。"
        )

    t0 = time.time()
    K = ds.n_classes
    V = ds.n_views

    client_data, client_indices = [], []
    for c in part.clients:
        if c.sample_idx is None:
            raise ValueError(f"客户端 {c.client_id} 未指定 sample_idx")
        idx = np.asarray(c.sample_idx, dtype=np.int64)
        client_data.append([X[idx] for X in views])
        client_indices.append(idx)

    params = FedMVKMParams(
        n_clusters=K, n_views=V,
        alpha=alpha, beta_k=beta_k,
        max_iterations=max_iterations,
        random_state=seed,
        **overrides,
    )

    model = FedMVKM(params, client_data, client_indices)
    result = model.fit()

    labels = np.asarray(result.labels).astype(np.int32)
    if labels.size != ds.n_pixels:
        raise ValueError(f"标签数 {labels.size} != 像素数 {ds.n_pixels}")

    detail = {
        "runtime_s": time.time() - t0,
        "n_iterations": int(result.n_iterations),
        "converged": bool(result.converged),
        "view_weights": [np.asarray(v).tolist() for v in result.view_weights],
        "betas": [np.asarray(b).tolist() for b in result.betas],
        "client_sizes": [int(ix.size) for ix in client_indices],
        "beta_k": beta_k, "alpha": alpha,
    }
    return labels, detail


def fed_mvkm_applicable(part: Partition) -> tuple[bool, str]:
    """判断某个划分下 Fed-MVKM 能否运行，返回 (能否, 原因)。"""
    if part.mode in ("region", "random"):
        return True, "水平联邦：客户端持样本子集 + 全部视图"
    return False, (
        "Fed-MVKM 要求每个客户端持有全部视图；视图划分下该前提不成立，"
        "代码会 raise ValueError（如实记录为 N/A）"
    )
