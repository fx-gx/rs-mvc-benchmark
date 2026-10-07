"""Local-only 下界 —— 各客户端独立聚类，不做任何联邦协作。

对应你截图里的第三行「Local-only（无协作）」。

两种范式的语义不同，融合方式也不同：

**范式 B（视图划分）**
    每个视图独立 k-means → 得到 V 个标签向量（同一批 N 个像素）。
    融合需要先解决「不同视图的簇编号不对应」的问题，本模块用
    **匈牙利对齐到锚视图 + 多数投票**（等价于证据累积，只依赖列联表，O(N·V)）。
    不对齐就没法投票 —— 直接投票等于随机，这是最容易踩的坑。

**范式 A（区域划分）**
    每块区域独立 k-means → 各区域标签空间独立，必须先对齐才能拼接。
    本模块用「**共享初始质心**」方案：所有客户端用同一组全局初始质心
    （在标准化后的共享特征空间里按同一个种子生成），因此本地第 k 个簇
    天然对应同一语义，拼接无需事后对齐，也不接触任何真值。
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from common.metrics import evaluate


# --------------------------------------------------------------------------- #
def _kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(X).astype(np.int32)


def _align(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把 `src` 的簇编号用匈牙利算法对齐到 `ref` 的编号（不使用真值）。"""
    ks, kr = int(src.max()) + 1, int(ref.max()) + 1
    n = max(ks, kr)
    C = np.zeros((n, n), dtype=np.int64)
    np.add.at(C, (src, ref), 1)          # C[i, j] = #{src==i 且 ref==j}
    row, col = linear_sum_assignment(-C)
    mapping = np.arange(ks, dtype=np.int32)
    valid = row < ks
    mapping[row[valid]] = col[valid]
    return mapping[src]


def _majority_vote(votes: np.ndarray) -> np.ndarray:
    """票矩阵 (V, N) -> 逐像素取众数。并列时取编号最小的簇。"""
    kmax = int(votes.max()) + 1
    score = np.zeros((kmax, votes.shape[1]), dtype=np.int32)
    for k in range(kmax):
        score[k] = (votes == k).sum(axis=0)
    return np.argmax(score, axis=0).astype(np.int32)


# --------------------------------------------------------------------------- #
def run_view_local_only(
    views: list[np.ndarray],
    n_clusters: int,
    seed: int = 0,
    anchor: int | str = "auto",
    return_detail: bool = False,
):
    """范式 B 的 Local-only：逐视图独立聚类 + 对齐 + 多数投票。

    Parameters
    ----------
    anchor : int | 'auto'
        以哪个视图为对齐锚。'auto'（默认）在所有视图里挑**对齐后投票一致率最高**
        的那个 —— 只用簇结构的一致性来选，不接触任何真值。
        固定取 view 0 是常见做法，但若 view 0 恰好是最弱的模态（Augsburg 的 HS
        就是），误差会通过对齐传染给所有其他视图，融合结果可能**低于最强单视图**。

    Returns
    -------
    (final_labels, detail) 或 final_labels
    """
    t0 = time.time()
    per_view = [_kmeans(X, n_clusters, seed) for X in views]
    V = len(per_view)

    # ---- 选锚：对每个候选锚算一遍全对齐后的一致率，取最高者 ----
    if anchor == "auto":
        best, best_score = 0, -1.0
        anchor_scores = {}
        for a in range(V):
            votes_a = np.stack([_align(lv, per_view[a]) for lv in per_view], axis=0)
            fa = _majority_vote(votes_a)
            sc = float((votes_a == fa[None, :]).mean())
            anchor_scores[a] = sc
            if sc > best_score:
                best, best_score = a, sc
        anchor_idx = best
    else:
        anchor_idx = int(anchor)
        anchor_scores = {}

    votes = np.stack([_align(lv, per_view[anchor_idx]) for lv in per_view], axis=0)
    final = _majority_vote(votes)

    if not return_detail:
        return final

    agree = float((votes == final[None, :]).mean())
    detail = {
        "per_view_labels": per_view,
        "aligned_views": votes,
        "vote_agreement": agree,
        "anchor_idx": anchor_idx,
        "anchor_name": None,                 # 由调用方补上视图名
        "anchor_scores": anchor_scores,
        "runtime_s": time.time() - t0,
    }
    return final, detail


def run_region_local_only(
    views_full: list[np.ndarray],
    client_idx: list[np.ndarray],
    n_clusters: int,
    seed: int = 0,
    return_detail: bool = False,
):
    """范式 A 的 Local-only：共享初始质心 + 各区域独立聚类 + 拼接。

    Parameters
    ----------
    views_full : list[(N, B_v)]
        全体像素的视图特征（已归一化）。
    client_idx : list[(n_m,)]
        各客户端持有的像素全局索引。

    Notes
    -----
    共享初始质心的做法：先把全局数据抽稀（每客户端等量抽样拼成一个小样本集），
    在拼起来的特征上跑一次 k-means++ 得到中心，再分发给各客户端作为初始中心。
    这些中心**不含标签信息**，只是共享的随机初始化，因此不构成信息泄漏。
    """
    t0 = time.time()
    N = views_full[0].shape[0]
    rng = np.random.default_rng(seed)

    # 1) 共享初始质心：在每个客户端的样本里等量抽稀
    per_client_cap = 2000
    parts = []
    for idx in client_idx:
        take = min(per_client_cap, idx.size)
        parts.append(rng.choice(idx, size=take, replace=False))
    pool = np.concatenate(parts)

    Z = np.concatenate([X[pool] for X in views_full], axis=1).astype(np.float32)
    init = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit(Z).cluster_centers_

    # 2) 各客户端在**自己的视图拼接特征**上独立跑 k-means，用共享中心初始化
    final = np.empty(N, dtype=np.int32)
    for idx in client_idx:
        Zc = np.concatenate([X[idx] for X in views_full], axis=1).astype(np.float32)
        km = KMeans(
            n_clusters=n_clusters, init=init.copy(), n_init=1, random_state=seed
        )
        final[idx] = km.fit_predict(Zc).astype(np.int32)

    if not return_detail:
        return final
    return final, {"runtime_s": time.time() - t0, "n_pool": int(pool.size)}


# --------------------------------------------------------------------------- #
def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return evaluate(y_true, y_pred)
