"""OSMAGC 的联邦化版本（视图划分：一个客户端 = 一个模态）。

为什么 OSMAGC 能【无损】联邦化
-----------------------------
看集中式主循环 `osmagc/core/model.py::opti` 的五步，只有 updateH 用到全部视图，
而且**只通过一个求和项**：

    H  <- solve( λ · Σ_v (R_v @ Z_v) )        # updateH 里 part1 = Σ_v R_v Z_v
    α_v <- e_v / Σ_v e_v                      # updateBeta，e_v 是标量残差

其余三步全是**逐视图独立**的：
    A_v = updateA(X_v, Z_v, q_v)
    Z_v = updateZv(A_v, X_v, R_v, H, α_v, q_v)
    R_v = updateRv(Z_v, H)                    # 只需 H 和本视图的 Z_v

于是可以这样切分（**数学上与集中式完全等价，无损**）：

    客户端 v（持有 X_v、q_v，原始数据不出门）
        A_v  <- updateA(X_v, Z_v, q_v)                    本地
        R_v  <- updateRv(Z_v, H)                          本地（H 来自服务器）
        Z_v  <- updateZv(A_v, X_v, R_v, H, α_v, q_v)      本地
        上传 S_v = R_v @ Z_v  （dim × N）和标量 e_v
    服务器（持有 H、α，以及**公开的**栅格几何 D——所有客户端共享同一幅图）
        H   <- solve( λ · Σ_v S_v )
        α   <- normalize([e_v])
        下发 H（dim × N）和 α_v

因此 **联邦化 gap 恒等于 0**；真实代价在**通信量**上：每客户端每轮
上传 + 下发各一个 (dim × N) 矩阵。本模块把这个代价量化出来。

⚠️ 这条路之所以走得通，依赖于「所有客户端共享同一幅栅格」——即视图划分。
   若改成区域划分，各客户端只有子栅格，`construct_D` 需按子栅格构造，
   块边界会切断空间平滑，那时才会有真正的精度损失。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CC = Path(__file__).resolve().parents[2]          # D:\cc
if str(CC / "OSMAGC") not in sys.path:
    sys.path.insert(0, str(CC / "OSMAGC"))

from common.data import RSDataset                  # noqa: E402
from methods.osmagc_central import OSMAGCParams, DEFAULT_PARAMS   # noqa: E402


# --------------------------------------------------------------------------- #
@dataclass
class CommStats:
    """通信量统计（float64，1 元素 = 8 字节）。"""

    rounds: int = 0
    dim: int = 0
    N: int = 0
    n_clients: int = 0
    up_per_client_per_round: int = 0      # bytes
    down_per_client_per_round: int = 0    # bytes
    _up: int = field(default=0, repr=False)
    _down: int = field(default=0, repr=False)

    def add_round(self, up: int, down: int) -> None:
        self._up += up
        self._down += down

    @property
    def total_up(self) -> int:
        return self._up

    @property
    def total_down(self) -> int:
        return self._down

    @property
    def total(self) -> int:
        return self._up + self._down

    def human(self, n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
            n /= 1024.0
        return f"{n:.1f} GB"


# --------------------------------------------------------------------------- #
def _client_setup(ds: RSDataset, params: OSMAGCParams, v: int):
    """客户端 v 的本地准备：只碰自己的那个视图。"""
    from osmagc.core.init import initA, initZ
    from osmagc.core.predata import predata
    from osmagc.data.loader import findindex

    ods_data = [ds.cubes[v].reshape((-1, ds.cubes[v].shape[2]), order="F")
                .T.astype(np.float64)]
    index_v = ([np.arange(ds.n_pixels, dtype=np.int64)] if ds.index is None
               else [np.asarray(ds.index[v], dtype=np.int64)])

    X_v, Q_v = findindex(ods_data, index_v)         # [(Bv,N)], (N,1)
    q_v = Q_v[:, 0].astype(np.float64)

    pre_v = predata([ds.cubes[v]], [int(params.num_pixel[v])],
                    pca_sign=1, mapminmax_axis=params.mapminmax_axis,
                    sagi_mode=params.sagi_mode)
    A_v = initA(X_v, pre_v.sp_labels, [int(params.num_pixel[v])])[0]

    return {
        "X": X_v[0],                                 # (Bv, N) L2 归一化
        "q": q_v,                                    # (N,)
        "A": A_v,                                    # (Bv, Mv)
        "sp_labels": pre_v.sp_labels[0],
        "unm": int(q_v.sum()),
    }


def run_osmagc_federated(
    ds: RSDataset,
    params: OSMAGCParams | None = None,
    seed: int = 7,
    n_init: int = 10,
    verbose: bool = False,
):
    """跑 OSMAGC 的联邦化版本（视图划分）。

    Returns
    -------
    (labels, detail)
        `labels`：与集中式同口径的诚实读出（最后一轮 H + sklearn KMeans，按 inertia 选初始化）
        `detail`：通信量统计、H 序列、α，以及可选的与集中式的一致性校验值
    """
    from osmagc.core.init import initZ, initH
    from osmagc.core.operators import construct_D
    from osmagc.core.updates import (
        HUpdater, updateA, updateRv, updateZv, updateBeta,
    )
    from sklearn.cluster import KMeans

    params = params or DEFAULT_PARAMS[ds.name]
    V, N = ds.n_views, ds.n_pixels
    dim = ds.n_classes
    lam, gamma = params.lam, params.gamma

    t0 = time.time()

    # ---------- 客户端本地准备（各自只碰自己的视图）----------
    clients = [_client_setup(ds, params, v) for v in range(V)]
    if verbose:
        print(f"    [客户端] {V} 个客户端各自完成超像素+锚点初始化")

    # 初始锚图：initZ 需要各视图的可用像素，逐客户端算
    idx_list = ([np.arange(N, dtype=np.int64)] * V if ds.index is None
                else [np.asarray(ds.index[v], dtype=np.int64) for v in range(V)])
    unm = np.array([c["unm"] for c in clients])
    Z = initZ([c["X"][:, c["q"] == 1] for c in clients], [c["A"] for c in clients],
              dim, idx_list, unm, N)

    # ---------- 服务器：公开的栅格几何 + 共享状态 ----------
    D = construct_D(ds.H, ds.W)                      # 公开：所有客户端同图
    solver = HUpdater(D, lam, gamma, V, use_v_factor=params.h_denom_v)
    H = initH(N, dim)                                # 确定性初始化
    alpha = np.ones(V) / V

    comm = CommStats(dim=dim, N=N, n_clients=V)

    if verbose:
        print(f"    [服务器] 栅格几何 D: {D.shape}, nnz={D.nnz}（公开信息）")

    # ---------- 联邦主循环 ----------
    H_list = []
    bytes_mat = 8 * dim * N                          # 一个 (dim,N) float64 矩阵

    for it in range(1, params.max_iter + 1):
        S_sum = np.zeros((dim, N), dtype=np.float64)
        e = np.zeros(V, dtype=np.float64)

        # --- 各客户端本地迭代（互相不通信）---
        for v in range(V):
            Xv, qv, Av, Zv = clients[v]["X"], clients[v]["q"], clients[v]["A"], Z[v]

            Av = updateA([Xv], [Zv], [qv])[0]
            Rv = updateRv(1, [Zv], H)[0]
            Zv = updateZv([Av], [Xv], N, [Rv], H, lam, np.array([alpha[v]]), [qv])[0]

            clients[v]["A"] = Av
            Z[v] = Zv

            S_v = Rv @ Zv                            # (dim, N)
            avail = qv == 1
            e[v] = float(np.linalg.norm(
                Xv[:, avail] - Av @ Zv[:, avail], "fro") ** 2)

            S_sum += S_v
            # 上传：S_v (dim×N) + 标量 e_v
            comm.add_round(bytes_mat + 8, bytes_mat + 8)   # 上传 / 下发成对计入

        # --- 服务器：只做聚合与全局更新 ---
        H = solver.solve(lam * S_sum)
        if params.h_simplex:
            from osmagc.core.updates import eproj_simplex_batch
            H = eproj_simplex_batch(H, 1.0)
        total = e.sum()
        alpha = e / total if total > 0 else np.full(V, 1.0 / V)

        H_list.append(H.copy())
        comm.rounds = it

        if verbose:
            print(f"    iter {it:2d}/{params.max_iter}  "
                  f"α=[{', '.join(f'{a:.3f}' for a in alpha)}]  "
                  f"累计通信 {comm.human(comm.total)}")

    comm.up_per_client_per_round = bytes_mat + 8
    comm.down_per_client_per_round = bytes_mat + 8

    # ---------- 诚实读出（与其他方法同一协议）----------
    X = np.ascontiguousarray(H_list[-1].T)
    km = KMeans(n_clusters=dim, n_init=n_init, random_state=seed)
    labels = km.fit_predict(X).astype(np.int32)

    detail = {
        "runtime_s": time.time() - t0,
        "alpha": alpha.tolist(),
        "H_list": H_list,
        "num_pixel": list(params.num_pixel),
        "lam": lam,
        "gamma": gamma,
        "max_iter": params.max_iter,
        "comm": comm,
        "dim": dim,
    }
    return labels, detail
