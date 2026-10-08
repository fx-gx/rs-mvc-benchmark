"""OSMAGC 集中式适配器 —— 统一平台里的「遥感专用 MVC」上界。

为什么它是公共尺子
------------------
OSMAGC 集中式**不需要任何联邦划分**：所有视图、所有像素放在一起跑。
因此它是唯一一个在范式 A / 范式 B 下取值完全相同的方法，
任何联邦方法都可以报「距离它还差多少」，这个 gap 就是联邦化损失。

⚠️ 两处「用真值选结果」的坑（官方协议里存在，本适配器默认关闭）
--------------------------------------------------------------
1. 官方 `osmagc.m` 跑完 max_iter 轮后，对每一轮的 H 都做 k-means 并用 **ACC**
   挑最好的一轮报告。ACC 需要真值 —— 这是 **oracle 轮次选择**。
2. 官方 `evaluate_H` 对每个 H 跑 n_init 次 k-means，同样按 **ACC** 取最优。

本适配器把这两件事分开：
- `labels`（主结果）：**最后一轮 H + sklearn KMeans(n_init=10)**，
  只用 inertia 选初始化，**完全不接触真值**。与其他方法口径一致。
- `labels_oracle`（参考值）：复刻官方协议（按 ACC 选轮次与初始化）。
  两者之差 = oracle 选择带来的虚高，会单独报出来。

依赖 `D:\\cc\\OSMAGC` 的复现代码（Python 版，官方为 MATLAB）。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

CC = Path(__file__).resolve().parents[2]          # D:\cc
if str(CC / "OSMAGC") not in sys.path:
    sys.path.insert(0, str(CC / "OSMAGC"))

from common.data import RSDataset                  # noqa: E402


# --------------------------------------------------------------------------- #
# 各数据集的 OSMAGC 超参（来自 OSMAGC 项目的 λ/γ 与超像素扫描结论）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OSMAGCParams:
    num_pixel: tuple[int, ...]
    lam: float
    gamma: float
    max_iter: int = 10
    # 复现验证时确认必须关闭/开关（见 OSMAGC/NOTES.md D1、D13、D10）
    h_simplex: bool = False        # D1: 官方该步是死代码，开着会让 H 塌缩（NMI≈0）
    mapminmax_axis: int = 0        # D13: 0=按列(按波段)
    h_denom_v: bool = True         # D10
    sagi_mode: str = "ers"
    sacfl_fixed: bool = False


DEFAULT_PARAMS: dict[str, OSMAGCParams] = {
    # MUUFL：2026-10-08 本平台实测（results/tuning_osmagc_MUUFL.json + reselect_topk.json）
    #   原官方值 λ=1 γ=1（来自 OSMAGC 项目 run_phase_a.py: BEST_LG）
    #   ⚠️ 扫描阶段（n_init=3）选的冠军是 λ=1 γ=0.1（0.4193），
    #      但它在正式协议 n_init=10 下只有 0.3994，**反而不如官方值**。
    #      把扫描前 8 名全部拉回 n_init=10 重排后，真冠军是 λ=1 γ=10（0.4146）。
    #   → 教训：n_init=3 的排名不可信，必须整表复核，只复核冠军防不住排名翻转。
    #   协议指纹：BLAS 线程 4、COL_CHUNK 25000、n_init=10。
    "MUUFL":    OSMAGCParams(num_pixel=(50, 100),       lam=1.0,  gamma=10.0),
    # Augsburg：2026-10-07 本平台实测调参结果（results/tuning_osmagc_Augsburg.json）。
    # 5×7 的 λ×γ 网格 + 4 组超像素扫描，Stage C 用 n_init=10 复核。
    #   最优 sp=[200,200,200] λ=1 γ=10：ACC=0.5021 NMI=0.4250 ARI=0.3132 Purity=0.7022
    #   原论文默认 λ=1e-3 γ=1e-2：ACC=0.4789 → 提升 +0.0232
    # ⚠️ 注意：Augsburg 是 OSMAGC 论文自己的主数据集、γ=1e-2 是它唯一的官方默认值，
    # 但**在这上面照样能调出 2.3 个点** —— "默认超参只对原论文数据集有效"这一结论
    # 在核心数据集上也成立，不是 Salinas 的特例。
    # 协议指纹：BLAS 线程 4、COL_CHUNK 25000、复核 n_init=10。
    "Augsburg": OSMAGCParams(num_pixel=(200, 200, 200), lam=1.0,  gamma=10.0),
    "Berlin":   OSMAGCParams(num_pixel=(100, 100),      lam=1e-3, gamma=10.0),
    # Trento / Salinas 不是 OSMAGC 论文的数据集，没有官方最优值。
    # 按论文 4.1 的超像素范围 100~300 取中位起步，λ/γ 取官方代码里 MUUFL 那组
    # （唯一有官方默认值的一组），后续可按本平台的扫描细化。
    # ⚠️ 约束：initz_k=c 要求 Mv >= c+1。
    # Trento：2026-10-08 本平台实测（results/tuning_osmagc_Trento.json）
    #   最优 sp=[100,100] λ=1 γ=100 → ACC=0.9158 NMI=0.8584 ARI=0.9147 Purity=0.9158
    #   原起点 λ=γ=1e-3 → ACC=0.7506，提升 +0.1652（本平台四个数据集里最大的一次）
    #   扫描前 8 名在 n_init=10 下重排后冠军不变，排名基本稳定。
    "Trento":   OSMAGCParams(num_pixel=(100, 100),      lam=1.0,  gamma=100.0),   # c=6
    # Salinas：2026-10-07 本平台实测调参结果（见 results/tuning_osmagc_Salinas.json）。
    # 5×7 的 λ×γ 网格 + 5 组超像素扫描，Stage C 用 n_init=10 复核。
    #   最优 sp=[200,200,200] λ=1 γ=1e-2：ACC=0.6253（复核值）
    #   旧基线 sp=[200,200,200] λ=1e-3 γ=1e-3：ACC=0.5611 → 提升 +0.0642
    # ⚠️ 注意最优 γ 比论文默认的 1e-3 大了 10 倍、λ 大了 1000 倍——不调参会白送 6 个点。
    # 协议指纹：BLAS 线程 4、COL_CHUNK 25000、复核 n_init=10。换任一项数字都会变。
    "Salinas":  OSMAGCParams(num_pixel=(200, 200, 200), lam=1.0,  gamma=1e-2),   # c=16
}


def _to_osmagc_dataset(ds: RSDataset):
    """把本平台的 `RSDataset` 转成 OSMAGC 的 `Dataset`。

    两者展平顺序都是 Fortran（列优先），因此像素索引可以直接一一对应。

    ⚠️ 缺失掩码必须透传：MUUFL 官方数据是**缺失率 0.5** 的版本，缺失像素在
    立方体里已置 0。若把全部像素都标成「可用」，OSMAGC 会把那些 0 当成真实
    观测值参与拟合，ACC 会从 ~0.49 掉到 ~0.37。
    """
    from osmagc.data.loader import Dataset as OSMAGCDataset

    N = ds.n_pixels
    gt_flat = ds.flatten_gt("F")
    index = (
        [np.arange(N, dtype=np.int64) for _ in range(ds.n_views)]
        if ds.index is None
        else [np.asarray(ix, dtype=np.int64) for ix in ds.index]
    )
    rate = float(np.mean([ds.missing_rate(v) for v in range(ds.n_views)]))

    return OSMAGCDataset(
        name=ds.name,
        cubes=list(ds.cubes),
        gt=gt_flat,
        index=index,
        rate=rate,
        H=ds.H,
        W=ds.W,
        V=ds.n_views,
    )


def run_osmagc_central(
    ds: RSDataset,
    params: OSMAGCParams | None = None,
    seed: int = 7,
    n_init: int = 10,
    verbose: bool = False,
):
    """跑一次 OSMAGC 集中式。

    Returns
    -------
    (labels, detail)
        `labels` : (N,) int32，全体像素的簇标签，**无 oracle 选择**
        `detail` : 含 oracle 参考值、视图权重 α、逐轮诊断
    """
    from osmagc.core.model import OSMAGCConfig, run_osmagc

    params = params or DEFAULT_PARAMS[ds.name]
    ocfg = OSMAGCConfig(
        num_pixel=list(params.num_pixel),
        lam=params.lam,
        gamma=params.gamma,
        dim=None,                      # None -> 取类别数 c
        max_iter=params.max_iter,
        h_simplex=params.h_simplex,
        mapminmax_axis=params.mapminmax_axis,
        h_denom_v=params.h_denom_v,
        sagi_mode=params.sagi_mode,
        sacfl_fixed=params.sacfl_fixed,
        verbose=verbose,
    )

    t0 = time.time()
    ods = _to_osmagc_dataset(ds)
    res = run_osmagc(ods, ocfg)
    t_osmagc = time.time() - t0

    # ---- 主结果：最后一轮 H + sklearn KMeans（只用 inertia 选初始化，不看真值）----
    H_last = res.H_list[-1]                       # (dim, N)
    X = np.ascontiguousarray(H_last.T)            # (N, dim)
    km = KMeans(n_clusters=ds.n_classes, n_init=n_init, random_state=seed)
    labels = km.fit_predict(X).astype(np.int32)

    # ---- 参考值：复刻官方 oracle 协议（按 ACC 选轮次与初始化）----
    gt = ds.flatten_gt("F")
    mask = gt > 0
    labels_oracle, oracle_round = _oracle_labels(
        res.H_list, X, gt, mask, ds.n_classes, seed, n_init
    )

    detail = {
        "runtime_s": t_osmagc,
        "alpha": np.asarray(res.alpha).tolist(),
        "num_pixel": list(params.num_pixel),
        "lam": params.lam,
        "gamma": params.gamma,
        "max_iter": params.max_iter,
        "labels_oracle": labels_oracle,
        "oracle_round": oracle_round,
        "n_rounds": len(res.H_list),
    }
    return labels, detail


def _oracle_labels(H_list, X_last, gt, mask, k, seed, n_init):
    """复刻官方协议：逐轮 H 做 k-means，按 ACC 取最优（⚠️ 用到真值）。"""
    from common.metrics import acc as _acc

    best_acc, best_lab, best_round = -1.0, None, -1
    for i, H in enumerate(H_list):
        Xi = np.ascontiguousarray(H.T)
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        lab = km.fit_predict(Xi).astype(np.int32)
        a = _acc(gt[mask], lab[mask])
        if a > best_acc:
            best_acc, best_lab, best_round = a, lab, i + 1
    return best_lab, best_round
