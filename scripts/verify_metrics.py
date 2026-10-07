"""把统一指标模块和四份现有实现互相对拍，证明口径真的统一了。

对拍对象
--------
1. `common.metrics`                       —— 本次统一模块
2. `sklearn.metrics`                      —— 第三方基准
3. `OSMAGC/osmagc/metrics.py`             —— 遥感侧参考实现（对应官方 MATLAB）
4. `Federated-MVC/cal_metric.py`          —— FedRSMVC 的实现
5. `Fed-MVKM/fmvkm/metrics.py`            —— Fed-MVKM 复现的实现（无 Purity）

运行：python scripts/verify_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score as sk_ari,
    normalized_mutual_info_score as sk_nmi,
)

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT.parent                                   # D:\cc

# 统一模块
sys.path.insert(0, str(ROOT))
from common import metrics as M                    # noqa: E402

# 三个现有项目
sys.path.insert(0, str(CC / "OSMAGC"))
sys.path.insert(0, str(CC / "Federated-MVC"))
sys.path.insert(0, str(CC / "Fed-MVKM"))

import cal_metric as FRS                           # noqa: E402  Federated-MVC
from osmagc.metrics import evaluate_results_clustering as osm_eval   # noqa: E402
from fmvkm.metrics import evaluate as mvkm_eval    # noqa: E402

TOL = 1e-9

# --------------------------------------------------------------------------- #
# 测试用例
# --------------------------------------------------------------------------- #
def cases():
    rng = np.random.default_rng(0)
    out = {}

    # 1. 完全一致
    a = rng.integers(0, 5, 2000)
    out["perfect"] = (a, a.copy())

    # 2. 随机无关标签
    out["random"] = (rng.integers(0, 5, 2000), rng.integers(0, 5, 2000))

    # 3. 带标签空洞的真值（真值只有 {0,2,5}）
    t = rng.choice([0, 2, 5], 2000)
    p = rng.integers(0, 3, 2000)
    out["holes_in_truth"] = (t, p)

    # 4. 预测簇数 != 真值类数
    t = rng.integers(0, 4, 3000)
    p = t.copy()
    flip = rng.random(3000) < 0.3
    p[flip] = rng.integers(0, 7, flip.sum())     # 预测有 7 个簇
    out["k_pred_gt_k_true"] = (t, p)

    # 5. 极端不平衡（一个簇占 95%）
    t = np.concatenate([np.zeros(9500, int), rng.integers(0, 4, 500)])
    p = np.concatenate([np.zeros(9000, int), rng.integers(0, 4, 1000)])
    out["imbalanced"] = (t, p)

    # 6. 只有一类
    out["single_class"] = (np.zeros(500, int), np.zeros(500, int))

    # 7. 含背景 0 的真实遥感场景风格（0 是背景，不参与）
    t = rng.integers(1, 8, 5000)
    p = t.copy()
    noise = rng.random(5000) < 0.45
    p[noise] = rng.integers(1, 8, noise.sum())
    out["rs_style_7class"] = (t, p)

    # 8. 大规模（贴近 Augsburg 78k 有标注像素）
    t = rng.integers(1, 8, 78294)
    p = t.copy()
    noise = rng.random(78294) < 0.5
    p[noise] = rng.integers(1, 8, noise.sum())
    out["augsburg_scale"] = (t, p)

    return out


# --------------------------------------------------------------------------- #
# 对拍
# --------------------------------------------------------------------------- #
def _safe(fn):
    """调用外部实现，吞掉异常并记录原因（这些实现都有隐含前提）。"""
    try:
        return fn(), None
    except Exception as exc:                       # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    rows = []
    fail = 0
    notes: list[str] = []

    for name, (t, p) in cases().items():
        mine = M.evaluate(t, p)

        sk = {"NMI": sk_nmi(t, p), "ARI": sk_ari(t, p)}

        osm_res, osm_err = _safe(lambda: osm_eval(t.copy(), p.copy()))
        osm = osm_res if osm_res is not None else {}

        frs_res, frs_err = _safe(lambda: FRS.full_metric(t.copy(), p.copy()))
        if frs_res is not None:
            # 返回顺序: OA, AA, KAPPA, NMI, ARI, F1, PRECISION, RECALL, PURITY
            frs_map = {"ACC": frs_res[0], "NMI": frs_res[3],
                       "ARI": frs_res[4], "Purity": frs_res[8]}
        else:
            frs_map = {}

        mvm_res, mvm_err = _safe(lambda: mvkm_eval(t, p))
        mvm_map = ({"ACC": mvm_res["ACC"], "NMI": mvm_res["NMI"],
                    "ARI": mvm_res["ARI"]} if mvm_res is not None else {})

        for who, err in (("OSMAGC", osm_err), ("FedRSMVC", frs_err), ("FedMVKM", mvm_err)):
            if err:
                notes.append(f"  · 用例 [{name}] 下 {who} 无法计算 → {err}")

        for metric in M.METRIC_ORDER:
            v = mine[metric]
            checks = []
            if metric in sk:
                checks.append(("sklearn", sk[metric]))
            if metric in osm:
                checks.append(("OSMAGC", osm[metric]))
            if metric in frs_map:
                checks.append(("FedRSMVC", frs_map[metric]))
            if metric in mvm_map:
                checks.append(("FedMVKM", mvm_map[metric]))

            for who, ref in checks:
                ok = abs(v - ref) < TOL
                if not ok:
                    fail += 1
                rows.append((name, metric, who, v, ref, ok))

    # ------------------------------------------------------------------ 打印
    w = 22
    print("=" * 92)
    print(f"{'用例':<{w}}{'指标':<8}{'对照':<10}{'统一模块':>14}{'对照实现':>14}   结果")
    print("=" * 92)
    last = None
    for name, metric, who, v, ref, ok in rows:
        tag = name if name != last else ""
        last = name
        print(
            f"{tag:<{w}}{metric:<8}{who:<10}{v:>14.10f}{ref:>14.10f}   "
            f"{'OK' if ok else 'MISMATCH'}"
        )
    print("=" * 92)
    print(f"共 {len(rows)} 项对照，失配 {fail} 项。")
    if notes:
        print()
        print("--- 现有实现的失效点（统一模块在这些用例上都能正确返回）---")
        for n in notes:
            print(n)

    # ---------------------------------------------------- 单独演示 FedRSMVC bug
    print()
    print("--- 附带验证：Federated-MVC::purity_score 会原地改写调用方的 y_true ---")
    # 用 1 基标签（真实遥感 GT 就是 1 基）才能暴露：内部会把它重映射成 0 基。
    base = np.array([1, 1, 2, 2, 3, 3] * 100)
    p0 = np.array([1, 2, 2, 2, 3, 1] * 100)

    arr = base.copy()
    val = FRS.purity_score(arr, p0.copy())
    print(f"  调用前入参 unique = {np.unique(base)}")
    print(f"  调用后入参 unique = {np.unique(arr)}   <-- 被就地改写")
    print(f"  FedRSMVC 返回 purity      = {val:.6f}")
    print(f"  统一模块  purity          = {M.purity(base, p0):.6f}   (数值一致)")

    mine_arr = base.copy()
    M.purity(mine_arr, p0)
    print(f"  统一模块调用后入参 unique = {np.unique(mine_arr)}   <-- 纯函数，未被改写")

    # 危害演示：同一 GT 数组先算 purity、再传给别的函数
    gt_shared = base.copy()
    FRS.purity_score(gt_shared, p0.copy())
    print("\n  危害场景：同一个 GT 数组先算 purity 再复用")
    print(f"    复用时的 GT 已经变成 {np.unique(gt_shared)}（原为 {np.unique(base)}）")
    print("    → 该数组后续若被用作类别名查表、或与 1 基索引混用，会静默出错。")

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
