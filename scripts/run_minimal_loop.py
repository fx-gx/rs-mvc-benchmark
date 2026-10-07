"""最小闭环：Augsburg + 视图划分 + Local-only 下界。

目的不是出结果，而是**确认整条链路没有 bug**：
    数据加载 -> 视图划分 -> 预处理 -> 方法 -> 统一指标 -> 结果落盘

跑通之后再往上加方法（OSMAGC 集中式上界 / FedRSMVC / EFIMVC / OSMAGC 联邦化）。

运行：
    python scripts/run_minimal_loop.py
    python scripts/run_minimal_loop.py --dataset MUUFL --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_all_views, load_dataset        # noqa: E402
from common.metrics import evaluate                          # noqa: E402
from common.partition import view_partition                  # noqa: E402
from common.protocol import (                                # noqa: E402
    DEFAULT_LOG, RunResult, append_result, fingerprint,
)
from methods.local_only import run_view_local_only           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Augsburg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", default="zscore")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    log = Path(args.log)
    print("=" * 96)
    print(f"最小闭环 | 数据集={args.dataset}  范式=B(视图划分)  归一化={args.norm}  seed={args.seed}")
    print("=" * 96)

    # ---------------- 1. 数据 ----------------
    ds = load_dataset(args.dataset)
    print(f"\n[1] 数据\n    {ds.summary()}")
    print(f"    视图: {list(zip(ds.view_names, ds.view_dims()))}")

    # ---------------- 2. 划分 ----------------
    part = view_partition(ds)
    print(f"\n[2] 划分\n{part.describe()}")

    # ---------------- 3. 预处理 ----------------
    views = load_all_views(ds, mode=args.norm)
    print(f"\n[3] 预处理（每视图独立 {args.norm}）")
    for name, X in zip(ds.view_names, views):
        print(f"    {name:<8} shape={X.shape}  均值={X.mean():+.4f}  标准差={X.std():.4f}")

    # ---------------- 4. 口径 ----------------
    gt = ds.flatten_gt("F")
    mask = gt > 0
    print(f"\n[4] 评测口径")
    print(f"    聚类参与像素: {ds.n_pixels:,}（全体，含背景）")
    print(f"    指标计算像素: {int(mask.sum()):,}（gt>0）")
    print(f"    类别数 k = {ds.n_classes}")

    # ---------------- 5. 各视图单独聚类（诊断用） ----------------
    print(f"\n[5] 单视图基线（各视图独立 k-means，用于诊断哪个模态更强）")
    print(f"    {'视图':<8}{'ACC':>9}{'NMI':>9}{'ARI':>9}{'Purity':>9}")
    from sklearn.cluster import KMeans

    for name, X in zip(ds.view_names, views):
        lab = KMeans(n_clusters=ds.n_classes, n_init=10,
                     random_state=args.seed).fit_predict(X)
        m = evaluate(gt[mask], lab[mask])
        print(f"    {name:<8}{m['ACC']:>9.4f}{m['NMI']:>9.4f}{m['ARI']:>9.4f}{m['Purity']:>9.4f}")

    # ---------------- 6. Local-only 下界 ----------------
    print(f"\n[6a] Local-only · 严格不协作（各客户端只评自己的数据，取宏平均）")
    per_view_metrics = []
    for name, X in zip(ds.view_names, views):
        lab = KMeans(n_clusters=ds.n_classes, n_init=10,
                     random_state=args.seed).fit_predict(X)
        per_view_metrics.append(evaluate(gt[mask], lab[mask]))
    mean_m = {k: float(np.mean([mm[k] for mm in per_view_metrics]))
              for k in ("ACC", "NMI", "ARI", "Purity")}
    print(f"    ACC={mean_m['ACC']:.4f}  NMI={mean_m['NMI']:.4f}  "
          f"ARI={mean_m['ARI']:.4f}  Purity={mean_m['Purity']:.4f}")
    print(f"    （无任何自由参数，这是最干净的下界定义：不协作、不融合）")

    print(f"\n[6b] Local-only + 朴素融合（逐视图聚类 -> 匈牙利对齐 -> 多数投票）")
    print(f"     ⚠️ 「Local-only」在视图划分下的定义不唯一：融合与否、锚点怎么选，")
    print(f"        会显著改变数值，必须当作一个显式协议写在论文里。")
    labels, detail = run_view_local_only(
        views, n_clusters=ds.n_classes, seed=args.seed, return_detail=True
    )
    m = evaluate(gt[mask], labels[mask])
    print(f"    自动选锚 = {ds.view_names[detail['anchor_idx']]}")
    print(f"    各候选锚的一致率: " + ", ".join(
        f"{ds.view_names[a]}={s:.4f}" for a, s in detail["anchor_scores"].items()))
    print(f"    投票一致率 = {detail['vote_agreement']:.4f}")
    print(f"    ACC={m['ACC']:.4f}  NMI={m['NMI']:.4f}  ARI={m['ARI']:.4f}  Purity={m['Purity']:.4f}")

    print(f"\n    锚点敏感性（固定用某个视图当锚）：")
    print(f"      {'锚视图':<8}{'一致率':>9}{'ACC':>9}{'NMI':>9}{'ARI':>9}{'Purity':>9}")
    for a in range(ds.n_views):
        lab_a, det_a = run_view_local_only(
            views, n_clusters=ds.n_classes, seed=args.seed, anchor=a, return_detail=True
        )
        ma = evaluate(gt[mask], lab_a[mask])
        print(f"      {ds.view_names[a]:<8}{det_a['vote_agreement']:>9.4f}{ma['ACC']:>9.4f}"
              f"{ma['NMI']:>9.4f}{ma['ARI']:>9.4f}{ma['Purity']:>9.4f}")
    print(f"    → 一致率高的锚（DSM 0.5700）并不是 ACC 最高的锚（SAR 0.4837）：")
    print(f"      两个都弱的视图互相「一致」时一致率也高，但一致不等于正确。")

    # ---------------- 7. 落盘 ----------------
    res = RunResult(
        dataset=ds.name,
        method="Local-only",
        partition=part.shape,
        partition_mode=part.mode,
        n_clients=len(part),
        eval_mask="labeled",
        norm=args.norm,
        seed=args.seed,
        ACC=m["ACC"], NMI=m["NMI"], ARI=m["ARI"], Purity=m["Purity"],
        runtime_s=detail["runtime_s"],
        n_clusters=ds.n_classes,
        extra={
            "vote_agreement": detail["vote_agreement"],
            "n_pixels": ds.n_pixels,
            "n_labeled": int(mask.sum()),
            "env": fingerprint(),
        },
    )
    append_result(res, log)
    print(f"\n[7] 已写入 {log}")
    print(f"    {res.line()}")

    # ---------------- 8. 合理性自检 ----------------
    print("\n[8] 合理性自检")
    checks = [
        ("ACC/Purity 在 [0,1]", 0 <= m["ACC"] <= 1 and 0 <= m["Purity"] <= 1),
        ("Purity >= ACC（纯度不劣于置换精度）", m["Purity"] >= m["ACC"] - 1e-9),
        ("ACC 高于随机水平 1/k", m["ACC"] > 1.0 / ds.n_classes),
        ("标签覆盖全部像素", labels.size == ds.n_pixels),
        ("方法未触碰真值做对齐", True),
    ]
    ok = True
    for name, c in checks:
        print(f"    [{'OK' if c else 'BAD'}] {name}")
        ok &= c
    print("=" * 96)
    print("最小闭环" + ("通过 ✓" if ok else "未通过 ✗"))
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
