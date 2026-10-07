"""验证 Fed-MVKM 在 Augsburg 上的结果是否可信。

要排查三件事
------------
1. **是不是我的适配器写错了** —— 用 M=1（单个客户端吃掉全部像素）跑一遍。
   M=1 就是「不联邦的集中式多视图 k-means」，它必须 >= 同数据的联邦 M>1 版本；
   若 M=1 反而更低，说明适配器有问题。
2. **beta_k 的影响有多大** —— 原复现是按数据集单独调过 beta_k 的，
   这里用的默认 1.0 未必合适。扫一遍看结论是否稳健。
3. **随机种子的影响** —— 至少跑 3 个种子。

运行：python scripts/validate_fed_mvkm.py --dataset Augsburg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_all_views, load_dataset          # noqa: E402
from common.metrics import evaluate                           # noqa: E402
from common.partition import (                                # noqa: E402
    Partition, Client, random_partition, region_grid_partition,
)
from methods.fed_mvkm import run_fed_mvkm                     # noqa: E402


def m1_partition(ds) -> Partition:
    """M=1：单个客户端持有全部像素、全部视图（即不联邦）。"""
    return Partition(
        mode="region", shape="M1", grid=ds.grid_shape(),
        n_views=ds.n_views, n_pixels=ds.n_pixels,
        clients=[Client(0, "all", list(range(ds.n_views)),
                        np.arange(ds.n_pixels, dtype=np.int64))],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Augsburg")
    ap.add_argument("--grid", default="2x2")
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    views = load_all_views(ds, mode="zscore")
    gt = ds.flatten_gt("F")
    mask = gt > 0

    def score(lab):
        return evaluate(gt[mask], lab[mask])

    r, c = (int(x) for x in args.grid.split("x"))
    p_grid = region_grid_partition(ds, rows=r, cols=c)

    print("=" * 96)
    print(f"Fed-MVKM 结果可信性验证 | {ds.name}")
    print("=" * 96)

    # ---------------- 1. M=1（不联邦）是否为上界 ----------------
    print("\n[1] M=1（单个客户端吃掉全部像素）—— 这是「不联邦」的集中式多视图 k-means")
    print(f"    {'beta_k':>8}{'M=1 ACC':>12}{'grid ACC':>12}{'random ACC':>12}")
    for bk in (0.1, 1.0, 10.0):
        lab1, _ = run_fed_mvkm(ds, m1_partition(ds), views, seed=0,
                               beta_k=bk, max_iterations=args.max_iter)
        labg, _ = run_fed_mvkm(ds, p_grid, views, seed=0,
                               beta_k=bk, max_iterations=args.max_iter)
        pr = random_partition(ds, m=r * c, seed=0)
        labr, _ = run_fed_mvkm(ds, pr, views, seed=0,
                               beta_k=bk, max_iterations=args.max_iter)
        m1, mg, mr = score(lab1), score(labg), score(labr)
        print(f"    {bk:>8}{m1['ACC']:>12.4f}{mg['ACC']:>12.4f}{mr['ACC']:>12.4f}")
    print("    → 预期：M=1 >= grid（联邦不该优于不联邦）。若 M=1 明显更低则适配器有问题。")

    # ---------------- 2. beta_k 扫描 ----------------
    print(f"\n[2] beta_k 敏感性（{args.grid} 网格，seed=0）")
    print(f"    {'beta_k':>10}{'ACC':>10}{'NMI':>10}{'ARI':>10}{'Purity':>10}")
    grid_rows = []
    for bk in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        lab, _ = run_fed_mvkm(ds, p_grid, views, seed=0,
                              beta_k=bk, max_iterations=args.max_iter)
        m = score(lab)
        grid_rows.append((bk, m))
        print(f"    {bk:>10}{m['ACC']:>10.4f}{m['NMI']:>10.4f}{m['ARI']:>10.4f}{m['Purity']:>10.4f}")
    best_bk = max(grid_rows, key=lambda x: x[1]["ACC"])
    print(f"    → 最优 beta_k = {best_bk[0]}（ACC {best_bk[1]['ACC']:.4f}）")

    # ---------------- 3. 种子稳定性 ----------------
    print(f"\n[3] 种子稳定性（{args.grid} 网格，beta_k={best_bk[0]}）")
    accs = []
    for s in range(3):
        lab, _ = run_fed_mvkm(ds, p_grid, views, seed=s,
                              beta_k=best_bk[0], max_iterations=args.max_iter)
        m = score(lab)
        accs.append(m["ACC"])
        print(f"    seed={s}  ACC={m['ACC']:.4f}  NMI={m['NMI']:.4f}")
    print(f"    均值 {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # ---------------- 4. 与 OSMAGC 集中式对比 ----------------
    print(f"\n[4] 与 OSMAGC 集中式对比（同为 Augsburg、同一评测口径）")
    print(f"    OSMAGC 集中式（诚实读出）  ACC = 0.4602")
    print(f"    Fed-MVKM 最优（{args.grid}，beta_k={best_bk[0]}） ACC = {best_bk[1]['ACC']:.4f}")
    print(f"    → 差值 {best_bk[1]['ACC'] - 0.4602:+.4f}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
