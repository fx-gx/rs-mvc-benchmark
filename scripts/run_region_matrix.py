"""区域划分（范式 A）下的方法对比矩阵。

这一轮把「普通联邦多视图聚类」第一次跑起来，并配上它的两个参照：
    Local-only（下界）  <  Fed-MVKM（联邦）  <  OSMAGC 集中式（上界）

同时演示并记录 Fed-MVKM 在**视图划分**下的 N/A（它要求客户端持有全部视图）。

运行：
    python scripts/run_region_matrix.py --dataset Augsburg
    python scripts/run_region_matrix.py --dataset Augsburg --grids 2x2 3x3
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
    random_partition, region_grid_partition, view_partition,
)
from common.protocol import (                                 # noqa: E402
    DEFAULT_LOG, RunResult, append_result, done_keys,
)
from methods.fed_mvkm import fed_mvkm_applicable, run_fed_mvkm  # noqa: E402
from methods.local_only import run_region_local_only          # noqa: E402


def parse_grid(s: str) -> tuple[int, int]:
    a, b = s.lower().split("x")
    return int(a), int(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Augsburg")
    ap.add_argument("--grids", nargs="+", default=["2x2", "3x3"])
    ap.add_argument("--norm", default="zscore")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-k", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    log = Path(args.log)
    ds = load_dataset(args.dataset)
    views = load_all_views(ds, mode=args.norm)
    gt = ds.flatten_gt("F")
    mask = gt > 0

    print("=" * 100)
    print(f"区域划分矩阵 | {ds.name} | 归一化={args.norm} | seed={args.seed}")
    print("=" * 100)
    print(f"  {ds.summary()}")
    print(f"  评测口径：{int(mask.sum()):,} 个有标注像素（gt>0），"
          f"聚类参与 {ds.n_pixels:,} 个像素")

    rows: list[tuple] = []

    # ---------------- Fed-MVKM 在视图划分下不可用 ----------------
    pv = view_partition(ds)
    ok, why = fed_mvkm_applicable(pv)
    print(f"\n[0] Fed-MVKM 在视图划分下是否可用？ → {'可用' if ok else '不可用'}")
    if not ok:
        print(f"    原因：{why}")
        print(f"    → 这一格记 N/A。这正是导师要的「方法对数据结构的硬依赖」的证据。")
        append_result(RunResult(
            dataset=ds.name, method="Fed-MVKM", partition=pv.shape,
            partition_mode=pv.mode, n_clients=len(pv), seed=args.seed,
            status="skip", note=why,
        ), log)

    # ---------------- 各划分下的对比 ----------------
    for gs in args.grids:
        r, c = parse_grid(gs)
        part = region_grid_partition(ds, rows=r, cols=c)
        print(f"\n{'=' * 100}")
        print(f"划分 = {gs} 网格（{len(part)} 个客户端）")
        print(f"  JS散度 均值 {part.stats['JS散度(均值)']:.4f} / "
              f"最大 {part.stats['JS散度(最大)']:.4f}  "
              f"缺类客户端 {part.stats['缺类的客户端数']} 个")
        print(f"  各客户端像素数 {part.stats['各客户端像素数']}")

        # --- Local-only ---
        lab_lo, det_lo = run_region_local_only(
            views, [c.sample_idx for c in part.clients], ds.n_classes,
            seed=args.seed, return_detail=True)
        m_lo = evaluate(gt[mask], lab_lo[mask])
        print(f"\n  [Local-only]  各区域独立聚类再拼接（下界）")
        print(f"    ACC={m_lo['ACC']:.4f}  NMI={m_lo['NMI']:.4f}  "
              f"ARI={m_lo['ARI']:.4f}  Purity={m_lo['Purity']:.4f}  "
              f"({det_lo['runtime_s']:.1f}s)")
        append_result(RunResult(
            dataset=ds.name, method="Local-only", partition=part.shape,
            partition_mode=part.mode, n_clients=len(part),
            eval_mask="labeled", norm=args.norm, seed=args.seed,
            ACC=m_lo["ACC"], NMI=m_lo["NMI"], ARI=m_lo["ARI"], Purity=m_lo["Purity"],
            runtime_s=det_lo["runtime_s"], n_clusters=ds.n_classes,
            extra={"js_mean": part.stats["JS散度(均值)"],
                   "js_max": part.stats["JS散度(最大)"],
                   "client_sizes": part.stats["各客户端像素数"]},
        ), log)
        rows.append(("Local-only", gs, m_lo))

        # --- Fed-MVKM ---
        print(f"\n  [Fed-MVKM]  普通联邦多视图聚类（横向联邦）")
        try:
            lab_fm, det_fm = run_fed_mvkm(
                ds, part, views, seed=args.seed,
                beta_k=args.beta_k, max_iterations=args.max_iter)
            m_fm = evaluate(gt[mask], lab_fm[mask])
            print(f"    迭代 {det_fm['n_iterations']} 轮，"
                  f"收敛={det_fm['converged']}，耗时 {det_fm['runtime_s']:.1f}s")
            print(f"    ACC={m_fm['ACC']:.4f}  NMI={m_fm['NMI']:.4f}  "
                  f"ARI={m_fm['ARI']:.4f}  Purity={m_fm['Purity']:.4f}")
            print(f"    视图权重（客户端0）= {[round(v, 3) for v in det_fm['view_weights'][0]]}")
            append_result(RunResult(
                dataset=ds.name, method="Fed-MVKM", partition=part.shape,
                partition_mode=part.mode, n_clients=len(part),
                eval_mask="labeled", norm=args.norm, seed=args.seed,
                ACC=m_fm["ACC"], NMI=m_fm["NMI"], ARI=m_fm["ARI"], Purity=m_fm["Purity"],
                runtime_s=det_fm["runtime_s"], n_clusters=ds.n_classes,
                extra={"n_iterations": det_fm["n_iterations"],
                       "converged": det_fm["converged"],
                       "beta_k": args.beta_k,
                       "view_weights_client0": det_fm["view_weights"][0],
                       "js_max": part.stats["JS散度(最大)"]},
            ), log)
            rows.append(("Fed-MVKM", gs, m_fm))

            gap = m_fm["ACC"] - m_lo["ACC"]
            print(f"    → 相对 Local-only 下界：ACCC gap = {gap:+.4f}")
        except Exception as exc:                          # noqa: BLE001
            print(f"    运行失败：{type(exc).__name__}: {exc}")
            append_result(RunResult(
                dataset=ds.name, method="Fed-MVKM", partition=part.shape,
                partition_mode=part.mode, n_clients=len(part), seed=args.seed,
                status="error", note=f"{type(exc).__name__}: {exc}",
            ), log)
            rows.append(("Fed-MVKM", gs, None))

        # --- 随机划分对照（打散空间结构）---
        part_r = random_partition(ds, m=r * c, seed=args.seed)
        lab_r, det_r = run_region_local_only(
            views, [c.sample_idx for c in part_r.clients], ds.n_classes,
            seed=args.seed, return_detail=True)
        m_r = evaluate(gt[mask], lab_r[mask])
        lab_fr, det_fr = run_fed_mvkm(ds, part_r, views, seed=args.seed,
                                      beta_k=args.beta_k, max_iterations=args.max_iter)
        m_fr = evaluate(gt[mask], lab_fr[mask])
        print(f"\n  [对照] 同样 {r * c} 个客户端，但改成**随机划分**（打散空间结构）")
        print(f"    Local-only  ACC={m_r['ACC']:.4f}   Fed-MVKM  ACC={m_fr['ACC']:.4f}")
        print(f"    → 对比网格划分：Local-only 差 {m_r['ACC'] - m_lo['ACC']:+.4f}，"
              f"Fed-MVKM 差 {m_fr['ACC'] - m_fm['ACC']:+.4f}")
        print(f"      （差值反映「空间结构」本身值多少）")

    # ---------------- 汇总 ----------------
    print(f"\n{'=' * 100}")
    print("汇总（区域划分，ACC）")
    print(f"  {'方法':<14}" + "".join(f"{g:>10}" for g in args.grids))
    for meth in ("Local-only", "Fed-MVKM"):
        vals = []
        for g in args.grids:
            mm = next((x[2] for x in rows if x[0] == meth and x[1] == g), None)
            vals.append(f"{mm['ACC']:.4f}" if mm else "N/A")
        print(f"  {meth:<14}" + "".join(f"{v:>10}" for v in vals))
    print(f"  {'（上界）OSMAGC':<14}" + "".join(f"{'集中式':>10}" for _ in args.grids)
          + "   ← 见 run_osmagc_central.py")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
