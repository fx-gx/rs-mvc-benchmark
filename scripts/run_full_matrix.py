"""全流程跑批：把所有已接入的方法 × 数据集 × 划分 × 种子跑一遍，汇总成一张表。

设计要点
--------
- **多种子**：每个配置跑 N 个种子，报 均值 ± 标准差（单种子数字不可信）。
- **诚实读出协议**：所有方法一律 sklearn KMeans(n_init=10, 按 inertia 选初始化)，
  固定种子，绝不用真值挑结果。
- **失败不中断**：某个 (方法, 数据集) 组合出错就记 error 并继续，最终报表里显示为 ERR。
- **落盘**：每条记录写 JSONL，可断点续跑。

当前接入的方法
--------------
| 方法              | 联邦设定      | 适用划分            |
|-------------------|---------------|---------------------|
| Local-only        | 无（下界）    | 视图划分 + 区域划分 |
| OSMAGC 集中式     | 不联邦（上界）| 无需划分            |
| OSMAGC 联邦化     | 视图划分      | 视图划分（已验证无损）|
| Fed-MVKM          | 水平联邦      | 仅区域/随机划分     |

运行：
    python scripts/run_full_matrix.py
    python scripts/run_full_matrix.py --datasets Augsburg MUUFL --seeds 0
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_all_views, load_dataset          # noqa: E402
from common.metrics import evaluate                           # noqa: E402
from common.partition import (                                # noqa: E402
    region_grid_partition, view_partition,
)
from common.protocol import (                                 # noqa: E402
    DEFAULT_LOG, RunResult, append_result,
)
from methods.amksc import amksc_applicable, run_amksc         # noqa: E402
from methods.fed_mvkm import run_fed_mvkm                     # noqa: E402
from methods.local_only import (                              # noqa: E402
    run_region_local_only, run_view_local_only,
)
from methods.osmagc_central import run_osmagc_central         # noqa: E402
from methods.osmagc_federated import run_osmagc_federated     # noqa: E402

METRICS = ("ACC", "NMI", "ARI", "Purity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Augsburg", "MUUFL", "Trento", "Salinas"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--norm", default="zscore")
    ap.add_argument("--grid", default="2x2", help="区域划分的网格，如 2x2")
    ap.add_argument("--skip-osmagc", action="store_true")
    ap.add_argument("--skip-fedmvkm", action="store_true")
    ap.add_argument("--skip-amksc", action="store_true")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    log = Path(args.log)
    r, c = (int(x) for x in args.grid.split("x"))
    K = r * c

    # results[(dataset, method)] = list of metric dicts
    results: dict[tuple[str, str], list[dict]] = {}
    errors: dict[tuple[str, str], str] = {}

    def record(ds, method, partition, mode, seed, metrics, runtime, extra, status="ok", note=""):
        key = (ds.name, method)
        if status == "ok":
            results.setdefault(key, []).append(metrics)
        else:
            errors[key] = note
        append_result(RunResult(
            dataset=ds.name, method=method, partition=partition,
            partition_mode=mode, n_clients=0, eval_mask="labeled",
            norm=args.norm, seed=seed,
            ACC=metrics.get("ACC", float("nan")) if metrics else float("nan"),
            NMI=metrics.get("NMI", float("nan")) if metrics else float("nan"),
            ARI=metrics.get("ARI", float("nan")) if metrics else float("nan"),
            Purity=metrics.get("Purity", float("nan")) if metrics else float("nan"),
            runtime_s=runtime, n_clusters=ds.n_classes,
            status=status, note=note, extra=extra or {},
        ), log)

    for dname in args.datasets:
        print("=" * 100)
        print(f"数据集 {dname}")
        print("=" * 100)
        try:
            ds = load_dataset(dname)
            views = load_all_views(ds, mode=args.norm)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  加载失败：{exc}")
            continue
        gt = ds.flatten_gt("F")
        mask = gt > 0
        print(f"  {ds.summary()}")

        pv = view_partition(ds)
        pg = region_grid_partition(ds, rows=r, cols=c)

        def sc(lab):
            return evaluate(gt[mask], lab[mask])

        for seed in args.seeds:
            print(f"\n  --- seed={seed} ---")

            # ---- 0. Local-only（视图划分，严格不协作：各视图独自评分取宏平均）----
            # 这是定义最干净的下界：无融合、无自由参数（不涉及"锚点怎么选"）。
            try:
                import time as _t
                t0 = _t.time()
                per = []
                for X in views:
                    from sklearn.cluster import KMeans
                    lab_v = KMeans(n_clusters=ds.n_classes, n_init=10,
                                   random_state=seed).fit_predict(X)
                    per.append(sc(lab_v))
                m = {k: float(np.mean([p[k] for p in per])) for k in METRICS}
                print(f"    Local-only(view,朴素) ACC={m['ACC']:.4f} NMI={m['NMI']:.4f}")
                record(ds, "Local-only (view, 不融合)", pv.shape, pv.mode, seed, m,
                       _t.time() - t0, {"per_view_ACC": [round(p['ACC'], 4) for p in per]})
            except Exception as exc:                                # noqa: BLE001
                print(f"    Local-only(view,朴素) 失败: {exc}")
                record(ds, "Local-only (view, 不融合)", pv.shape, pv.mode, seed, None, 0,
                       None, "error", f"{type(exc).__name__}: {exc}")

            # ---- 1. Local-only（视图划分 + 朴素融合）----
            try:
                lab, det = run_view_local_only(views, ds.n_classes, seed=seed,
                                               return_detail=True)
                m = sc(lab)
                print(f"    Local-only(view,融合) ACC={m['ACC']:.4f} NMI={m['NMI']:.4f}")
                record(ds, "Local-only (view, 融合)", pv.shape, pv.mode, seed, m,
                       det["runtime_s"], {"anchor": det["anchor_idx"]})
            except Exception as exc:                                # noqa: BLE001
                print(f"    Local-only(view,融合) 失败: {exc}")
                record(ds, "Local-only (view, 融合)", pv.shape, pv.mode, seed, None, 0,
                       None, "error", f"{type(exc).__name__}: {exc}")

            # ---- 2. Local-only（区域划分）----
            try:
                lab, det = run_region_local_only(
                    views, [cl.sample_idx for cl in pg.clients],
                    ds.n_classes, seed=seed, return_detail=True)
                m = sc(lab)
                print(f"    Local-only(region)    ACC={m['ACC']:.4f} NMI={m['NMI']:.4f}")
                record(ds, "Local-only (region)", pg.shape, pg.mode, seed, m,
                       det["runtime_s"], {})
            except Exception as exc:                                # noqa: BLE001
                print(f"    Local-only(region) 失败: {exc}")
                record(ds, "Local-only (region)", pg.shape, pg.mode, seed, None, 0, None,
                       "error", f"{type(exc).__name__}: {exc}")

            # ---- 3. OSMAGC 集中式（上界）----
            if not args.skip_osmagc:
                try:
                    lab, det = run_osmagc_central(ds, seed=seed)
                    m = sc(lab)
                    print(f"    OSMAGC 集中式         ACC={m['ACC']:.4f} NMI={m['NMI']:.4f}"
                          f"  ({det['runtime_s']:.0f}s)")
                    record(ds, "OSMAGC (central, upper bound)", "none", "central", seed,
                           m, det["runtime_s"], {"alpha": det["alpha"]})
                except Exception as exc:                            # noqa: BLE001
                    print(f"    OSMAGC 集中式 失败: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    record(ds, "OSMAGC (central, upper bound)", "none", "central", seed,
                           None, 0, None, "error", f"{type(exc).__name__}: {exc}")

                # ---- 4. OSMAGC 联邦化（视图划分）----
                try:
                    lab, det = run_osmagc_federated(ds, seed=seed)
                    m = sc(lab)
                    comm = det["comm"]
                    print(f"    OSMAGC 联邦化(view)   ACC={m['ACC']:.4f} NMI={m['NMI']:.4f}"
                          f"  通信={comm.human(comm.total)}")
                    record(ds, "OSMAGC (federated, view)", pv.shape, pv.mode, seed, m,
                           det["runtime_s"],
                           {"comm_bytes": comm.total, "alpha": det["alpha"]})
                except Exception as exc:                            # noqa: BLE001
                    print(f"    OSMAGC 联邦化 失败: {type(exc).__name__}: {exc}")
                    record(ds, "OSMAGC (federated, view)", pv.shape, pv.mode, seed,
                           None, 0, None, "error", f"{type(exc).__name__}: {exc}")

            # ---- 5. Fed-MVKM（区域划分，水平联邦）----
            if not args.skip_fedmvkm:
                try:
                    lab, det = run_fed_mvkm(ds, pg, views, seed=seed)
                    m = sc(lab)
                    print(f"    Fed-MVKM(region {args.grid})   ACC={m['ACC']:.4f} "
                          f"NMI={m['NMI']:.4f}  ({det['n_iterations']}轮)")
                    record(ds, "Fed-MVKM (region)", pg.shape, pg.mode, seed, m,
                           det["runtime_s"], {"n_iterations": det["n_iterations"]})
                except Exception as exc:                            # noqa: BLE001
                    print(f"    Fed-MVKM 失败: {type(exc).__name__}: {exc}")
                    record(ds, "Fed-MVKM (region)", pg.shape, pg.mode, seed,
                           None, 0, None, "error", f"{type(exc).__name__}: {exc}")

            # ---- 6. AMKSC（遥感专用 · 逐像素空间块）----
            if not args.skip_amksc:
                ok_a, why_a = amksc_applicable(ds)
                if not ok_a:
                    # 「不适用」本身即结论，与 Fed-MVKM 在视图划分下不可运行同类
                    print(f"    AMKSC 不适用: {why_a}")
                    record(ds, "AMKSC", "none", "central", seed,
                           None, 0, None, "na", why_a)
                else:
                    try:
                        t0a = _t.time()
                        lab, det = run_amksc(ds, seed=seed)
                        el_a = _t.time() - t0a
                        m = sc(lab)
                        print(f"    AMKSC                 ACC={m['ACC']:.4f} "
                              f"NMI={m['NMI']:.4f}  ({el_a:.0f}s, {det['device']})")
                        record(ds, "AMKSC", "none", "central", seed, m, el_a,
                               {k: v for k, v in det.items() if k != "mu"})
                    except Exception as exc:                        # noqa: BLE001
                        print(f"    AMKSC 失败: {type(exc).__name__}: {exc}")
                        record(ds, "AMKSC", "none", "central", seed,
                               None, 0, None, "error", f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- 汇总报表
    print("\n")
    print("=" * 100)
    print(f"汇总（{len(args.seeds)} 个种子，均值 ± 标准差）")
    print("=" * 100)
    methods = [
        "OSMAGC (central, upper bound)",
        "OSMAGC (federated, view)",
        "AMKSC",
        "Fed-MVKM (region)",
        "Local-only (view, 不融合)",
        "Local-only (view, 融合)",
        "Local-only (region)",
    ]
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        head = f"  {'方法':<30}" + "".join(f"{d:>18}" for d in args.datasets)
        print(head)
        print("  " + "-" * (len(head) - 2))
        for meth in methods:
            cells = []
            for d in args.datasets:
                vals = [mm[metric] for mm in results.get((d, meth), [])]
                if vals:
                    cells.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}")
                elif (d, meth) in errors:
                    cells.append("ERR")
                else:
                    cells.append("N/A")
            print(f"  {meth:<30}" + "".join(f"{x:>18}" for x in cells))

    print(f"\n--- 无法运行的组合 ---")
    if errors:
        for (d, m), note in errors.items():
            print(f"  {d:<12} {m:<30} {note[:80]}")
    else:
        print("  （无）")

    print("\n--- 说明 ---")
    print("  · OSMAGC 联邦化与集中式【逐元素相同】，此处两者的逐种子差异仅来自 k-means 随机初始化")
    print("  · Fed-MVKM 是水平联邦，视图划分下不适用（未列入，属方法设定所限）")
    print("  · 全部数字由 common/metrics.py 统一计算，未采信任何方法自报值")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
