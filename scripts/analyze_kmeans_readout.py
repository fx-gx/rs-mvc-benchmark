"""量化「k-means 读出协议」对 OSMAGC 类方法的分数影响。

背景
----
OSMAGC（以及 Fed-MVKM、FedRSMVC、Local-only）最终的簇标签都来自
「对学到的表示 H 跑一次 k-means」。这一步有两个可以作弊/踩坑的地方：

1. **按 ACC 选初始化**：跑 n_init 次 k-means，用真值挑 ACC 最高的那次。
2. **按 ACC 选轮次**：对每一轮的 H 都跑 k-means，用真值挑最好的一轮。

本脚本量化这两者各自贡献了多少虚高。

⚠️ 核心事实：**k-means 的目标函数（inertia）与聚类准确率（ACC）不一致**。
   因此「跑更多次初始化、挑 inertia 最小的」并不保证 ACC 更好；
   而「挑 ACC 最高的」是纯粹的 oracle 作弊。

运行：python scripts/analyze_kmeans_readout.py --dataset MUUFL --n-inits 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "OSMAGC"))

from common.data import load_dataset                     # noqa: E402
from common.metrics import acc as ACC                    # noqa: E402
from methods.osmagc_central import (                     # noqa: E402
    DEFAULT_PARAMS, _to_osmagc_dataset,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUUFL")
    ap.add_argument("--n-inits", type=int, default=30,
                    help="每个 H 上重复多少次 k-means 初始化")
    ap.add_argument("--max-iter", type=int, default=10)
    args = ap.parse_args()

    from osmagc.core.model import OSMAGCConfig, run_osmagc

    ds = load_dataset(args.dataset)
    gt = ds.flatten_gt("F")
    mask = gt > 0
    k = ds.n_classes
    p = DEFAULT_PARAMS[ds.name]

    print("=" * 96)
    print(f"k-means 读出协议分析 | {ds.name} | 每个 H 跑 {args.n_inits} 次初始化")
    print("=" * 96)
    print(f"  {ds.summary()}")

    ods = _to_osmagc_dataset(ds)
    cfg = OSMAGCConfig(
        num_pixel=list(p.num_pixel), lam=p.lam, gamma=p.gamma, dim=None,
        max_iter=args.max_iter, h_simplex=p.h_simplex,
        mapminmax_axis=p.mapminmax_axis, h_denom_v=p.h_denom_v,
        sagi_mode=p.sagi_mode, verbose=False,
    )
    res = run_osmagc(ods, cfg)
    H_list = res.H_list
    print(f"\n  OSMAGC 产出 {len(H_list)} 轮 H，每轮 shape {H_list[-1].shape}")

    # ---- 逐轮：多次初始化，记录每个解的 (ACC, inertia) ----
    print(f"\n[1] 逐轮扫描（每轮 {args.n_inits} 次初始化）")
    print(f"    {'轮次':<6}{'ACC最小':>10}{'ACC中位':>10}{'ACC最大':>10}{'极差':>10}"
          f"{'inertia最优-ACC':>18}{'oracle-ACC':>12}{'虚高':>10}")
    per_round = []
    for i, H in enumerate(H_list, 1):
        X = np.ascontiguousarray(H.T)
        accs, inerts, labs = [], [], []
        for s in range(args.n_inits):
            km = KMeans(n_clusters=k, n_init=1, random_state=1000 + s)
            lab = km.fit_predict(X).astype(np.int32)
            labs.append(lab)
            accs.append(ACC(gt[mask], lab[mask]))
            inerts.append(km.inertia_)
        accs = np.array(accs)
        inerts = np.array(inerts)
        best_inertia_acc = accs[int(np.argmin(inerts))]   # 诚实：按目标函数选
        oracle_acc = float(accs.max())                    # 作弊：按 ACC 选
        per_round.append(dict(round=i, min=accs.min(), med=np.median(accs),
                              max=accs.max(), honest=best_inertia_acc,
                              oracle=oracle_acc,
                              spread=accs.max() - accs.min()))
        print(f"    {i:<6}{accs.min():>10.4f}{np.median(accs):>10.4f}{accs.max():>10.4f}"
              f"{accs.max() - accs.min():>10.4f}{best_inertia_acc:>18.4f}"
              f"{oracle_acc:>12.4f}{oracle_acc - best_inertia_acc:>+10.4f}")

    # ---- 汇总三种协议 ----
    last = per_round[-1]
    honest = last["honest"]
    oracle_both = max(r["oracle"] for r in per_round)

    print(f"\n[2] 三种协议的最终 ACC")
    print(f"    (a) 诚实：最后一轮 + 按 inertia 选初始化          ACC = {honest:.4f}")
    print(f"    (b) 只按 ACC 选初始化（最后一轮）                 ACC = {last['oracle']:.4f}"
          f"   虚高 {last['oracle'] - honest:+.4f}")
    print(f"    (c) 官方 oracle：按 ACC 选初始化 **且** 选轮次     ACC = {oracle_both:.4f}"
          f"   虚高 {oracle_both - honest:+.4f}")

    print(f"\n[3] 稳定性")
    spreads = [r["spread"] for r in per_round]
    print(f"    同一轮内 ACC 极差：均值 {np.mean(spreads):.4f}，最大 {np.max(spreads):.4f}")
    print(f"    各轮 oracle-ACC 的极差：{max(r['oracle'] for r in per_round) - min(r['oracle'] for r in per_round):.4f}")
    all_honest = [r["honest"] for r in per_round]
    print(f"    各轮诚实-ACC 的极差  ：{max(all_honest) - min(all_honest):.4f}")

    print(f"\n[4] 结论")
    print(f"    · 同一个 H、同一份数据，仅换 k-means 初始化，ACC 极差可达 "
          f"{np.max(spreads):.4f}")
    print(f"    · 「按 ACC 选初始化」单独就贡献 {last['oracle'] - honest:+.4f}；")
    print(f"      再加上「按 ACC 选轮次」，合计虚高 {oracle_both - honest:+.4f}")
    print(f"    · 因此本平台强制：**所有方法用同一套诚实 k-means 读出协议**")
    print(f"      （固定 n_init、按 inertia 选初始化、固定用最后一轮），")
    print(f"      并在论文里单独报一列 oracle 值供参考。")
    print("=" * 96)

    # 落盘
    out = ROOT / "results" / f"kmeans_readout_{ds.name}.json"
    import json
    out.write_text(json.dumps(
        {"dataset": ds.name, "n_inits": args.n_inits, "k": k,
         "per_round": per_round,
         "honest_acc": honest, "oracle_init_acc": last["oracle"],
         "oracle_both_acc": oracle_both},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
