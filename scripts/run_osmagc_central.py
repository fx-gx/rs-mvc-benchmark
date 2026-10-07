"""OSMAGC 集中式上界 —— 平台的第一把公共尺子。

同时做三件事：
1. 跑通 OSMAGC 集中式，产出像素级标签
2. 用统一指标重算，并与 OSMAGC 自带指标模块对拍（确认口径一致）
3. 量化「官方 oracle 协议」带来的虚高（按 ACC 选轮次/初始化）

运行：python scripts/run_osmagc_central.py --dataset Augsburg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_dataset                       # noqa: E402
from common.metrics import evaluate                        # noqa: E402
from common.protocol import (                              # noqa: E402
    DEFAULT_LOG, RunResult, append_result, fingerprint,
)
from methods.osmagc_central import (                       # noqa: E402
    DEFAULT_PARAMS, run_osmagc_central,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Augsburg")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    print("=" * 96)
    print(f"OSMAGC 集中式 | 数据集={args.dataset}  seed={args.seed}")
    print("=" * 96)

    ds = load_dataset(args.dataset)
    print(f"\n[1] 数据\n    {ds.summary()}")

    p = DEFAULT_PARAMS[ds.name]
    print(f"\n[2] OSMAGC 配置（取自 OSMAGC 项目实测最优）")
    print(f"    锚点数 num_pixel = {list(p.num_pixel)}")
    print(f"    λ = {p.lam:g}   γ = {p.gamma:g}   max_iter = {args.max_iter}")
    print(f"    开关: h_simplex={p.h_simplex}  mapminmax_axis={p.mapminmax_axis}  "
          f"h_denom_v={p.h_denom_v}")

    print(f"\n[3] 运行 OSMAGC（无 oracle，最后一轮 H + sklearn KMeans）...")
    from dataclasses import replace
    labels, detail = run_osmagc_central(
        ds, params=replace(p, max_iter=args.max_iter),
        seed=args.seed, n_init=args.n_init, verbose=args.verbose,
    )
    print(f"    耗时 {detail['runtime_s']:.1f}s，共 {detail['n_rounds']} 轮")
    print(f"    最终视图权重 α = {[round(a, 4) for a in detail['alpha']]}")

    gt = ds.flatten_gt("F")
    mask = gt > 0

    m = evaluate(gt[mask], labels[mask])
    print(f"\n[4] 统一指标（无 oracle）")
    print(f"    ACC={m['ACC']:.4f}  NMI={m['NMI']:.4f}  ARI={m['ARI']:.4f}  Purity={m['Purity']:.4f}")

    # ---- 与 OSMAGC 自带指标模块对拍 ----
    sys.path.insert(0, str(ROOT.parent / "OSMAGC"))
    from osmagc.metrics import evaluate_results_clustering as osm_eval
    om = osm_eval(gt[mask].copy(), labels[mask].copy())
    print(f"\n[5] 与 OSMAGC 自带指标模块对拍")
    print(f"    {'指标':<8}{'统一模块':>12}{'OSMAGC':>12}{'差':>12}")
    for k, kk in (("ACC", "acc"), ("NMI", "nmi"), ("ARI", "ari"), ("Purity", "purity")):
        d = m[k] - om[kk]
        print(f"    {k:<8}{m[k]:>12.6f}{om[kk]:>12.6f}{d:>12.2e}")
    agree = all(abs(m[k] - om[kk]) < 1e-9 for k, kk in
                (("ACC", "acc"), ("NMI", "nmi"), ("ARI", "ari"), ("Purity", "purity")))
    print(f"    → {'口径完全一致 ✓' if agree else '存在差异 ✗'}")

    # ---- oracle 协议虚高 ----
    lab_or = detail["labels_oracle"]
    mo = evaluate(gt[mask], lab_or[mask])
    print(f"\n[6] ⚠️ 官方 oracle 协议带来的虚高（按 ACC 选轮次/初始化，用到真值）")
    print(f"    oracle 选的轮次 = 第 {detail['oracle_round']} 轮 / 共 {detail['n_rounds']} 轮")
    print(f"    无 oracle : ACC={m['ACC']:.4f}  NMI={m['NMI']:.4f}  ARI={m['ARI']:.4f}  Purity={m['Purity']:.4f}")
    print(f"    oracle 版 : ACC={mo['ACC']:.4f}  NMI={mo['NMI']:.4f}  ARI={mo['ARI']:.4f}  Purity={mo['Purity']:.4f}")
    print(f"    虚高      : ΔACC={mo['ACC'] - m['ACC']:+.4f}  ΔNMI={mo['NMI'] - m['NMI']:+.4f}")
    print(f"    → 本平台一律用「无 oracle」列，否则 OSMAGC 会凭真值挑轮次而白拿优势。")

    # ---- 落盘 ----
    res = RunResult(
        dataset=ds.name, method="OSMAGC (central, upper bound)",
        partition="none", partition_mode="central", n_clients=0,
        eval_mask="labeled", norm="osmagc-specific(L2+predata)", seed=args.seed,
        ACC=m["ACC"], NMI=m["NMI"], ARI=m["ARI"], Purity=m["Purity"],
        runtime_s=detail["runtime_s"], n_clusters=ds.n_classes,
        extra={
            "alpha": detail["alpha"], "num_pixel": detail["num_pixel"],
            "lam": detail["lam"], "gamma": detail["gamma"],
            "oracle_ACC": mo["ACC"], "oracle_round": detail["oracle_round"],
            "oracle_gain_ACC": mo["ACC"] - m["ACC"],
            "n_pixels": ds.n_pixels, "n_labeled": int(mask.sum()),
            "env": fingerprint(),
        },
    )
    append_result(res, Path(args.log))
    print(f"\n[7] 已写入 {args.log}\n    {res.line()}")
    print("=" * 96)
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
