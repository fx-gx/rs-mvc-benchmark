"""验证 OSMAGC 的联邦化是【无损】的，并量化其通信代价。

核心断言
--------
由于集中式里唯一的跨视图耦合是一个求和项 `Σ_v R_v·Z_v`，把每个客户端
本地的 `R_v·Z_v` 加起来就能精确重建同一轮 H。因此联邦版的 H 序列应当与
集中式**逐元素相同**（初始 H 是确定性的，见 `osmagc/core/init.py::initH`）。

如果这个断言成立，就得到一个干净的结论：
    **OSMAGC 的联邦化精度损失 = 0；真实代价在通信量上。**

运行：python scripts/verify_osmagc_federated.py --dataset MUUFL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "OSMAGC"))

from common.data import load_dataset                       # noqa: E402
from common.metrics import evaluate                        # noqa: E402
from common.protocol import (                              # noqa: E402
    DEFAULT_LOG, RunResult, append_result,
)
from methods.osmagc_central import (                       # noqa: E402
    DEFAULT_PARAMS, _to_osmagc_dataset, run_osmagc_central,
)
from methods.osmagc_federated import run_osmagc_federated  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUUFL")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    p = DEFAULT_PARAMS[ds.name]
    print("=" * 96)
    print(f"OSMAGC 联邦化验证 | {ds.name} | 视图划分（一个客户端 = 一个模态）")
    print("=" * 96)
    print(f"  {ds.summary()}")
    print(f"  客户端数 = {ds.n_views}（{', '.join(ds.view_names)}）")

    from dataclasses import replace
    cfg = replace(p, max_iter=args.max_iter)

    # ---------------- 集中式 ----------------
    print(f"\n[1] 集中式（所有视图、所有像素放一起跑）")
    from osmagc.core.model import OSMAGCConfig, run_osmagc
    ocfg = OSMAGCConfig(
        num_pixel=list(p.num_pixel), lam=p.lam, gamma=p.gamma, dim=None,
        max_iter=args.max_iter, h_simplex=p.h_simplex,
        mapminmax_axis=p.mapminmax_axis, h_denom_v=p.h_denom_v,
        sagi_mode=p.sagi_mode, verbose=False,
    )
    res_c = run_osmagc(_to_osmagc_dataset(ds), ocfg)
    H_c = res_c.H_list
    print(f"    完成 {len(H_c)} 轮，H 形状 {H_c[-1].shape}")

    # ---------------- 联邦化 ----------------
    print(f"\n[2] 联邦化（各客户端只碰自己的视图，原始数据不出门）")
    labels_f, det_f = run_osmagc_federated(
        ds, params=cfg, seed=args.seed, verbose=args.verbose
    )
    H_f = det_f["H_list"]
    comm = det_f["comm"]
    print(f"    完成 {len(H_f)} 轮，耗时 {det_f['runtime_s']:.1f}s")
    print(f"    最终视图权重 α = {[round(a, 4) for a in det_f['alpha']]}")

    # ---------------- 无损性断言 ----------------
    print(f"\n[3] 无损性断言：联邦版的 H 序列 vs 集中式")
    worst = 0.0
    for i, (hc, hf) in enumerate(zip(H_c, H_f), 1):
        d = float(np.abs(hc - hf).max())
        worst = max(worst, d)
        if i <= 3 or i == len(H_c):
            print(f"    第 {i:2d} 轮  max|ΔH| = {d:.3e}   "
                  f"相对偏差 = {d / max(np.abs(hc).max(), 1e-12):.3e}")
        elif i == 4:
            print(f"    ...")
    identical = worst < 1e-12
    print(f"    全部 {len(H_c)} 轮的最大逐元素偏差 = {worst:.3e}")
    print(f"    → {'联邦化完全无损 ✓（H 逐元素相同）' if identical else '存在偏差 ✗'}")

    # ---------------- 精度对比 ----------------
    gt = ds.flatten_gt("F")
    mask = gt > 0
    labels_c, det_c = run_osmagc_central(ds, params=cfg, seed=args.seed)
    m_c = evaluate(gt[mask], labels_c[mask])
    m_f = evaluate(gt[mask], labels_f[mask])
    print(f"\n[4] 精度对比（同一套诚实 k-means 读出协议）")
    print(f"    {'':<12}{'ACC':>10}{'NMI':>10}{'ARI':>10}{'Purity':>10}")
    print(f"    {'集中式':<12}{m_c['ACC']:>10.4f}{m_c['NMI']:>10.4f}"
          f"{m_c['ARI']:>10.4f}{m_c['Purity']:>10.4f}")
    print(f"    {'联邦化':<12}{m_f['ACC']:>10.4f}{m_f['NMI']:>10.4f}"
          f"{m_f['ARI']:>10.4f}{m_f['Purity']:>10.4f}")
    print(f"    {'gap':<12}{m_f['ACC'] - m_c['ACC']:>+10.4f}"
          f"{m_f['NMI'] - m_c['NMI']:>+10.4f}{m_f['ARI'] - m_c['ARI']:>+10.4f}"
          f"{m_f['Purity'] - m_c['Purity']:>+10.4f}")
    print(f"    （gap 不为 0 只可能来自 k-means 随机初始化，与联邦化无关）")

    # ---------------- 通信代价 ----------------
    print(f"\n[5] 通信代价（这是联邦化真正的成本）")
    print(f"    每客户端每轮：")
    print(f"      上传  S_v = R_v·Z_v  ({comm.dim}×{comm.N:,} float64) = {comm.human(comm.up_per_client_per_round)}")
    print(f"      下发  H               ({comm.dim}×{comm.N:,} float64) = {comm.human(comm.down_per_client_per_round)}")
    print(f"    合计 {comm.rounds} 轮 × {comm.n_clients} 客户端：")
    print(f"      总上传 = {comm.human(comm.total_up)}")
    print(f"      总下发 = {comm.human(comm.total_down)}")
    print(f"      总计   = {comm.human(comm.total)}")
    print(f"    ⚠️ 通信量与像素数 N 成正比，这是 OSMAGC 联邦化的主要瓶颈。")

    # ---------------- 落盘 ----------------
    for tag, mm in (("OSMAGC (federated, view-split)", m_f),):
        append_result(RunResult(
            dataset=ds.name, method=tag, partition="view", partition_mode="view",
            n_clients=ds.n_views, eval_mask="labeled", norm="osmagc-specific",
            seed=args.seed, ACC=mm["ACC"], NMI=mm["NMI"], ARI=mm["ARI"],
            Purity=mm["Purity"], runtime_s=det_f["runtime_s"], n_clusters=ds.n_classes,
            extra={
                "alpha": det_f["alpha"], "num_pixel": det_f["num_pixel"],
                "lossless_max_dH": worst,
                "comm_total_bytes": comm.total,
                "comm_up_bytes": comm.total_up, "comm_down_bytes": comm.total_down,
                "comm_rounds": comm.rounds,
                "central_ACC": m_c["ACC"],
            },
        ), Path(args.log))
    print(f"\n[6] 已写入 {args.log}")
    print("=" * 96)
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
