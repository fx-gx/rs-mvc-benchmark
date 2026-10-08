"""OSMAGC 逐数据集调参 —— 用于排除「超参没调」这一混淆因素。

背景
----
OSMAGC 论文只给了 MUUFL 的官方超参（λ=γ=1e-3），其余数据集需自行调优
（论文 4.1 节说明了这一点，但没给值）。项目实测发现最优 γ 与官方默认值
差 2~4 个数量级，且 **不调参会让消融实验反号**。

因此「Salinas 上 OSMAGC 输给下界」这一结果，必须先排除「超参没调」的解释。

调参口径（必须与所有方法一视同仁）
----------------------------------
- 调参用的就是评测指标本身（聚类没有验证集，这是该领域的通行做法）
- **所有方法都享有同等的调参机会**，报告里要写明各方法各自的调参网格与预算
- 读出仍走诚实协议（sklearn KMeans n_init=10，按 inertia 选初始化，取最后一轮）

两阶段
------
Stage A：λ × γ 网格（默认 5×6 = 30 组）
Stage B：在 Stage A 最优的 λ/γ 上扫超像素数

运行：
    python scripts/tune_osmagc.py --dataset Salinas --jobs 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RES = ROOT / "results"

LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
GAMMA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

# 各数据集超像素候选（Stage B）
SP_CANDIDATES = {
    "Salinas": [(100, 100, 100), (150, 150, 150), (200, 200, 200),
                (300, 300, 300), (150, 200, 300)],
    "Trento": [(50, 50), (100, 100), (200, 200), (300, 300)],
    "Augsburg": [(100, 100, 100), (200, 200, 200), (300, 300, 300), (150, 200, 300)],
    "MUUFL": [(50, 50), (50, 100), (100, 100), (100, 300), (200, 200)],
}


def _run_one(task: dict) -> dict:
    """在工作进程里跑一组配置。只接收数据集名，由本进程自行加载（避免大对象跨进程传输）。"""
    from common.data import load_dataset
    from common.metrics import evaluate
    from methods.osmagc_central import DEFAULT_PARAMS, run_osmagc_central
    from osmagc.core.updates import COL_CHUNK

    ds = load_dataset(task["dataset"])
    base = DEFAULT_PARAMS[ds.name]
    p = replace(base, lam=task["lam"], gamma=task["gamma"],
                num_pixel=tuple(task["num_pixel"]))

    t0 = time.time()
    try:
        labels, det = run_osmagc_central(ds, params=p, seed=task["seed"],
                                         n_init=task.get("n_init", 10))
        gt = ds.flatten_gt("F")
        mask = gt > 0
        m = evaluate(gt[mask], labels[mask])
        return {**task, "status": "ok", "runtime_s": time.time() - t0,
                # 协议指纹：分块大小会改变浮点舍入、进而改变结果（实测 ACC 差 0.5 点），
                # 所以它和 n_init 一样必须逐条记录，否则半年后没人知道数字怎么来的。
                "col_chunk": int(COL_CHUNK),
                **{k: float(v) for k, v in m.items()}}
    except Exception as exc:                                  # noqa: BLE001
        return {**task, "status": "error", "runtime_s": time.time() - t0,
                "note": f"{type(exc).__name__}: {exc}"}


def run_batch(tasks: list[dict], jobs: int, journal: Path) -> list[dict]:
    """跑一批配置。

    **每完成一组就立刻追加到 `journal`（JSONL）**，中途被杀也不会整轮白跑。
    这条是上一次的教训：原版等到全部跑完才写结果，任务一被系统杀掉就全丢了。
    jobs=1 时走纯串行路径（内存最省）；jobs>1 时用 joblib 并行，
    但仍是"跑完一组记一组"（并行时按完成顺序写）。
    """
    results: list[dict] = []
    done: dict[tuple, dict] = {}

    def _emit(r: dict) -> None:
        results.append(r)
        with journal.open("a", encoding="utf-8") as f:
            import json as _json
            f.write(_json.dumps(r, ensure_ascii=False) + "\n")

    if jobs <= 1:
        for i, t in enumerate(tasks, 1):
            r = _run_one(t)
            _emit(r)
            tag = "OK " if r["status"] == "ok" else "ERR"
            acc = f"{r['ACC']:.4f}" if r["status"] == "ok" else r.get("note", "")[:40]
            print(f"    [{i:>2}/{len(tasks)}] {tag} "
                  f"λ={t['lam']:g} γ={t['gamma']:g} sp={t['num_pixel']} -> ACC={acc} "
                  f"({r['runtime_s']:.0f}s)", flush=True)
    else:
        from joblib import Parallel, delayed
        for i, r in enumerate(
                Parallel(n_jobs=jobs, return_as="generator_unordered")(
                    delayed(_run_one)(t) for t in tasks), 1):
            _emit(r)
            print(f"    [{i:>2}/{len(tasks)}] 完成", flush=True)
            done[(r["lam"], r["gamma"], tuple(r["num_pixel"]))] = r

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Salinas")
    ap.add_argument("--jobs", type=int, default=1,
                    help="1=串行（内存最省，推荐）；>1 用 joblib 并行")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda-grid", nargs="+", type=float, default=LAMBDA_GRID)
    ap.add_argument("--gamma-grid", nargs="+", type=float, default=GAMMA_GRID)
    ap.add_argument("--skip-stage-b", action="store_true")
    ap.add_argument("--n-init", type=int, default=3,
                    help="扫描阶段（Stage A/B）k-means 的初始化次数。"
                         "所有配置统一用同一值，故配置间比较仍公平；"
                         "**注意它同时作用于 Stage A 和 Stage B**")
    ap.add_argument("--confirm-n-init", type=int, default=10,
                    help="Stage C：用该 n_init 复核扫描出的最优配置（默认 10，"
                         "即正式读出协议），报出去的最终数字以它为准。"
                         "设为与 --n-init 相同则跳过复核")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略已有 journal，从头跑")
    args = ap.parse_args()

    from common.data import load_dataset
    from methods.osmagc_central import DEFAULT_PARAMS
    from osmagc.core.updates import COL_CHUNK

    ds = load_dataset(args.dataset)
    base = DEFAULT_PARAMS[ds.name]
    print("=" * 96)
    print(f"OSMAGC 调参 | {ds.name} | seed={args.seed} | 并行 {args.jobs}")
    print("=" * 96)
    print(f"  {ds.summary()}")
    print(f"  基线配置：num_pixel={list(base.num_pixel)} λ={base.lam:g} γ={base.gamma:g}")

    # ---------------- 断点续跑：跳过 journal 里已完成的配置 ----------------
    journal = RES / f"tuning_journal_{ds.name}.jsonl"
    done_keys: set[tuple] = set()
    prev: list[dict] = []
    if journal.exists() and not args.no_resume:
        for ln in journal.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            prev.append(r)
            if r.get("status") == "ok":
                # col_chunk 和 n_init 都必须进 key：两者都会改变结果。
                # - 换 col_chunk：ACC 差 0.5 点（实测）
                # - 换 n_init：不只是数值变，**配置之间的排名都会翻**（实测：
                #   Salinas 扫描第 8 名在 n_init=10 下成了冠军，扫描冠军掉到第 3）
                # 不带上它们，会把"用别的协议算出来的记录"当成已完成而错误跳过。
                done_keys.add((r["lam"], r["gamma"], tuple(r["num_pixel"]),
                               r.get("stage"), r.get("col_chunk"), r.get("n_init")))
        print(f"\n[断点续跑] journal 已有 {len(prev)} 条，其中成功 {len(done_keys)} 条将被跳过")

    # ---------------- Stage A：λ × γ ----------------
    all_a = [
        dict(dataset=ds.name, lam=lam, gamma=gam,
             num_pixel=list(base.num_pixel), seed=args.seed, stage="A")
        for lam in args.lambda_grid for gam in args.gamma_grid
    ]
    tasks = [t for t in all_a
             if (t["lam"], t["gamma"], tuple(t["num_pixel"]), "A",
                 int(COL_CHUNK), args.n_init) not in done_keys]
    for t in tasks:
        t["n_init"] = args.n_init
    est = len(tasks) * (55 + 9 * args.n_init) / 60
    print(f"\n[Stage A] λ × γ 网格：共 {len(all_a)} 组，"
          f"本次需跑 {len(tasks)} 组（n_init={args.n_init}，约 {est:.0f} 分钟）")

    res_a = list(prev) if done_keys else []
    if tasks:
        t0 = time.time()
        res_a = res_a + run_batch(tasks, args.jobs, journal)
        print(f"  完成，耗时 {time.time() - t0:.0f}s  （明细已逐组追加到 journal）")

    ok_a = [r for r in res_a if r["status"] == "ok"]
    ok_a.sort(key=lambda r: -r["ACC"])
    print(f"\n  Top 10（按 ACC）：")
    print(f"    {'λ':>10}{'γ':>10}{'ACC':>9}{'NMI':>9}{'ARI':>9}{'Purity':>9}")
    for r in ok_a[:10]:
        print(f"    {r['lam']:>10g}{r['gamma']:>10g}{r['ACC']:>9.4f}{r['NMI']:>9.4f}"
              f"{r['ARI']:>9.4f}{r['Purity']:>9.4f}")

    # 与基线对比
    base_res = next((r for r in res_a
                     if r["lam"] == base.lam and r["gamma"] == base.gamma), None)
    if base_res and base_res["status"] == "ok":
        print(f"\n  当前基线 (λ={base.lam:g}, γ={base.gamma:g})：ACC={base_res['ACC']:.4f} "
              f"ARI={base_res['ARI']:.4f}")
    if ok_a:
        b = ok_a[0]
        print(f"  最优 (λ={b['lam']:g}, γ={b['gamma']:g})：ACC={b['ACC']:.4f} "
              f"ARI={b['ARI']:.4f}")
        print(f"  → ACC 提升 {b['ACC'] - (base_res['ACC'] if base_res else 0):+.4f}")

    # ---------------- Stage B：超像素数 ----------------
    best_lam = ok_a[0]["lam"] if ok_a else base.lam
    best_gam = ok_a[0]["gamma"] if ok_a else base.gamma
    res_b = []
    if not args.skip_stage_b:
        cands = SP_CANDIDATES.get(ds.name, [tuple(base.num_pixel)])
        # ⚠️ 必须显式带上 n_init：`_run_one` 是 `task.get("n_init", 10)`，
        # 漏掉这个键会让 Stage B 静默用 10，与 Stage A 的 --n-init 不一致，
        # 使 λ×γ 排名表混用两种读出协议（2026-10-07 实测踩过）。
        tasks_b = [
            dict(dataset=ds.name, lam=best_lam, gamma=best_gam,
                 num_pixel=list(sp), seed=args.seed, stage="B",
                 n_init=args.n_init)
            for sp in cands
        ]
        print(f"\n[Stage B] 超像素数扫描（固定 λ={best_lam:g}, γ={best_gam:g}）："
              f"{len(tasks_b)} 组")
        t0 = time.time()
        res_b = run_batch(tasks_b, args.jobs, RES / f"tuning_journal_{ds.name}.jsonl")
        print(f"  完成，耗时 {time.time() - t0:.0f}s")
        ok_b = [r for r in res_b if r["status"] == "ok"]
        ok_b.sort(key=lambda r: -r["ACC"])
        print(f"\n    {'超像素':>22}{'ACC':>9}{'NMI':>9}{'ARI':>9}{'Purity':>9}")
        for r in ok_b:
            print(f"    {str(r['num_pixel']):>22}{r['ACC']:>9.4f}{r['NMI']:>9.4f}"
                  f"{r['ARI']:>9.4f}{r['Purity']:>9.4f}")

    # ---------------- Stage C：最优配置用正式 n_init 复核 ----------------
    # 扫描阶段用低 n_init 提速（配置间比较仍公平），但**报出去的最终数字**必须回到
    # 正式读出协议，否则扫描出的"最优"和复核值对不上。此前这一步只写在 help 里、
    # main() 中并未实现，导致 Stage B 意外顶替了复核角色。
    res_c = []
    scan_ok = [r for r in (res_a + res_b) if r["status"] == "ok"]
    if scan_ok and args.confirm_n_init != args.n_init:
        scan_ok.sort(key=lambda r: -r["ACC"])
        b = scan_ok[0]
        task_c = dict(dataset=ds.name, lam=b["lam"], gamma=b["gamma"],
                      num_pixel=list(b["num_pixel"]), seed=args.seed,
                      stage="C", n_init=args.confirm_n_init)
        print(f"\n[Stage C] 用 n_init={args.confirm_n_init} 复核扫描最优 "
              f"(λ={b['lam']:g}, γ={b['gamma']:g}, sp={b['num_pixel']})")
        t0 = time.time()
        res_c = run_batch([task_c], args.jobs,
                          RES / f"tuning_journal_{ds.name}.jsonl")
        print(f"  完成，耗时 {time.time() - t0:.0f}s")
        for r in res_c:
            if r["status"] == "ok":
                print(f"    扫描值 ACC={b['ACC']:.4f}  →  复核值 ACC={r['ACC']:.4f}"
                      f"  (差 {r['ACC'] - b['ACC']:+.4f})")

    # ---------------- 落盘 ----------------
    all_res = [r for r in (res_a + res_b + res_c) if r["status"] == "ok"]
    # 最终值以复核（stage C）为准，没有复核则退回扫描值
    all_res.sort(key=lambda r: (r.get("stage") != "C", -r["ACC"]))
    out = RES / f"tuning_osmagc_{ds.name}.json"
    out.write_text(json.dumps({
        "dataset": ds.name, "seed": args.seed,
        "baseline": {"num_pixel": list(base.num_pixel),
                     "lam": base.lam, "gamma": base.gamma},
        "n_configs": len(all_res),
        "top20": all_res[:20],
        "all": all_res,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out}")

    if all_res:
        t = all_res[0]
        print("\n" + "=" * 96)
        print(f"调参后最优：num_pixel={t['num_pixel']} λ={t['lam']:g} γ={t['gamma']:g}")
        print(f"  ACC={t['ACC']:.4f}  NMI={t['NMI']:.4f}  ARI={t['ARI']:.4f}  "
              f"Purity={t['Purity']:.4f}")
        print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
