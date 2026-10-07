"""验证数据加载层与划分层。

检查项
------
1. 各数据集能否正确加载、形状是否自洽
2. **展平顺序自洽性**：Fortran 序展平后 reshape 回 (H,W,B) 必须逐元素相同
3. **坐标一致性**：`coords()` 与 `flatten_gt()` 在同一像素上必须对应同一个 (row, col)
4. 划分是否严格「恰好一次」覆盖全部像素（分区不重叠、不遗漏）
5. 2x2 网格划分的逐客户端类别计数，必须与我早先用 scipy 直接切 2D 真值算出的
   结果完全一致（交叉验证，而不是自证）

运行：python scripts/verify_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import partition as P                    # noqa: E402
from common.data import SPECS, load_dataset          # noqa: E402

OK, BAD = "  [OK]  ", "  [BAD] "
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print((OK if cond else BAD) + msg)
    if not cond:
        fails.append(msg)


# --------------------------------------------------------------------------- #
print("=" * 96)
print("1. 加载全部数据集")
print("=" * 96)
datasets = {}
for name in SPECS:
    try:
        ds = load_dataset(name)
        datasets[name] = ds
        print(f"  {ds.summary()}")
        print(f"        视图维度 {ds.view_dims()}  备注: {ds.note}")
    except Exception as exc:                          # noqa: BLE001
        print(f"  {BAD}{name} 加载失败：{type(exc).__name__}: {exc}")

print()
ds = datasets["Augsburg"]

# --------------------------------------------------------------------------- #
print("=" * 96)
print("2. 展平顺序自洽性（Fortran 序）")
print("=" * 96)
for v in range(ds.n_views):
    c = ds.cubes[v]
    H, W, B = c.shape
    back = ds.flatten_view(v, "F").reshape(H, W, B, order="F")
    check(np.array_equal(back, c.astype(back.dtype)),
          f"视图 {ds.view_names[v]}: F 序展平 -> reshape 还原，逐元素相同")

# --------------------------------------------------------------------------- #
print()
print("=" * 96)
print("3. 坐标与真值的一致性")
print("=" * 96)
gt_f = ds.flatten_gt("F")
coords = ds.coords("F")
check(coords.shape == (ds.n_pixels, 2), f"coords 形状 = {coords.shape}")
check(gt_f.shape == (ds.n_pixels,), f"flatten_gt 形状 = {gt_f.shape}")

# 抽查 5 个像素：用 (row, col) 回查 2D 真值，必须与展平结果相同
rng = np.random.default_rng(0)
sample = rng.choice(ds.n_pixels, size=5, replace=False)
ok = True
for i in sample:
    r, c = coords[i]
    if ds.gt[r, c] != gt_f[i]:
        ok = False
        print(f"       像素 {i}: coords=({r},{c}) gt2d={ds.gt[r, c]} 但 flatten_gt={gt_f[i]}")
check(ok, "随机 5 像素：(row,col) 回查 2D 真值 == 展平真值")

# --------------------------------------------------------------------------- #
print()
print("=" * 96)
print("4. 两种范式的划分")
print("=" * 96)

pv = P.view_partition(ds)
print("\n--- 范式 B：视图划分 ---")
print(pv.describe())

pg = P.region_grid_partition(ds, rows=2, cols=2)
print("\n--- 范式 A：2x2 网格 ---")
print(pg.describe())

ps = P.region_stripe_partition(ds, n=4, axis="col")
print("\n--- 范式 A：4 竖条带 ---")
print(ps.describe())

pr = P.random_partition(ds, m=4, seed=0)
print("\n--- 对照：随机样本划分（打散栅格）---")
print(pr.describe())

# --------------------------------------------------------------------------- #
print()
print("=" * 96)
print("5. 覆盖性：每个像素恰好属于一个客户端")
print("=" * 96)
for part in (pg, ps, pr):
    seen = np.zeros(ds.n_pixels, dtype=np.int32)
    for c in part.clients:
        seen[c.sample_idx] += 1
    check(bool((seen == 1).all()),
          f"{part.shape}: 全部像素恰好被覆盖一次（重复 {int((seen > 1).sum())}，遗漏 {int((seen == 0).sum())}）")

# --------------------------------------------------------------------------- #
print()
print("=" * 96)
print("6. 与独立实现交叉验证：2x2 网格的逐客户端类别计数")
print("=" * 96)
# 独立实现：直接切 (H,W) 的 2D 真值，不走本次的展平/划分代码
raw = sio.loadmat(SPECS["Augsburg"].gt.path)["augsburg_gt"].astype(int)
H, W = raw.shape
h2, w2 = H // 2, W // 2
ref_blocks = [
    raw[:h2, :w2], raw[:h2, w2:],
    raw[h2:, :w2], raw[h2:, w2:],
]
ref_counts = [np.bincount(b[b > 0], minlength=8)[1:8].tolist() for b in ref_blocks]
our_counts = [c.meta["class_counts"].astype(int).tolist() for c in pg.clients]

for k in range(4):
    same = ref_counts[k] == our_counts[k]
    check(same, f"客户端 {k}: 独立实现 {ref_counts[k]}  vs  本模块 {our_counts[k]}")

# --------------------------------------------------------------------------- #
print()
print("=" * 96)
print(f"结论：{'全部通过' if not fails else str(len(fails)) + ' 项失败'}")
for f in fails:
    print("  - " + f)
print("=" * 96)
raise SystemExit(1 if fails else 0)
