# 对 `D:\cc\OSMAGC` 的改动记录

`D:\cc\OSMAGC` **不在任何版本控制之下**，所以本文件是该项目的改动台账。
原始文件已就地备份为 `*.orig-20261007`。

改动时间：2026-10-07。改动目的：降低单次运行的内存峰值，以便并行跑调参。

---

## 1. `osmagc/core/updates.py` — `COL_CHUNK` 200_000 → 25_000

```python
COL_CHUNK = 25_000      # 原 200_000
```

**原因**：`200_000` 这个阈值是为 Berlin（N=820,148）量身定的。而本平台的数据集
N 全部小于 200,000（Salinas 111,104 / MUUFL 71,500 / Trento 99,600 / Augsburg 161,020），
于是 `F.shape[1] <= chunk` 恒成立，**分块逻辑一次都不会触发**，`updateZv` 与
`eproj_simplex_batch` 全程走全尺寸路径。调到 25,000 后峰值 **3.6 GB → 2.61 GB**。

⚠️ **这个改动会改变结果**，不是纯性能优化：实测同一配置（Salinas λ=1 γ=1 sp=[200,200,200]
seed=0 n_init=3）ACC 从 **0.5775 变成 0.5728**。原因是分块会改变 BLAS 的 kernel
选择与浮点舍入顺序，进而改变迭代轨迹。

因此 `COL_CHUNK` 与本平台的**线程数、n_init 同级，属于「协议指纹」的一部分**，
已写入 `scripts/tune_osmagc.py` 每条 journal 记录的 `col_chunk` 字段，
并纳入断点续跑的 `done_keys` 去重键（换分块大小不会错误跳过旧记录）。

## 2. `osmagc/core/init.py` — `initZ` 合并两段循环

原先「先攒齐 `Z_dense`（3×178 MB）再逐个散回 `Z_new`（3×178 MB）」，合并成
单循环、算完一个视图立刻散回并释放。

**数值逐位不变**（实测 ACC/NMI/ARI/Purity 四项全等），实际内存收益仅 **~90 MB**——
远小于预估，因为峰值不在这一层。

## 3. `osmagc/core/model.py` — `opti` 中 `updateZv` 前先 `del Z`

`updateZv` 会新建 3 个 `(Mv,N)`，不先释放旧 `Z` 的话新旧同时驻留（2×533 MB）。
旧 `Z` 在其上两步 `updateA`/`updateRv` 之后已无人引用。

**数值逐位不变**（同上实测），实际收益同样很小。

---

## 没做成的事：`--jobs 2` 仍然不可行

第 2、3 项加起来只把峰值从 2.61 GB 降到 **2.52 GB**。真正的峰值在
`eproj_simplex_batch` 的 Newton while 循环内部——它同时驻留 6~7 个 `(d,N)`
float64 数组（Salinas 下每个 178 MB）**≈ 1.25 GB**，这一层没动。

- `--jobs 2` 需要 2 × 2.52 = 5.04 GB 常驻
- 本机可用内存 6.3 GB（Cursor 常占 1.26 GB）
- 跑起来只剩 ~1.7 GB → **Claude Code 的后台任务内存看门狗在 ~2 GB 时直接杀任务**
  （2026-10-07 实际发生：已完成 30/35 组的任务被 kill，靠 journal 断点续跑保住）

**结论：本平台短期内一律 `--jobs 1`。** 要并行得重写 eproj 核心循环（预估最多再省
~500 MB，仍未必够）或先关掉 Cursor 腾出 1.3 GB。

## 回滚方法

```bash
cd "D:/cc/OSMAGC"
cp osmagc/core/updates.py.orig-20261007 osmagc/core/updates.py
cp osmagc/core/init.py.orig-20261007    osmagc/core/init.py
# model.py 无备份：改动只有一行，即删除 `del Z`（位于 opti 内 updateZv 调用之前）
```

⚠️ 回滚后**必须清空 journal 重跑**——`col_chunk` 变了，旧记录与新记录不可混用。
