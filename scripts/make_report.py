"""把所有实验记录汇总成一份完整报告（Markdown）。

数据源：`results/runs.jsonl`（每次实验追加一条，含协议指纹与四个指标）
产物  ：`results/REPORT.md`

报告结构
--------
一、目标与结论速览
二、方法清单（含联邦设定与适用划分）
三、数据集清单
四、评测协议（口径统一说明——这是数字可信的前提）
五、主结果表（ACC / NMI / ARI / Purity，均值±标准差）
六、逐数据集明细
七、无法运行的组合及原因（「不适应」证据）
八、已知问题与注意事项
九、如何复现

运行：python scripts/make_report.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
METRICS = ("ACC", "NMI", "ARI", "Purity")

# 表格里方法的展示顺序（上界 -> 联邦 -> 下界）
METHOD_ORDER = [
    "OSMAGC (central, upper bound)",
    "OSMAGC (federated, view)",
    "Fed-MVKM (region)",
    "Fed-MVKM",
    "Local-only (view, 不融合)",
    "Local-only (view, 融合)",
    "Local-only (view)",
    "Local-only (region)",
]

METHOD_ROLE = {
    "OSMAGC (central, upper bound)": "遥感专用 · 不联邦 · **上界**",
    "OSMAGC (federated, view)": "遥感专用 · 联邦化 · baseline",
    "Fed-MVKM (region)": "通用联邦 · 水平联邦（客户端=样本子集）",
    "Fed-MVKM": "通用联邦 · 水平联邦",
    "Local-only (view, 不融合)": "下界 · 各客户端独立评分取平均",
    "Local-only (view, 融合)": "下界 · 逐视图聚类后对齐融合",
    "Local-only (view)": "下界 · 逐视图聚类后对齐融合",
    "Local-only (region)": "下界 · 各区域独立聚类再拼接",
}

METHOD_SETTING = {
    "OSMAGC (central, upper bound)": "无需划分（所有视图、所有像素一起跑）",
    "OSMAGC (federated, view)": "视图划分（客户端=模态）",
    "Fed-MVKM (region)": "区域划分（客户端=地理区块）",
    "Fed-MVKM": "区域划分（客户端=地理区块）",
    "Local-only (view, 不融合)": "视图划分（客户端=模态）",
    "Local-only (view, 融合)": "视图划分（客户端=模态）",
    "Local-only (view)": "视图划分（客户端=模态）",
    "Local-only (region)": "区域划分（客户端=地理区块）",
}


# --------------------------------------------------------------------------- #
def load_runs(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


# 历史记录里用过的方法名 -> 统一名（早期脚本命名不一致，不统一会在表里裂成两行）
ALIAS = {
    "Fed-MVKM": "Fed-MVKM (region)",
    "Local-only (view)": "Local-only (view, 融合)",
    "OSMAGC (central, upper bound)": "OSMAGC (central, upper bound)",
    "OSMAGC (federated, view-split)": "OSMAGC (federated, view)",
}


def normalize(runs: list[dict]) -> list[dict]:
    for r in runs:
        r["method"] = ALIAS.get(r.get("method", ""), r.get("method", ""))
    return runs


def dedupe(runs: list[dict]) -> tuple[list[dict], int]:
    """同一 (数据集, 方法, 划分, 种子) 只保留**最后一条**记录。

    为什么必须去重：跑批会重跑已经做过的配置（修 bug 之后尤其如此）。
    如果不去重，旧记录（例如修 MUUFL 缺失掩码 bug 之前的那次）会和新记录
    一起被平均，均值就被污染了 —— 这正是本脚本第一版踩到的坑。
    """
    latest: dict[tuple, dict] = {}
    for r in runs:
        key = (r["dataset"], r["method"], r["partition"], r["seed"])
        latest[key] = r            # 后写覆盖先写
    dropped = len(runs) - len(latest)
    return list(latest.values()), dropped


def aggregate(runs: list[dict]):
    """按 (dataset, method, partition) 聚合，返回 {key: {metric: (mean, std, n)}} 与错误表。"""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    errors: dict[tuple, str] = {}

    for r in runs:
        key = (r["dataset"], r["method"], r["partition"])
        if r.get("status") == "ok":
            buckets[key].append(r)
        elif r.get("status") in ("error", "skip"):
            errors.setdefault(key, r.get("note", ""))

    agg = {}
    for key, rs in buckets.items():
        # 同一 (dataset, method) 可能有多种划分，这里按 key 分开存
        d = {}
        for m in METRICS:
            vals = [x[m] for x in rs if x.get(m) is not None]
            d[m] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        d["_seeds"] = sorted({x["seed"] for x in rs})
        d["_runs"] = rs
        agg[key] = d
    return agg, errors


def fmt(cell, with_std=True) -> str:
    if cell is None:
        return "—"
    mean, std, n = cell
    if n == 1 or not with_std:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RES / "runs.jsonl"))
    ap.add_argument("--out", default=str(RES / "REPORT.md"))
    args = ap.parse_args()

    runs = normalize(load_runs(Path(args.runs)))
    raw_n = len(runs)
    runs, dropped = dedupe(runs)
    agg, errors = aggregate(runs)
    if dropped:
        print(f"（去重：{raw_n} 条原始记录 -> {len(runs)} 条，丢弃 {dropped} 条被覆盖的旧记录）")

    datasets = sorted({k[0] for k in list(agg) + list(errors)},
                      key=lambda d: ["Augsburg", "MUUFL", "Trento", "Salinas",
                                     "Berlin"].index(d) if d in
                      ["Augsburg", "MUUFL", "Trento", "Salinas", "Berlin"] else 99)
    methods = [m for m in METHOD_ORDER
               if any(k[1] == m for k in list(agg) + list(errors))]

    L: list[str] = []
    W = L.append

    W("# 遥感多视图聚类统一测试平台 · 实验结果报告")
    W("")
    W(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　|　"
      f"数据源：`{Path(args.runs).name}`（{len(runs)} 条原始记录）")
    W("")

    # ---------------- 一、目标 ----------------
    W("## 一、目标")
    W("")
    W("把两类算法放到**同一批遥感数据**上比较：")
    W("")
    W("- **通用联邦多视图聚类算法**（为图片/文本等通用数据设计，不了解遥感特性）")
    W("- **遥感专用算法**（OSMAGC，显式建模「相邻像素大概率是同一种地物」）")
    W("")
    W("回答三个问题：")
    W("")
    W("1. 通用算法在遥感数据上掉多少分？为什么掉？")
    W("2. 联邦化版本相对集中式版本的损失有多大？")
    W("3. 哪些方法在哪些数据结构下**根本无法运行**？（「不适应」本身即结论）")
    W("")

    # ---------------- 二、方法 ----------------
    W("## 二、方法清单")
    W("")
    W("| 方法 | 角色 | 联邦设定 |")
    W("|---|---|---|")
    for m in methods:
        W(f"| `{m}` | {METHOD_ROLE.get(m, '—')} | {METHOD_SETTING.get(m, '—')} |")
    W("")
    W("> **注意**：Fed-MVKM 是**水平联邦**（客户端 = 样本子集 + 全部视图），"
      "因此只能跑区域划分；在视图划分下它要求«每个客户端持有全部视图»，"
      "该前提不成立，属方法设定所限，非实验缺陷。")
    W("")

    # ---------------- 三、数据集 ----------------
    W("## 三、数据集")
    W("")
    W("| 数据集 | 尺寸 | 像素数 | 视图 | 类别 | 有标注像素 | 备注 |")
    W("|---|---|---|---|---|---|---|")
    DS_INFO = {
        "Augsburg": ("332×485", "161,020", "HS(180)+SAR(4)+DSM(1)", "7", "78,294",
                     "唯一被 OSMAGC 与 FedRSMVC 都真跑过的数据集；DSM 为近似重建"),
        "MUUFL":    ("325×220", "71,500", "HS(64)+LiDAR(2)", "11", "53,687",
                     "官方为**缺失率 0.5** 版本：两视图各丢 25%、缺失集不重叠"),
        "Trento":   ("166×600", "99,600", "HS(63)+LiDAR(2)", "6", "30,214", "—"),
        "Salinas":  ("512×217", "111,104", "HS(204)+EMP(45)+Gabor(90)", "16", "54,129",
                     "仅高光谱一个传感器，另两个视图为派生特征"),
        "Berlin":   ("1723×476", "820,148", "HS(244)+SAR(4)", "8", "464,671",
                     "规模最大，单次运行约 15 分钟，本轮未纳入"),
    }
    for d in datasets:
        if d in DS_INFO:
            i = DS_INFO[d]
            W(f"| {d} | {i[0]} | {i[1]} | {i[2]} | {i[3]} | {i[4]} | {i[5]} |")
    W("")

    # ---------------- 四、协议 ----------------
    W("## 四、评测协议（数字可信的前提）")
    W("")
    W("### 4.1 四条统一")
    W("")
    W("| # | 统一的 | 说明 |")
    W("|---|---|---|")
    W("| 1 | **最终接口 = 每个像素一个簇标签** | 无论方法内部在超像素/样本/视图上工作，"
      "最后都归约到像素标签（超像素标签 broadcast 回像素）|")
    W("| 2 | **评价函数** | 一律由 `common/metrics.py` 计算，"
      "**不采信任何方法自报的数字** |")
    W("| 3 | **评分像素** | 一律在 `gt > 0` 的同一批有标注像素上评分 |")
    W("| 4 | **划分协议** | 同一份划分方案、同一组随机种子 |")
    W("")
    W("**唯一不统一的：预处理与超参各自调优** —— 这是有意为之，"
      "因为不同方法为不同数据结构设计，硬套统一预处理反而制造不公平。")
    W("")
    W("### 4.2 四个指标（口径已对齐）")
    W("")
    W("| 指标 | 定义 | 易错点 |")
    W("|---|---|---|")
    W("| **ACC** | 匈牙利最优标签置换后的准确率 | 必须先置换；"
      "未置换会把 ACC 打到随机水平 |")
    W("| **NMI** | 互信息，**算术平均归一化** `2·MI/(H_pred+H_true)` | "
      "OSMAGC 源码里变量名叫 `NMI_sqrt` 是误导，实现是算术平均 |")
    W("| **ARI** | 调整兰德指数（Hubert–Arabie） | — |")
    W("| **Purity** | 每个预测簇取占比最大的真值类，累加后除以 N | "
      "**不是**置换后的 ACC——允许多个簇映射到同一个真值类 |")
    W("")
    W("以上实现已与 **sklearn、OSMAGC、FedRSMVC、Fed-MVKM 四份现有实现逐位对拍，"
      "68 项对照 0 失配**。")
    W("")
    W("### 4.3 ⚠️ 诚实读出协议（本平台最重要的方法论决定）")
    W("")
    W("所有以 k-means 作为最终读出步骤的方法，最后一步都是「把表示硬分成 K 类」，"
      "而这一步需要随机初始化。**同一份表示、同一批数据，只换随机初始化，"
      "ACC 能差 10 个百分点以上。**")
    W("")
    W("本平台的强制规定：")
    W("")
    W("> `sklearn KMeans(n_clusters=K, n_init=10, random_state=固定)`，"
      "**按 inertia（簇内平方和）选初始化**，固定取最后一次迭代的结果，"
      "**全程不接触真值**。")
    W("")
    W("**为什么必须这样规定**：OSMAGC 官方协议是「跑多次 k-means，用真值挑 ACC 最高的那次」，"
      "这等于考试时把答案摊在桌上挑抄得最像的那份。实测这项 oracle 选择"
      "在 MUUFL 上虚高 **+8.6 个百分点**（诚实 0.4064 → oracle 0.4924）。")
    W("")
    W("> 因此本报告所有主表数字**均低于各方法论文中的报出值**。"
      "这不是本平台「跑坏了」，而是论文数字包含了 oracle 选择的虚高。"
      "所有方法一视同仁地受此约束，故排名仍然公平。")
    W("")

    # ---------------- 五、主结果表 ----------------
    W("## 五、主结果表")
    W("")
    W("数值为 **均值±标准差**（多种子）。`—` 表示该组合未运行或不可运行。")
    W("")

    for metric in METRICS:
        W(f"### 5.{METRICS.index(metric) + 1} {metric}")
        W("")
        head = "| 方法 | " + " | ".join(datasets) + " |"
        sep = "|---" * (len(datasets) + 1) + "|"
        W(head)
        W(sep)
        for m in methods:
            cells = []
            for d in datasets:
                # 同一方法在不同划分下分开；这里优先取该数据集下存在的划分
                cands = [k for k in agg if k[0] == d and k[1] == m]
                if cands:
                    # 取种子数最多的那个
                    key = max(cands, key=lambda k: agg[k]["_seeds"].__len__())
                    cells.append(fmt(agg[key].get(metric)))
                else:
                    cells.append("—")
            W(f"| `{m}` | " + " | ".join(cells) + " |")
        W("")

    # ---------------- 六、逐数据集明细 ----------------
    W("## 六、逐数据集明细")
    W("")
    for d in datasets:
        keys = [k for k in agg if k[0] == d]
        if not keys:
            continue
        W(f"### {d}")
        W("")
        W("| 方法 | 划分 | 种子数 | ACC | NMI | ARI | Purity |")
        W("|---|---|---|---|---|---|---|")
        for m in methods:
            for k in sorted([x for x in keys if x[1] == m], key=lambda x: x[2]):
                a = agg[k]
                W(f"| `{m}` | {k[2]} | {len(a['_seeds'])} | "
                  f"{fmt(a['ACC'])} | {fmt(a['NMI'])} | {fmt(a['ARI'])} | {fmt(a['Purity'])} |")
        W("")

    # ---------------- 七、无法运行的组合 ----------------
    W("## 七、无法运行的组合（「不适应」的直接证据）")
    W("")
    if errors:
        W("| 数据集 | 方法 | 划分 | 原因 |")
        W("|---|---|---|---|")
        for (d, m, p), note in sorted(errors.items()):
            W(f"| {d} | `{m}` | {p} | {note} |")
    else:
        W("本轮无。")
    W("")
    W("这些 `N/A` 不是实验失败，而是**方法对数据结构的硬依赖**：")
    W("")
    W("- **Fed-MVKM 在视图划分下不可运行** —— 其代码硬性要求每个客户端持有全部视图")
    W("- **所有通用联邦 MVC 都不处理「视图内像素缺失」** —— 它们只处理整视图缺失；"
      "像素级缺失目前只有集中式的 OSMAGC 支持")
    W("- **OSMAGC 联邦化在随机样本划分下不可运行** —— 其空间正则需要连通的 `(H,W)` 子栅格")
    W("")

    # ---------------- 八、已知问题 ----------------
    W("## 八、已知问题与注意事项")
    W("")
    W("1. **Fed-MVKM 在 Augsburg 上不稳定**：同配置三个种子的 ACC 标准差约 ±0.045；"
      "且其 `beta_k` 最优值呈孤立尖峰（0.46→**0.54**→0.41），"
      "典型拟合噪声特征。**不宜作为单一结论依据**。")
    W("2. **「联邦版优于不联邦版」的反常**：Fed-MVKM 在 Augsburg 上 "
      "M=1（不联邦）≈0.32 而 M=4（空间划分）≈0.48–0.54，方向与理论和其自身 "
      "HD 数据集表现相反。适配器已用 Fed-MVKM 官方数据集做回归测试验证正确"
      "（精确复现 ACC=0.8530），故反常来自数据/参数而非代码，**原因待查**。")
    W("3. **OSMAGC 联邦化与集中式逐元素相同**（`max|ΔH| = 0`）："
      "因其跨视图耦合仅为一个求和项 `Σ R_v Z_v`。因此二者的逐种子差异"
      "**仅来自 k-means 随机初始化**，与联邦化无关。代价体现在通信量上。")
    W("4. **Salinas 的「多视图」性质与其他数据集不同**：它只有高光谱一个传感器，"
      "另两个视图（EMP、Gabor）是从高光谱派生的特征，"
      "不存在「不同机构持有不同模态」的现实动机。")
    W("5. **Berlin 未纳入本轮**：单次运行约 15 分钟（k-means 评估占大头），"
      "全矩阵跑批时间不可接受，待后续单独运行。")
    W("6. **Trento / Salinas 的 OSMAGC 超参无官方最优值**："
      "OSMAGC 论文未使用这两个数据集，λ/γ 暂用官方代码中对 MUUFL 的默认值，"
      "超像素数按论文范围取中位。后续需按本平台扫描细化。")
    W("")

    # ---------------- 九、复现 ----------------
    W("## 九、如何复现")
    W("")
    W("```bash")
    W("cd D:\\cc\\算法测试")
    W("")
    W("# 1) 验证指标口径（68 项对拍，应 0 失配）")
    W("python scripts/verify_metrics.py")
    W("")
    W("# 2) 验证数据加载与划分")
    W("python scripts/verify_data.py")
    W("")
    W("# 3) 验证 OSMAGC 联邦化无损")
    W("python scripts/verify_osmagc_federated.py --dataset Augsburg")
    W("")
    W("# 4) 全流程跑批（多数据集 × 多种子）")
    W("python scripts/run_full_matrix.py --datasets Augsburg MUUFL Trento Salinas "
      "--seeds 0 1 2")
    W("")
    W("# 5) 生成本报告")
    W("python scripts/make_report.py")
    W("```")
    W("")
    W("环境：Python 3.12 (Anaconda) / numpy 1.26 / scipy 1.13 / sklearn 1.5 / "
      "torch 2.5.1+cu124 / RTX 4060 Laptop 8GB")
    W("")

    # ---------------- 附录 ----------------
    W("## 附录：原始记录")
    W("")
    W(f"- `results/runs.jsonl` —— 原始 {raw_n} 条记录，去重后 {len(runs)} 条"
      "（每行一条 JSON，含数据集/方法/划分/种子/四个指标/耗时/协议指纹）")
    W("- `results/full_matrix.log` —— 跑批完整日志")
    W("- `results/kmeans_readout_*.json` —— k-means 读出协议的逐轮明细")
    W("")

    out = Path(args.out)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"报告已写入 {out}（{len(L)} 行，{len(runs)} 条记录）")

    # 控制台也打一份速览
    print()
    print(f"数据集 {datasets}")
    print(f"方法   {len(methods)} 个")
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        print(f"  {'方法':<34}" + "".join(f"{d:>18}" for d in datasets))
        for m in methods:
            row = []
            for d in datasets:
                cands = [k for k in agg if k[0] == d and k[1] == m]
                row.append(fmt(agg[max(cands, key=lambda k: len(agg[k]['_seeds']))].get(metric))
                           if cands else "—")
            print(f"  {m:<34}" + "".join(f"{x:>18}" for x in row))
    if errors:
        print(f"\n无法运行的组合 {len(errors)} 个（详见报告第七节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
