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
    "AMKSC",
    "DIVIDE",
    "FedRSMVC",
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
    "AMKSC": "遥感专用 · 不联邦 · **第二个遥感方法**",
    "DIVIDE": "通用 · 对比学习 · 非遥感设计",
    "FedRSMVC": "遥感专用 · 联邦 · **用户自己的方法**",
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
    "AMKSC": "无需划分（逐像素空间块，不联邦）",
    "DIVIDE": "无需划分（集中式，不联邦）",
    "FedRSMVC": "视图划分（客户端=模态）",
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
        elif r.get("status") in ("error", "skip", "na"):
            # "na" = 方法在该数据结构/规模下**不可运行**（不是跑挂了）。
            # 例：Fed-MVKM 在视图划分下、AMKSC 在高光谱高维数据上。
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

    def _acc(method: str, dataset: str):
        """聚合结果里取某方法在某数据集上的 ACC 均值（供结论节自动生成数字）。"""
        for k, v in agg.items():
            if k[0] == dataset and k[1] == method and "ACC" in v:
                return v["ACC"][0]
        return None

    # ---------------- 结论 ----------------
    W("## 结论")
    W("")
    W("**1. 通用算法在遥感数据上掉分。** 最强通用（DIVIDE） vs 最强遥感专用（OSMAGC）：")
    W("")
    W("| 数据集 | DIVIDE | OSMAGC | 差 |")
    W("|---|---|---|---|")
    for d in ("Trento", "MUUFL", "Salinas", "Augsburg"):
        a, b = _acc("DIVIDE", d), _acc("OSMAGC (central, upper bound)", d)
        if a is not None and b is not None:
            W(f"| {d} | {a:.4f} | {b:.4f} | {a-b:+.4f} |")
    W("")
    W("3/4 数据集上遥感专用方法胜。Augsburg 打平（差 0.007 < DIVIDE 自身种子波动 0.0175）。")
    W("")
    W("⚠️ **「为什么掉」本平台未做归因实验**，以下三条是待检验假说，**不要当已证结论**：")
    W("① 缺少空间邻域建模（遥感地物成片连续，通用方法把像素当独立样本）；"
      "② 不处理像素级缺失（MUUFL 有 25%×25% 像素级缺失）；"
      "③ 高光谱高维特征不适配（Augsburg 180 / Salinas 204 波段）。")
    W("")
    W("**2. OSMAGC 联邦化零损失。** central 与 federated 的 H 序列**逐元素相同**"
      "（10 轮 max|ΔH| = 0）。因其唯一跨视图耦合是求和项 `Σ R_v Z_v`，"
      "客户端本地算 `R_v Z_v`、服务器相加即可精确重建——不是近似，是同一个数。"
      "代价全在通信量（Trento 182 MB，随像素数 N 线性增长）。"
      "⚠️ 前提是视图划分（各客户端共享完整栅格）；区域划分会切断空间平滑，那时才有损失。")
    W("")
    W("**3. 三个方法跑不了，分两类原因：**")
    W("")
    W("| 方法 | 原因 | 类别 |")
    W("|---|---|---|")
    W("| Fed-MVKM | 视图划分下要求客户端持有全部视图 | 联邦设定不匹配 |")
    W("| AMKSC | 内存随「波段数 ×81」增长 | 数据规模 |")
    W("| PMIMC | 显存随「N × protorate × batch」增长 | 数据规模 |")
    W("")
    W("→ **「能不能跑」本身是方法设计的一个维度。**")
    W("")
    W("⚠️ **局限**：通用算法仅 2 个样本（Fed-MVKM、DIVIDE），"
      "不足以支撑「通用算法普遍不适应遥感」这类表述。")
    W("")

    # ---------------- 一、目标 ----------------
    W("## 一、目标")
    W("")
    W("把通用联邦多视图聚类算法（为图片/文本设计，不了解遥感）与遥感专用算法"
      "（OSMAGC / AMKSC，显式建模空间邻域）放到同一批遥感数据上比较，"
      "回答：通用算法掉多少分？联邦化损失多大？哪些方法跑不了？")
    W("")
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
    W("## 四、评测协议")
    W("")
    W("**四条统一**：① 最终接口 = 每像素一个簇标签；② 评价函数一律由 "
      "`common/metrics.py` 计算，**不采信任何方法自报的数字**；③ 只评 `gt>0` 的像素；"
      "④ 同一份划分、同一组种子。")
    W("")
    W("**指标口径**：ACC（匈牙利置换后）、NMI（算术平均归一化）、ARI、"
      "Purity（每簇取最大真值类累加，非置换后的 ACC）。已与 sklearn / OSMAGC / "
      "FedRSMVC / Fed-MVKM 四份实现逐位对拍，68 项 0 失配。")
    W("")
    W("**诚实读出协议**：以 k-means 读出的方法统一用 "
      "`KMeans(n_init=10, random_state=固定)`，按 inertia 选初始化，"
      "**全程不接触真值**。OSMAGC 官方协议是「用真值挑 ACC 最高的那轮」，"
      "实测在 MUUFL 上虚高 8.6 个点，本平台禁用。")
    W("")
    W("**协议指纹**：BLAS 线程数 4、COL_CHUNK 25000、读出 n_init=10 —— "
      "换任一项数字都会变，逐条记录在 `runs.jsonl`。")
    W("")
    W("## 五、主结果表")
    W("")
    W("![主结果：四个数据集上的 ACC 对比](figures/chart.png)")
    W("")
    W("> 四个面板共用同一横轴量程（自 0 起）。蓝=遥感专用、橙=通用、绿=不协作下界。"
      "交互版（含悬停数值与深色模式）：`results/figures/accuracy_by_dataset.html`，"
      "用 `python scripts/make_chart.py` 重生成。")
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
    # （原「六、逐数据集明细」已删：与第五节的表重复；按划分拆分的明细见 runs.jsonl）

    W("## 六、无法运行的组合")
    W("")
    if errors:
        W("| 数据集 | 方法 | 划分 | 原因 |")
        W("|---|---|---|---|")
        for (d, m, pp), note in sorted(errors.items()):
            W(f"| {d} | `{m}` | {pp} | {note} |")
    else:
        W("本轮无。")
    W("")
    W("`N/A` 不是实验失败，而是**方法对数据结构/规模的硬依赖**——"
      "与 Fed-MVKM 在视图划分下不可运行属同一类结论（视图划分要求客户端持有全部视图）。")
    W("")
    W("## 七、已知问题")
    W("")
    W("1. **Fed-MVKM 在 Augsburg 上不稳定**：三种子 ACC 标准差 ±0.045；"
      "`beta_k` 最优值呈孤立尖峰（0.46→0.54→0.41），典型拟合噪声。"
      "**不宜作为单一结论依据**。")
    W("2. **Fed-MVKM「联邦版优于不联邦版」反常**：Augsburg 上 M=1 ≈0.32、"
      "M=4 ≈0.48–0.54，方向与理论相反。适配器已用其官方数据集回归验证"
      "（精确复现 ACC=0.8530），故反常来自数据/参数而非代码，**原因待查**。")
    W("3. **Salinas 的「多视图」性质特殊**：只有高光谱一个真实传感器，"
      "EMP/Gabor 是派生特征，不存在「不同机构持有不同模态」的现实动机。")
    W("4. **OSMAGC 超参已逐数据集调优**（`results/tuning_osmagc_*.json`）："
      "四个数据集最优 λ 都是 1，最优 γ 跨两个数量级（1 / 10 / 10 / 100）——"
      "λ 可迁移、γ 必须逐数据集调。**论文默认值在四个数据集上全部不是最优**。")
    W("5. **未纳入的**：Berlin（本地未接入）；PMIMC（适配器已写但不适用，见第六节）；"
      "ASCR / FPFC / MSSAGF / SIMVC-SA（本机无 MATLAB）；CeFMC（本地无代码）。")
    W("6. **局限**：通用算法仅 2 个样本，不足以支撑「普遍不适应」这类表述。")
    W("")
    W("## 八、如何复现")
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
