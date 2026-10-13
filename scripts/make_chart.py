"""从 results/runs.jsonl 生成主结果图表（自包含 HTML + 内联 SVG）。

设计依据（dataviz 方法的取舍）
------------------------------
- **形式**：数据的工作是"在同一数据集内比较各方法的量级"，且方法身份是重点。
  8 个方法超过单图彩色上限，故用**小倍数**：4 个数据集各一格，格内横向条形，
  方法顺序**固定**（按类别分组），方便跨格扫同一行。
- **颜色按类别（3 类）**：遥感专用 / 通用 / 下界。小倍数是"全对比"形式，
  上限 3 色——正好。
- **配色已过校验器**（不靠肉眼判断）：
    浅色 #2a78d6,#eb6834,#1baf7a  → 全通过，其中青色对比度 2.74 触发 WARN，
       按规矩**必须配可见数值标签**（本图每根条都直接标注，已满足）
    深色 #3987e5,#d95926,#199e70  → 全通过
- **刻度从 0 起**：ACC 是比例，截断轴会夸大差异。所有面板共用同一条 x 轴范围，
  跨数据集可比。

运行：python scripts/make_chart.py
输出：results/figures/accuracy_by_dataset.html
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ---- 方法顺序固定：按类别分组（读者可跨面板扫同一行）----
CATEGORIES = [
    ("遥感专用", "#2a78d6", "#3987e5", [
        ("OSMAGC (central, upper bound)", "OSMAGC 集中式"),
        ("OSMAGC (federated, view)", "OSMAGC 联邦化"),
        ("AMKSC", "AMKSC"),
    ]),
    ("通用（非遥感设计）", "#eb6834", "#d95926", [
        ("DIVIDE", "DIVIDE"),
        ("Fed-MVKM (region)", "Fed-MVKM"),
    ]),
    ("下界（不协作）", "#1baf7a", "#199e70", [
        ("Local-only (view, 不融合)", "Local-only 不融合"),
        ("Local-only (view, 融合)", "Local-only 融合"),
        ("Local-only (region)", "Local-only 区域"),
    ]),
]

DATASETS = ["Augsburg", "MUUFL", "Trento", "Salinas"]
METRIC = "ACC"


def load_stats() -> dict:
    rows = [json.loads(ln) for ln in (RES / "runs.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    acc = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok" and r.get(METRIC) is not None:
            acc[(r["dataset"], r["method"])].append(r[METRIC])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in acc.items()}


def rounded_bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """右端 4px 圆角、左端贴基线的横条路径（数据端圆角，锚定端方角）。"""
    r = min(r, max(w, 0.0), h / 2)
    if w <= 0.5:
        return ""
    return (f"M{x:.2f},{y:.2f} H{x + w - r:.2f} "
            f"A{r:.2f},{r:.2f} 0 0 1 {x + w:.2f},{y + r:.2f} "
            f"V{y + h - r:.2f} A{r:.2f},{r:.2f} 0 0 1 {x + w - r:.2f},{y + h:.2f} "
            f"H{x:.2f} Z")


def build() -> str:
    stats = load_stats()
    flat = [(m, lab, ci) for ci, (_, _, _, ms) in enumerate(CATEGORIES) for m, lab in ms]

    # ---- 版面 ----
    PW, PH = 560, 300          # 单面板尺寸
    GAP_X, GAP_Y = 28, 54
    M_LEFT, M_TOP = 128, 46    # 面板内的绘图区边距
    M_RIGHT, M_BOT = 62, 34
    COLS, ROWS = 2, 2
    TOTAL_W = M_LEFT + COLS * PW + (COLS - 1) * GAP_X + 20
    TOTAL_H = M_TOP + ROWS * PH + (ROWS - 1) * GAP_Y + 46

    plot_w = PW - M_LEFT - M_RIGHT
    plot_h = PH - M_TOP - M_BOT
    n = len(flat)
    row_h = plot_h / n
    bar_h = row_h - 2          # 2px 表面间隙

    x_max = 1.0                # ACC ∈ [0,1]，从 0 起，四面板共用，跨数据集可比

    out: list[str] = []
    A = out.append

    for pi, ds in enumerate(DATASETS):
        cx, cy = pi % COLS, pi // COLS
        ox = M_LEFT + cx * (PW + GAP_X)
        oy = M_TOP + cy * (PH + GAP_Y)
        A(f'<g class="panel" transform="translate({ox},{oy})">')
        A(f'  <text class="ds-title" x="0" y="-16">{ds}</text>')

        # 竖向网格（recessive）
        for t in np.arange(0, 1.0001, 0.2):
            gx = t / x_max * plot_w
            A(f'  <line class="grid" x1="{gx:.2f}" y1="0" x2="{gx:.2f}" y2="{plot_h:.2f}"/>')
            A(f'  <text class="tick" x="{gx:.2f}" y="{plot_h + 15:.0f}">{t:.1f}</text>')

        # 类别分隔线
        prev_ci, acc_i = None, 0
        for m, lab, ci in flat:
            y = acc_i * row_h
            if prev_ci is not None and ci != prev_ci:
                A(f'  <line class="sep" x1="{-8}" y1="{y - 1:.2f}" x2="{plot_w:.2f}" y2="{y - 1:.2f}"/>')
            prev_ci = ci

            st = stats.get((ds, m))
            A(f'  <text class="mlabel" x="{-12}" y="{y + bar_h / 2 + 4:.2f}">{lab}</text>')
            if st is None:
                A(f'  <text class="na" x="4" y="{y + bar_h / 2 + 4:.2f}">不适用</text>')
            else:
                mean, sd, cnt = st
                w = mean / x_max * plot_w
                color = CATEGORIES[ci][1]
                A(f'  <path class="bar" d="{rounded_bar_path(0, y, w, bar_h)}" fill="{color}"/>')
                A(f'  <text class="val" x="{w + 7:.2f}" y="{y + bar_h / 2 + 4:.2f}">{mean:.3f}</text>')
                A(f'  <rect class="hit" x="0" y="{y:.2f}" width="{plot_w:.2f}" height="{bar_h:.2f}" '
                  f'data-d="{ds}" data-m="{lab}" data-v="{mean:.4f}" data-s="{sd:.4f}" data-n="{cnt}"/>')
            acc_i += 1

        # 基线
        A(f'  <line class="axis" x1="0" y1="0" x2="0" y2="{plot_h:.2f}"/>')
        A(f'  <text class="xlab" x="{plot_w / 2:.0f}" y="{plot_h + 32:.0f}">ACC</text>')
        A('</g>')

    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{name}</span>'
        for name, c, _, _ in CATEGORIES
    )

    return f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>遥感多视图聚类 · 主结果（ACC）</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin:0; padding:28px 32px 40px; background:#fcfcfb; color:#0b0b0b;
         font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }}
  h1 {{ font-size:19px; margin:0 0 4px; font-weight:600; }}
  .sub {{ color:#52514e; font-size:13px; margin-bottom:18px; }}
  .legend {{ margin:0 0 10px; }}
  .lg {{ display:inline-flex; align-items:center; margin-right:20px; font-size:13px; color:#52514e; }}
  .lg i {{ width:11px; height:11px; border-radius:2px; margin-right:6px; display:inline-block; }}
  svg {{ display:block; }}
  .ds-title {{ font-size:13px; font-weight:600; fill:#0b0b0b; }}
  .mlabel {{ font-size:11.5px; fill:#52514e; text-anchor:end; }}
  .val {{ font-size:11.5px; fill:#0b0b0b; font-variant-numeric:tabular-nums; }}
  .na {{ font-size:11.5px; fill:#8a8a86; }}
  .tick {{ font-size:10.5px; fill:#8a8a86; text-anchor:middle; font-variant-numeric:tabular-nums; }}
  .xlab {{ font-size:11px; fill:#8a8a86; text-anchor:middle; }}
  .grid {{ stroke:#e8e7e3; stroke-width:1; }}
  .sep {{ stroke:#d9d8d3; stroke-width:1; stroke-dasharray:3 3; }}
  .axis {{ stroke:#c9c8c3; stroke-width:1; }}
  .bar {{ shape-rendering:geometricPrecision; }}
  .hit {{ fill:transparent; }}
  .hit:hover {{ fill:#00000010; }}
  .tip {{ position:fixed; pointer-events:none; opacity:0; transition:opacity .1s;
         background:#0b0b0b; color:#f0efec; padding:7px 10px; border-radius:6px;
         font-size:12px; line-height:1.5; white-space:nowrap; z-index:9;
         font-variant-numeric:tabular-nums; }}
  .note {{ margin-top:16px; color:#52514e; font-size:12.5px; max-width:960px; }}
  .note b {{ color:#0b0b0b; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#1a1a19; color:#f0efec; }}
    .sub, .lg, .mlabel, .note {{ color:#c3c2b7; }}
    .ds-title, .val {{ fill:#f0efec; }} .val {{ color:#f0efec; }}
    .tick, .xlab, .na {{ fill:#8f8e88; }}
    .grid {{ stroke:#2e2e2c; }} .sep {{ stroke:#3a3a37; }} .axis {{ stroke:#4a4a46; }}
    .hit:hover {{ fill:#ffffff14; }}
    .note b {{ color:#f0efec; }}
  }}
</style></head>
<body>
<h1>遥感多视图聚类 · 主结果（ACC，3 种子均值）</h1>
<div class="sub">横轴自 0 起，四个面板共用同一量程，可跨数据集比较。条端数字为该数据集上的 ACC 均值。</div>
<div class="legend">{legend}</div>
<svg width="{TOTAL_W}" height="{TOTAL_H}" viewBox="0 0 {TOTAL_W} {TOTAL_H}" role="img"
     aria-label="四个遥感数据集上各方法的 ACC 横向条形对比">
{"".join(out)}
</svg>
<div class="note">
  <b>怎么读</b>：蓝色（遥感专用）整体长于橙色（通用）；三个青色是「各客户端完全不协作」的下界，
  用来判断某个方法是不是连"不协作"都不如。<b>AMKSC 在 Augsburg / Salinas 上不适用</b>——
  它给每个像素取 9×9 邻域块，特征矩阵分别达 9.65 GB / 12.20 GB，超出本机内存。
</div>
<div class="tip" id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('.hit').forEach(el => {{
  el.addEventListener('mousemove', e => {{
    const d = el.dataset;
    tip.innerHTML = `<b>${{d.d}}</b><br>${{d.m}}<br>ACC ${{d.v}} ± ${{d.s}}（n=${{d.n}} 种子）`;
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top  = (e.clientY + 14) + 'px';
    tip.style.opacity = 1;
  }});
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
}});
</script>
</body></html>
'''


if __name__ == "__main__":
    outdir = RES / "figures"
    outdir.mkdir(exist_ok=True)
    path = outdir / "accuracy_by_dataset.html"
    path.write_text(build(), encoding="utf-8")
    print(f"已写入 {path}")
