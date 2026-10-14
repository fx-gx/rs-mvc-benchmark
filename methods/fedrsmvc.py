"""FedRSMVC 适配器 —— **遥感专用 · 联邦**（用户自己的方法，ICML 在审）。

为什么必须自己重跑读出
----------------------
`Flsimulator._save_cluster_map()` 在可视化前调用 `map_labels(y, sp_pred)`——
**用真值做匈牙利最优映射**（它自己的注释也写了"簇号→GT 类别号"）。与 OSMAGC 官方的
oracle 轮次选择、DIVIDE / PMIMC 的 `get_y_preds` 是同一类问题，平台一律不采信。

适配器只取 `Flsimulator` 第 271 行那个**未经映射**的 `cluster_labels_raw`，
经超像素→像素还原后交给 `common/metrics.py` 重算。

⚠️ 三处必须处理
----------------
1. **像素顺序**：`utils.superpixel_utils.create_association_mat` 用
   `superpixel_labels.reshape(-1)`，是 **C 序（行优先）**；平台统一用 **F 序**。
   不做置换标签会整体错位，指标掉到随机水平。
2. **`main()` 路径不便复用**：`main.py` 直接 `simulator.start()` 且把结果写日志。
   适配器改为直接构造 `FLSimulator` 并给 `full_metric` 装探针，拦下原始标签。
   **不改它一行代码。**
3. **它自带数据加载**（`config.json` 指向 `Federated-MVC/data/<数据集>/`）。
   本适配器沿用它的加载路径，但会核对形状与像素数是否与平台一致，
   不一致就报错而不是硬跑。

支持的数据集：Trento / Salinas / Augsburg（**它的 config.json 没有 MUUFL**）。
实测单次 300 轮约 42 分钟（GPU）。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRS_DIR = Path(__file__).resolve().parents[2] / "Federated-MVC"

# FedRSMVC 的 config.json 里有配置的数据集
SUPPORTED = {"Trento", "Salinas", "Augsburg"}


@dataclass(frozen=True)
class FedRSMVCParams:
    """默认沿用其 config.json 的论文设置，只暴露 300 轮这个通信轮数。"""

    epochs: int = 300


DEFAULT_PARAMS: dict[str, FedRSMVCParams] = {d: FedRSMVCParams() for d in SUPPORTED}


def fedrsmvc_applicable(ds) -> tuple[bool, str]:
    if ds.name not in SUPPORTED:
        return False, (f"FedRSMVC 的 config.json 未配置数据集 {ds.name}"
                       f"（仅支持 {sorted(SUPPORTED)}）")
    return True, ""


def run_fedrsmvc(ds, params: FedRSMVCParams | None = None, seed: int = 42,
                 device: str | None = None, verbose: bool = False):
    """跑一次 FedRSMVC，返回平台 F 序的簇标签 (N,)。"""
    if str(FRS_DIR) not in sys.path:
        sys.path.insert(0, str(FRS_DIR))

    import torch
    import Flsimulator as fls
    from Flsimulator import FLSimulator

    p = params or DEFAULT_PARAMS.get(ds.name, FedRSMVCParams())

    config = json.loads((FRS_DIR / "config.json").read_text(encoding="utf-8"))
    config["dataset_name"] = ds.name
    config["seed"] = int(seed)
    config["n_epoches"] = int(p.epochs)
    # config.json 里的数据路径是相对 Federated-MVC/ 的；不 chdir，改成绝对路径
    cds = config[ds.name]
    cds["mm_data_path"] = [str(FRS_DIR / q) for q in cds["mm_data_path"]]
    cds["gt_path"] = str(FRS_DIR / cds["gt_path"])
    config["save_model"] = False           # 不落模型，省时间与磁盘
    config["log_path"] = str(FRS_DIR / "log")
    config["save_path"] = str(FRS_DIR / "save")
    config["result_path"] = str(FRS_DIR / "results")
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # ---- 探针：拦下 full_metric 收到的**原始**聚类标签（未做 oracle 映射）----
    original_full_metric = fls.full_metric
    captured: dict[str, np.ndarray] = {}

    def spy(y, pred, is_refined=False):
        captured["raw_sp"] = np.asarray(pred).reshape(-1).copy()
        return original_full_metric(y, pred, is_refined=is_refined)

    fls.full_metric = spy
    try:
        sim = FLSimulator(config, device=dev)
        sim.start()
    finally:
        fls.full_metric = original_full_metric

    if "raw_sp" not in captured:
        raise RuntimeError("未能从 FedRSMVC 捕获原始聚类标签")

    sp_pred = captured["raw_sp"]

    # ---- 超像素 -> 像素（复用它的还原逻辑，但不做任何真值映射）----
    full_pred = sim.dataset.recover_background(sp_pred)   # (总超像素数,)，值为簇号
    n_sp = int(sim.dataset.association_mat_copy.shape[1])  # 超像素数（≠ 类别数！）
    K = int(full_pred.max())                               # 簇号最大值
    assert full_pred.size == n_sp, f"超像素数不符: {full_pred.size} vs {n_sp}"
    onehot = np.zeros((n_sp, K + 1), dtype=np.float32)
    onehot[np.arange(n_sp), full_pred] = 1.0
    pixel = np.argmax(sim.dataset.association_mat_copy @ onehot, axis=1)   # (H*W,)，C 序

    H, W = int(ds.H), int(ds.W)
    if pixel.size != H * W:
        raise RuntimeError(f"FedRSMVC 像素数 {pixel.size} != 平台 {H*W}")

    # ---- C 序 -> 平台 F 序 ----
    labels = pixel.reshape(H, W).reshape(-1, order="F").astype(np.int32)

    detail = {
        "n_superpixels": int(sim.dataset.association_mat_copy.shape[1]),
        "epochs": p.epochs,
        "device": str(dev),
        "raw_label_source": "Flsimulator 第 271 行 cluster_labels_raw（未做 oracle 映射）",
    }
    return labels, detail
