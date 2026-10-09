"""AMKSC 适配器 —— 第二个「遥感专用」方法，与 OSMAGC 并列。

AMKSC（TNNLS'24, Cai et al.）的全名是 *Anchor-based Multiview Kernel Subspace
Clustering with Spatial Regularization*，论文标题直接写着
"Joint Clustering of **Hyperspectral and LiDAR** Data"——**它带空间总变差正则
（`lambda_tv`），是为遥感数据设计的**，不是通用 MVC。

所以本平台把它归到「遥感专用」一侧。这样表里就有两个遥感专用方法（OSMAGC + AMKSC），
"两个遥感专用方法是否都优于通用方法"比"一个方法赢了"有说服力得多。

⚠️ 两个必须处理的差异
--------------------
1. **像素顺序**：AMKSC 内部按 **C 序（行优先）** 展平，本平台统一用 **F 序（列优先）**。
   不做置换的话标签会整体错位，指标掉到随机水平。
2. **空间块**：AMKSC 对每个像素取 `patch_size×patch_size` 的邻域块（reflect 填充）拼成
   特征，这是它空间正则的一部分。**必须用它自己的 `get_HSI_patches_rw`**，
   自己重写一遍很容易在填充方式上出现细微差别。

依赖：`D:\\cc\\OSMAGC\\baselines\\AMKSC`（原论文代码，未改动）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AMKSC_DIR = Path(__file__).resolve().parents[2] / "OSMAGC" / "baselines" / "AMKSC"


@dataclass(frozen=True)
class AMKSCParams:
    """默认值取自 AMKSC 官方 config.yaml 里 Trento 那一组（它唯一启用的配置）。"""

    patch_size: int = 9
    n_anchor: int = 3            # anchor_type='k-means' 时，最终锚点数 = n_anchor × 类别数
    anchor_type: str = "k-means"
    lambda_tv: float = 1e-4      # 空间总变差权重 α
    smooth_coef: float = 1.15
    weight_decay: float = 10000  # γ
    max_iter: int = 5
    n_kernel_sampling: int = 300
    kernel: str | None = "rbf"


DEFAULT_PARAMS: dict[str, AMKSCParams] = {
    "Trento":   AMKSCParams(),
    "Augsburg": AMKSCParams(),
    "MUUFL":    AMKSCParams(),
}


def amksc_applicable(ds) -> tuple[bool, str]:
    """AMKSC 在这份数据上能不能跑。"""
    if ds.n_views < 2:
        return False, f"AMKSC 需要至少 2 个视图，{ds.name} 只有 {ds.n_views} 个"
    return True, ""


def _to_c_order_labels(y_pred_c: np.ndarray, H: int, W: int) -> np.ndarray:
    """AMKSC 的 C 序预测 → 平台的 F 序标签。

    C 序展平是从 (H,W) 按行拉直；平台用 F 序按列拉直。
    先 reshape 回 (H,W)（C 序可逆），再按 F 序拉直即可。
    """
    assert y_pred_c.size == H * W, f"预测长度 {y_pred_c.size} != H*W = {H * W}"
    return y_pred_c.reshape(H, W).reshape(-1, order="F").astype(np.int32)


def run_amksc(ds, params: AMKSCParams | None = None, seed: int = 0,
              device: str | None = None, verbose: bool = False):
    """在 `RSDataset` 上跑一次 AMKSC。

    Returns
    -------
    (labels, detail)
        `labels` : (N,) int32，**平台 F 序**的簇标签，全像素
    """
    if str(AMKSC_DIR) not in sys.path:
        sys.path.insert(0, str(AMKSC_DIR))

    import torch
    from sklearn.preprocessing import StandardScaler
    from Preprocessing import Processor
    from AMKSC import AMKSC

    params = params or DEFAULT_PARAMS.get(ds.name, AMKSCParams())
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    H, W = ds.H, ds.W
    gt2d = ds.gt.astype(np.int8)

    # ---- 1. 每个视图抽空间块（用 AMKSC 自己的实现，保证填充方式一致）----
    proc = Processor()
    modality_list = []
    for v, cube in enumerate(ds.cubes):
        xp, _ = proc.get_HSI_patches_rw(
            cube.astype(np.float32), gt2d,
            (params.patch_size, params.patch_size),
            stride=(1, 1), padding="reflect", is_labeled=False,
        )
        # xp: (N, ps, ps, B)，N = H*W，**C 序**
        n = xp.shape[0]
        x = xp.reshape(n, -1)
        x = StandardScaler().fit_transform(x)          # 与官方 loader 一致
        modality_list.append(x.astype(np.float32))

    # ---- 2. 跑模型 ----
    model = AMKSC(
        (H, W), int(ds.n_classes), n_anchor=params.n_anchor,
        anchor_type=params.anchor_type, lambda_tv=params.lambda_tv,
        smooth_coef=params.smooth_coef, weight_decay=params.weight_decay,
        max_iter=params.max_iter, n_kernel_sampling=params.n_kernel_sampling,
        epsilon=1e-5, device=device, kernel=params.kernel,
        verbose=verbose, seed=seed,
    )
    model.train(modality_list)
    y_pred_c = np.asarray(model.predict()).reshape(-1)

    labels = _to_c_order_labels(y_pred_c, H, W)

    detail = {
        "n_anchor_final": int(getattr(model, "n_anchor", -1)),
        "mu": None if getattr(model, "mu", None) is None
              else np.asarray(model.mu.detach().cpu()).tolist(),
        "patch_size": params.patch_size,
        "lambda_tv": params.lambda_tv,
        "weight_decay": params.weight_decay,
        "device": device,
    }
    return labels, detail
