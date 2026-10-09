"""DIVIDE 适配器 —— 通用多视图聚类（AAAI'24），**非遥感设计**。

DIVIDE = Decoupled Contrastive Multi-View Clustering with High-Order Random Walks。
它是本平台「通用算法」那一侧第二个方法（此前只有 Fed-MVKM，n=1 撑不起论点）。

⚠️ 为什么必须自己重跑读出，而不能用它的 ACC
------------------------------------------------
`engine_train.evaluate()` 内部：
    1. 取学到的表示 features_cat（(n_sample, n_views×embed_dim)，L2 归一化）
    2. `KMeans(n_clusters=K, random_state=0).fit_predict(features_cat)`  ← 这一步干净
    3. `utils.evaluate(labels, kmeans_label)` → 内部 `get_y_preds` **用混淆矩阵做
       最优匹配（Munkres）** 算 ACC

第 3 步就是本平台明令禁止的「用真值挑结果」（与 OSMAGC 官方的 oracle 轮次选择同类）。
所以本适配器**只取第 2 步的聚类标签**，第 3 步换成平台的 `common/metrics.py`。

⚠️ 另外，平台统一用 `KMeans(n_init=10, random_state=seed)` 读出，
而 DIVIDE 内部是 `random_state=0`。本适配器**不用它内部的 kmeans**，
而是取出 features_cat 后按平台协议重跑（见 `common/protocol` 的诚实读出协议）。

依赖：`D:\\cc\\OSMAGC\\baselines\\DIVIDE`（原论文代码，未改动）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DIVIDE_DIR = Path(__file__).resolve().parents[2] / "OSMAGC" / "baselines" / "DIVIDE"


@dataclass(frozen=True)
class DIVIDEParams:
    """默认值取自 DIVIDE 官方 config 的 Scene15（低维输入那一组）与论文默认。"""

    epochs: int = 200
    start_rectify_epoch: int = 100
    warmup_epochs: int = 20
    batch_size: int = 1024
    blr: float = 5e-4          # base lr（会按 batch_size/256 缩放）
    temperature: float = 0.5
    momentum: float = 0.98
    drop_rate: float = 0.2
    hidden: tuple[int, ...] = (1024, 1024, 1024)
    embed_dim: int = 128
    train_time: int = 1        # 官方默认 5（内部重复 5 次取均值）；本平台自带多种子，取 1


DEFAULT_PARAMS: dict[str, DIVIDEParams] = {
    "Augsburg": DIVIDEParams(),
    "MUUFL":    DIVIDEParams(),
    "Trento":   DIVIDEParams(),
    "Salinas":  DIVIDEParams(),
}


def _build_encoder_dims(ds, p: DIVIDEParams) -> list[list[int]]:
    """每个视图一个 MLP 结构：[输入维度, *hidden, embed_dim]。"""
    return [[int(c.shape[2]), *p.hidden, p.embed_dim] for c in ds.cubes]


def _availability_mask(ds) -> np.ndarray:
    """(N, V) bool 的「该像素在该视图是否可用」掩码。

    平台里只有 MUUFL 有像素级缺失（HS/LiDAR 各缺 25%、缺失集不重叠）；
    其余数据集 `index` 为 None，表示全部可用。
    """
    n, v = ds.n_pixels, ds.n_views
    m = np.ones((n, v), dtype=bool)
    if ds.index is not None:
        for i in range(v):
            m[:, i] = False
            m[np.asarray(ds.index[i], dtype=np.int64), i] = True
    return m


def run_divide(ds, params: DIVIDEParams | None = None, seed: int = 0,
               device: str | None = None, verbose: bool = False):
    """在 `RSDataset` 上跑一次 DIVIDE。返回平台 F 序的簇标签。

    数据顺序：用 `flatten_view(v, "F")` 展平，索引与平台的 F 序**逐位对应**，
    因此标签不需要置换（DIVIDE 只在样本维上操作，不碰空间结构）。
    """
    if str(DIVIDE_DIR) not in sys.path:
        sys.path.insert(0, str(DIVIDE_DIR))

    import torch
    from torch.utils.data import DataLoader
    from dataset_loader import IncompleteMultiviewDataset, IncompleteDatasetSampler
    from model import DIVIDE
    from engine_train import train_one_epoch

    p = params or DEFAULT_PARAMS.get(ds.name, DIVIDEParams())
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # ---- 1. 数据：平台 RSDataset -> DIVIDE 期望的 (N, B) 矩阵列表 ----
    data_x = [np.asarray(ds.flatten_view(v, "F"), dtype=np.float32) for v in range(ds.n_views)]
    label_y = ds.flatten_gt("F").astype(np.int64)
    n_sample = int(data_x[0].shape[0])
    mask_np = _availability_mask(ds)

    # 用 DIVIDE 自带的 Dataset（它的 __getitem__ 返回 (idx, [views], mask, label)，
    # 正是 train_one_epoch 解包需要的四元组）。
    # missing_rate 传 0：不让它**随机生成**掩码，随后注入平台记录的真实可用性掩码。
    # 这样 MUUFL 的像素级缺失（HS/LiDAR 各缺 25%、缺失集不重叠）会真正生效——
    # DIVIDE 的 extract_feature 会据此走 cross_view_decoder 补全缺失视图。
    dataset = IncompleteMultiviewDataset(ds.n_views, data_x, label_y, missing_rate=0.0)
    dataset.missing_mask = torch.from_numpy(mask_np).bool()
    # 注意：IncompleteDatasetSampler 是 **Sampler**（逐个产出索引），不是 BatchSampler，
    # 必须传给 `sampler=` 让 DataLoader 自己去分批（官方 main_train.py:88 就是这么用的）。
    sampler = IncompleteDatasetSampler(dataset, seed=seed)
    bs = min(p.batch_size, len(sampler))
    loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=bs,
                        num_workers=0, pin_memory=False, drop_last=False)

    # ---- 2. 构造 args（DIVIDE 的 train_one_epoch 会读这些字段）----
    class _A:
        pass

    args = _A()
    args.n_views = ds.n_views
    args.n_classes = int(ds.n_classes)
    args.n_sample = n_sample
    args.embed_dim = p.embed_dim
    enc = _build_encoder_dims(ds, p)
    args.encoder_dim = enc
    args.batch_size = p.batch_size
    args.temperature = p.temperature
    args.momentum = p.momentum
    args.drop_rate = p.drop_rate
    args.start_rectify_epoch = p.start_rectify_epoch
    args.epochs = p.epochs
    args.warmup_epochs = p.warmup_epochs
    args.lr = p.blr * (p.batch_size / 256)
    args.print_freq = 50
    args.print_this_epoch = False
    args.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DIVIDE(n_views=ds.n_views, layer_dims=enc, temperature=p.temperature,
                   n_classes=ds.n_classes, drop_rate=p.drop_rate).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # ---- 3. 训练 ----
    for epoch in range(p.epochs):
        args.print_this_epoch = verbose and ((epoch + 1) % args.print_freq == 0)
        train_one_epoch(model, loader, loader, optimizer, dev, epoch, None, args)

    # ---- 4. 取出学到的表示（复刻 engine_train.evaluate 的特征部分）----
    model.eval()
    with torch.no_grad():
        feats = torch.zeros(ds.n_views, n_sample, p.embed_dim, device=dev)
        for idx, views, mask, _ in loader:
            views = [views[i].to(dev, non_blocking=True) for i in range(ds.n_views)]
            mask = mask.to(dev, non_blocking=True)
            z = model.extract_feature(views, mask)
            for i in range(ds.n_views):
                feats[i][idx] = z[i]
        features_cat = feats.permute(1, 0, 2).reshape(n_sample, -1)
        features_cat = torch.nn.functional.normalize(features_cat, dim=-1).cpu().numpy()

    # ---- 5. 平台协议读出：KMeans(n_init=10, random_state=seed)，不碰真值 ----
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=int(ds.n_classes), n_init=10,
                    random_state=seed).fit_predict(features_cat).astype(np.int32)

    detail = {
        "encoder_dims": enc,
        "epochs": p.epochs,
        "feature_dim": int(features_cat.shape[1]),
        "device": str(dev),
    }
    return labels, detail


def divide_applicable(ds) -> tuple[bool, str]:
    """DIVIDE 在这份数据上能不能跑。"""
    if ds.n_views < 2:
        return False, f"DIVIDE 需要至少 2 个视图，{ds.name} 只有 {ds.n_views} 个"
    return True, ""
