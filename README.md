# 遥感多视图聚类统一测试平台

把**通用联邦多视图聚类算法**（为图片/文本设计，不了解遥感）与**遥感专用算法**（显式建模空间邻域）
放到同一批遥感数据、同一套划分、同一组种子上比较，回答三个问题：

1. 通用算法在遥感数据上掉多少分？
2. 把遥感专用算法联邦化，损失有多大？
3. 哪些方法在遥感数据的规模/结构下**根本跑不了**？

> **完整实验结果与结论：[`results/REPORT.md`](results/REPORT.md)** —— 含 ACC/NMI/ARI/Purity
> 四张主结果表（均值±标准差）、无法运行的组合清单、以及已知问题（含反常识结果的记录）。

## 结论速览

**1. 通用算法在遥感数据上掉分。** DIVIDE 对比 OSMAGC：

| 数据集 | DIVIDE | OSMAGC | 差 |
|---|---|---|---|
| Trento | 0.6458 | 0.9158 | **−0.2700** |
| MUUFL | 0.3601 | 0.4145 | −0.0544 |
| Salinas | 0.6227 | 0.6736 | −0.0508 |
| Augsburg | 0.5068 | 0.4994 | +0.0075（打平，差 < 种子波动 0.0175） |

> ⚠️ 平台**未做归因实验**，报里给的三条（缺空间邻域建模 / 不处理像素级缺失 / 高光谱高维不适配）
> 是**待检验假说，不是已证结论**。通用算法也只有 2 个样本，不足以支撑「普遍不适应」这类表述。

**2. OSMAGC 联邦化零损失。** central 与 federated 的 H 序列**逐元素相同**（10 轮 max|ΔH| = 0）
——因为它唯一的跨视图耦合是求和项 `Σ R_v Z_v`，客户端本地算、服务器相加即可精确重建。
代价全在通信量（Trento 182 MB，随像素数线性增长）。

**3. 「能不能跑」本身是方法设计的一个维度。** 见 [`results/REPORT.md` §6](results/REPORT.md)：

| 方法 | 卡在哪 |
|---|---|
| `Fed-MVKM` | 视图划分下它要求客户端持有全部视图 —— 联邦设定不匹配 |
| `AMKSC` | 内存随「波段数 ×81」增长：Augsburg 需 9.65 GB、Salinas 12.20 GB |
| `PMIMC` | 显存随「N × protorate × batch」增长 |

## 方法清单

| 方法 | 角色 | 联邦设定 |
|---|---|---|
| `OSMAGC (central)` | 遥感专用 · 不联邦 · **上界** | 无需划分 |
| `OSMAGC (federated)` | 遥感专用 · 联邦化 · baseline | 视图划分（客户端 = 模态） |
| `AMKSC` | 遥感专用 · 不联邦 · 第二个遥感方法 | 无需划分 |
| `DIVIDE` | 通用 · 对比学习 · 非遥感设计 | 无需划分 |
| `Fed-MVKM (region)` | 通用联邦 · 水平联邦 | 区域划分（客户端 = 地理区块） |
| `FedRSMVC` | **遥感专用 · 联邦**（本文方法，ICML 在审） | 视图划分 |
| `Local-only (view, 不融合 / 融合)` | 下界 | 视图划分 |
| `Local-only (region)` | 下界 | 区域划分 |

## 数据集

| 数据集 | 尺寸 | 像素数 | 视图 | 类别 | 备注 |
|---|---|---|---|---|---|
| Augsburg | 332×485 | 161,020 | HS(180)+SAR(4)+DSM(1) | 7 | 唯一被 OSMAGC 与 FedRSMVC 都跑过的数据集；DSM 为近似重建 |
| MUUFL | 325×220 | 71,500 | HS(64)+LiDAR(2) | 11 | 官方**缺失率 0.5** 版本，两视图各丢 25%、缺失集不重叠 |
| Trento | 166×600 | 99,600 | HS(63)+LiDAR(2) | 6 | — |
| Salinas | 512×217 | 111,104 | HS(204)+EMP(45)+Gabor(90) | 16 | 仅高光谱一个真实传感器，另两个视图为派生特征 |

## 评测协议

**四条统一**：① 最终接口 = 每像素一个簇标签；② 评价函数一律由 `common/metrics.py` 计算，
**不采信任何方法自报的数字**；③ 只评 `gt > 0` 的像素；④ 同一份划分、同一组种子。

**诚实读出协议**：以 k-means 读出的方法统一用 `KMeans(n_init=10, random_state=固定)`，
按 inertia 选初始化，**全程不接触真值**。OSMAGC 官方协议是「用真值挑 ACC 最高的那轮」，
实测在 MUUFL 上虚高 8.6 个点，本平台禁用。

**指标对拍**：ACC / NMI / ARI / Purity 已与 sklearn、OSMAGC、FedRSMVC、Fed-MVKM
四份实现逐位核对，68 项 0 失配。

## 目录结构

```
├── common/               # 平台公共层
│   ├── data.py           #   统一遥感数据加载（像素一律 F 序列展平，与 OSMAGC MATLAB 对齐）
│   ├── metrics.py        #   ACC/NMI/ARI/Purity 唯一实现
│   ├── partition.py      #   视图划分 / 区域划分
│   ├── protocol.py       #   运行记录落盘（runs.jsonl，可断点续跑）
│   └── threads.py        #   BLAS 线程数锁定（协议指纹的一部分）
├── methods/              # 各方法适配器（每个方法一个文件）
│   ├── fedrsmvc.py       #   ⭐ 本文方法
│   ├── osmagc_central.py / osmagc_federated.py
│   ├── amksc.py / divide.py / pmimc.py / fed_mvkm.py
│   └── local_only.py     #   不协作下界
├── scripts/              # 跑批与出图
│   ├── run_full_matrix.py    # 主跑批入口
│   ├── make_report.py / make_chart.py
│   ├── tune_osmagc.py        # 超参逐数据集调优
│   └── verify_*.py           # 数据/指标/适配器自检
├── results/              # 结果与报告（REPORT.md 是主结论）
└── OSMAGC_MODIFICATIONS.md   # 对 OSMAGC 本地副本的改动台账
```

## 运行

```bash
pip install numpy scipy scikit-learn torch

python scripts/run_full_matrix.py                      # 全量跑批
python scripts/run_full_matrix.py --datasets Augsburg MUUFL --seeds 0
python scripts/make_chart.py                           # 重生成主结果图
```

## ⚠️ 数据依赖：需要同级项目目录

平台**不自带数据**（遥感数据集体积大，且多受原始发布方条款约束）。`common/data.py` 里的
路径以 `D:\cc` 为根、指向**同级项目目录**，运行前需要这样摆：

```
D:\cc\
├── 算法测试\              ← 本项目
├── OSMAGC\               ← 提供 Augsburg / Berlin / MUUFL 数据
│   ├── data\
│   └── official_code\data\    (MUUFL 的 .mat 在此)
└── Federated-MVC\        ← 提供 Trento / Salinas 数据
    └── data\
```

路径定义集中在 `common/data.py` 顶部的 `CC` 常量与 `SPECS` 字典，换机器只需改这两处。
各数据集的原始下载来源见 [OSMAGC](https://github.com/ZhangYongshan/OSMAGC) 与
[FedRSMVC](https://github.com/GuanRX/FedRSMVC) 的官方仓库说明。

## 相关项目

- [`osmagc-repro`](https://github.com/fx-gx/osmagc-repro) —— OSMAGC 的 MATLAB→Python 复现（本平台的上界方法来源）
- [`fedrsmvc-repro`](https://github.com/fx-gx/fedrsmvc-repro) —— FedRSMVC 复现（本平台的方法来源）

## 许可

MIT，见 [LICENSE](LICENSE)。第三方方法（OSMAGC / AMKSC / DIVIDE / PMIMC / Fed-MVKM）
的原始实现未随本仓库分发，出处见各自 README 或 `results/REPORT.md`。
