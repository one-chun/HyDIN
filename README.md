# HyDIN

HyDIN 是一个用于蛋白质相互作用预测（Protein-Protein Interaction, PPI）的图学习项目。当前实现将双曲空间图编码与门控图小波变换结合起来，对蛋白质节点进行表示学习，并在链路预测任务上输出 ROC-AUC、AP、F1、MCC 等指标。

## 项目概览

当前代码的主流程分为两段：

1. 使用 `HyboNet` 在 Lorentz 双曲空间中编码蛋白质图结构与节点特征。
2. 使用 `GWTNet` 对编码后的表示做多尺度图小波变换与门控融合，再完成链路预测。

默认任务是：

- 输入蛋白质特征矩阵 `protein_feature.txt`
- 输入蛋白质相互作用邻接矩阵 `mat_protein_protein.txt`
- 预测未知蛋白质对之间是否存在相互作用

## 主要特性

- 双曲空间编码：基于 Lorentz manifold 的图编码器
- 多尺度小波融合：在 `models/models.py` 中实现 `GWTNet`
- 链路预测评估：支持 ROC-AUC、AP、F1、MCC
- Bootstrap 统计：评估阶段会计算置信区间与 p-value
- 两种训练方式：
  - 单次划分训练
  - K 折交叉验证
- Top-K 新相互作用发现：输出预测分数最高的候选蛋白质对

## 目录结构

```text
HyDIN/
├─ data/                  数据文件
│  ├─ protein_feature.txt
│  ├─ mat_protein_protein.txt
│  ├─ drug_feature.txt
│  ├─ disease_feature.txt
│  ├─ mat_drug_drug.txt
│  ├─ mat_drug_protein.txt
│  └─ zeng/               备用/对照数据
├─ layers/                图层与解码器
├─ manifolds/             双曲流形实现
├─ models/                编码器与图小波模型
├─ utils/                 数据处理、训练与评估工具
├─ train.py               训练入口
└─ requirements.txt       依赖列表
```

## 默认数据说明

当前默认训练路径实际使用的是以下两个文件：

- `data/protein_feature.txt`：蛋白质节点特征，尺寸为 `1512 x 128`
- `data/mat_protein_protein.txt`：蛋白质相互作用邻接矩阵，尺寸为 `1512 x 1512`

仓库中还包含以下文件，但默认训练入口 `load_data_lp()` 没有直接使用它们：

- `data/drug_feature.txt`：`708 x 128`
- `data/disease_feature.txt`：`5603 x 128`
- `data/mat_drug_drug.txt`：`708 x 708`
- `data/mat_drug_protein.txt`：`708 x 1512`
- `data/zeng/`：另一套蛋白质相关数据与 `.mat` 文件

如果你要替换自己的数据，至少需要保证默认输入文件满足下面的格式：

- `protein_feature.txt`：纯文本矩阵，每行一个节点，每列一个特征
- `mat_protein_protein.txt`：纯文本邻接矩阵，方阵，元素一般为 `0/1`

## 环境依赖

`requirements.txt` 中列出的核心依赖如下：

```txt
numpy==1.16.2
scikit-learn==0.20.3
torch
geoopt==0.5.0
torchvision==0.2.2
networkx==2.2
matplotlib<3.3
```



## 安装

```bash
pip install -r requirements.txt
```

如果你的环境里存在多个 Python 版本，建议显式指定解释器。

## 运行方式

直接运行：

```bash
python train.py
```

当前 `train.py` 默认会执行 5 折交叉验证，并在训练过程中输出：

- 每个 epoch 的训练损失与指标
- 验证集指标
- 每折的测试结果
- 分数最高的 Top-3 候选蛋白质相互作用
- 交叉验证 ROC/PR 曲线图 `cv_test_roc_pr_folds.png`

## 当前默认配置

虽然 `train.py` 里定义了命令行参数，但在文件末尾又手动覆盖了一遍默认值，因此直接传参未必会生效。当前代码实际默认配置大致如下：

```text
model = HyboNet
num_layers = 2
dim = 128
manifold = Lorentz
dropout = 0.0
val_prop = 0.25
test_prop = 0.25
split_seed = 523
cross_validation = 1
n_folds = 5
scales = [1, 2, 3, 4]
cuda = -1
```

另外，`single_fold_train()` 当前训练轮数写死为：

```python
for epoch in range(5):
```

也就是说，虽然脚本中保留了 `patience`、`min_epochs` 等参数，但默认训练实际上只跑 5 个 epoch。

## 关键模块说明

### 1. 双曲编码器

- 位置：`models/encoders.py`
- 默认模型：`HyboNet`
- 作用：在 Lorentz 双曲空间中对蛋白质图进行表示学习

### 2. 图小波融合模块

- 位置：`models/models.py`
- 默认模型：`GWTNet`
- 作用：对节点表示做多尺度随机游走小波变换，并通过门控机制融合多尺度信息

### 3. 数据处理与划分

- 位置：`utils/data_utils.py`
- 功能：
  - 特征与邻接矩阵归一化
  - 正负样本构造
  - 严格训练/验证/测试划分
  - 交叉验证数据准备

## 输出结果



## 自定义数据

如果你想把项目切换到自己的 PPI 数据集，最小改动方案通常是：

1. 用你自己的节点特征替换 `data/protein_feature.txt`
2. 用你自己的邻接矩阵替换 `data/mat_protein_protein.txt`
3. 保证节点数量一致
4. 保证特征矩阵与邻接矩阵按同一节点顺序排列

如果你要引入药物、疾病或多模态关系，当前默认训练路径还不够，需要继续修改 `load_data_lp()` 与训练逻辑。

## 已知事项

- 当前仓库的原始 `README` 只有标题，本文件是根据现有代码结构重新整理的说明文档。
- `train.py` 中存在“先解析参数，再手动覆盖参数”的行为。
- 默认训练入口只使用蛋白质相关数据，未直接接入药物/疾病分支。
- 代码中包含部分实验性逻辑与注释保留实现，适合在复现实验前先自行核对训练配置。

## 建议的后续整理方向

如果你准备继续维护这个项目，优先建议做下面几件事：

1. 去掉 `train.py` 末尾对参数的硬编码覆盖
2. 把训练 epoch 数改成真正可配置
3. 将数据加载器拆成 PPI、DTI、多模态三套明确入口
4. 补充实验配置、随机种子和结果复现说明

