# Principles and Methods of Pattern Recognition - Lab 2 Report

这个项目有两个主要目标：(1) 实现两种不同的三层前向神经网络反向传播算法（批量梯度下降和随机梯度下降）；(2) 分析隐含层节点数（hidden layer nodes）和学习率（learning rate）对神经网络性能的影响。通过系统的对比实验和详细的性能分析，本项目不仅展示了不同梯度下降策略在三维数据分类任务上的表现差异，还深入探讨了网络结构和训练参数对分类性能的影响。

## 作者
李庚晟 (Gengsheng Li)
2024E8014682056
Department: Laboratory of Brain Atlas and Brain-inspired Intelligence, Institute of Automation, Chinese Academy of Sciences, Beijing, China
December 19, 2024

## 项目结构

```
.
├── config.py           # 实验配置文件
├── model.py           # 神经网络模型实现
├── train.py           # 训练脚本
├── visualize_data.py  # 数据可视化脚本
└── results/           # 实验结果保存目录
```

## 环境要求

- Python 3.12.4
- NumPy
- Matplotlib

## 数据集

数据集包含三类三维数据点，每类10个样本：
- 第一类：主要分布在z轴负半空间
- 第二类：分布相对集中，波动范围较小
- 第三类：分布最为分散，特别是在x轴方向

## 模型特征

### 网络结构
- 输入层：3个节点（对应三维数据）
- 隐含层：可配置节点数（默认支持5, 10, 20）
- 输出层：3个节点（对应三个类别）

### 激活函数
- 隐含层：双曲正切函数（tanh）
- 输出层：sigmoid函数

### 训练方法
1. 批量梯度下降（BatchNeuralNetwork）
2. 随机梯度下降（StochasticNeuralNetwork）

### 支持特性
- 固定学习率训练
- 自适应学习率训练
- 损失函数可视化
- 准确率统计
- 支持多组超参数对比实验

## 使用说明

### 1. 配置实验参数

在 `config.py` 中设置实验参数：

```python
class ExperimentConfig:
    def __init__(self):
        # 设置随机种子（可选）
        self.random_seed = None
        
        # 配置隐含层节点数实验
        self.exp1 = True
        self.hidden_size_config = {
            'hidden_sizes': [5, 10, 20],
            'epochs': 2000,
            'learning_rate': 0.1,
        }
        
        # 其他实验配置...
```

### 2. 运行实验

执行训练脚本：
```bash
python train.py
```

### 3. 可视化数据分布

执行数据可视化脚本：
```bash
python visualize_data.py
```

## 实验类型

本项目支持四种实验：

1. 隐含层节点数影响分析（exp1）
   - 比较不同隐含层节点数对模型性能的影响
   - 默认节点数：[5, 10, 20]

2. 学习率影响分析（exp2）
   - 比较不同学习率对模型训练的影响
   - 默认学习率：[0.001, 0.01, 0.1]

3. 自适应学习率分析（exp3）
   - 比较固定学习率和自适应学习率的性能差异
   - 支持配置patience和衰减因子

4. 特定配置下的损失函数分析（exp4）
   - 观察指定配置下损失函数的变化趋势

## 输出说明

实验结果将保存在 `results/` 目录下，包含：
- 损失函数变化曲线图
- 准确率对比图
- 数值结果文本文件
- 实验配置记录

## 结果示例

每个实验会生成多个可视化结果：
- `hidden_size_loss_curves.png`: 不同隐含层节点数的损失曲线对比
- `learning_rate_loss_curves.png`: 不同学习率的损失曲线对比
- `adaptive_lr_comparison.png`: 自适应学习率与固定学习率的对比
- `specific_config_loss_curves.png`: 特定配置下的损失函数变化

实验结果将以时间戳命名的目录形式保存，便于追踪和比较不同实验。

## 注意事项

1. 每次运行实验前请确认配置文件中的参数设置
2. 建议首次运行时使用较小的epoch数进行测试
3. 对于大规模对比实验，建议分组进行以便管理结果
4. 自适应学习率实验中，确保设置合适的patience和衰减因子

## 可扩展性

项目支持以下扩展：
1. 添加新的优化策略
2. 修改网络结构
3. 自定义损失函数
4. 增加新的实验类型
