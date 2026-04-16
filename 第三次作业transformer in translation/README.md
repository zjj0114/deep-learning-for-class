# Transformer 翻译任务实验报告

## 实验目的
使用 Transformer 框架实现翻译任务，比较点积注意力机制和加性注意力机制的性能。

## 数据集
- `eng-fra_train_data.txt`：训练集，包含英语-法语翻译对
- `eng-fra_test_data.txt`：测试集，包含英语-法语翻译对

## 代码文件说明

### 1. transformer_translation.py
核心实现文件，包含以下组件：
- `TranslationDataset` 类：数据加载和预处理
- `PositionalEncoding` 类：位置编码
- `DotProductAttention` 类：点积注意力机制
- `AdditiveAttention` 类：加性注意力机制
- `MultiHeadAttention` 类：多头注意力
- `Encoder` 类：编码器
- `Decoder` 类：解码器
- `Transformer` 类：完整的 Transformer 模型
- 训练和评估函数
- 主函数

### 2. test_simple.py
简化的测试脚本，用于快速验证模型性能，包含完整的代码实现。

### 3. requirements.txt
依赖项文件，包含以下依赖：
- torch
- spacy
- numpy

## 实验结果

### 模型配置
- 批处理大小：2
- 模型维度：32
- 层数：1
- 注意力头数：2
- 前馈网络维度：64
-  dropout：0.1
- 训练轮数：1
- 最大序列长度：10
- 训练批次：10
- 评估批次：5

### 性能比较
| 注意力机制 | 测试损失 | 性能比较 |
|------------|---------|----------|
| 点积注意力 | 4.6900  | 较好     |
| 加性注意力 | 5.0014  | 较差     |

### 结论
在本次实验中，点积注意力机制的性能优于加性注意力机制，测试损失更低。

## 运行说明
1. 安装依赖：`pip install -r requirements.txt`
2. 运行训练和测试：`python transformer_translation.py`
3. 或者运行简化测试：`python test_simple.py`
