# 20 Newsgroups 文本分类作业

这个目录按根目录下 `作业` 的结构整理，用 PyTorch 做二分类文本分类。当前数据划分已经对齐根目录 [`20_news_data.py`](C:\Users\Administrator\Desktop\工作文档\20_news_data.py) 的官方 `fetch_20newsgroups(subset='train'/'test')` 方式。主模型选用双向 GRU，因为它比普通 RNN 更稳定、比 LSTM 参数更省，在这个规模的二分类任务上通常有更好的速度和精度平衡。

本作业默认分类两个类别：

- `alt.atheism`
- `soc.religion.christian`

## 目录结构

- `src/data_utils.py`: 本地数据读取、文本清洗、划分训练/验证/测试集、构建词表
- `src/train_gru_classifier.py`: 双向 GRU 训练与评估主脚本
- `docs/实验说明.md`: 模型选择、参数设置和结果解读
- `artifacts/`: 训练后自动生成的模型和图表
- `data/`: 可选的本地数据目录
- `environment.yml`: conda 环境文件
- `run_homework.bat`: Windows 下优先复用本机 PyTorch 环境并运行训练

`environment.yml` 里已经显式包含 `pytorch` 和 `pytorch-cuda`，如果本机没有现成环境，可以直接按该文件创建 conda 环境。

## 数据放置方式

脚本会优先从以下位置查找本地数据：

- `20news作业\data\sklearn_data\20news-bydate_py3.pkz`
- `20news作业\data\20_newsgroups`
- 根目录下 `.tmp\20news_data\20_newsgroups`
- `20news作业\data\20news-bydate-train` 与 `20news作业\data\20news-bydate-test`

现在项目里已经可以直接携带官方缓存文件 `data\sklearn_data\20news-bydate_py3.pkz`，不再必须依赖根目录 `.tmp`。

如果根目录下 `.tmp\sklearn_data` 里已经有 `20news-bydate_py3.pkz`，脚本会优先使用这个官方缓存，并保持和 `20_news_data.py` 一致的 train/test 划分。

## 运行方式

直接双击：

```bat
run_homework.bat
```

或手动运行：

```powershell
python src\train_gru_classifier.py --data-root ..\.tmp\20news_data
```

## 默认训练设置

- 模型：BiGRU
- `embedding_dim = 128`
- `hidden_dim = 128`
- `num_layers = 2`
- `dropout = 0.35`
- `batch_size = 64`
- `learning_rate = 7e-4`
- `epochs = 18`
- `patience = 4`

## 训练结果输出

运行后 `artifacts/` 中会生成：

- `best_model.pt`: 最优模型参数
- `metrics.json`: 数据规模和测试集精度
- `classification_report.txt`: Precision / Recall / F1
- `training_history.png`: 训练与验证曲线
- `confusion_matrix.png`: 混淆矩阵

