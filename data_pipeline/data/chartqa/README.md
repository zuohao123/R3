# ChartQA Dataset

ChartQA 是一个专注于图表理解和问答的数据集，包含各种类型的统计图表。

## 数据集结构

```
chartqa/
├── chartqa_train.json      # 训练集标注文件
├── chartqa_val.json        # 验证集标注文件
├── chartqa_test.json       # 测试集标注文件（可选）
└── charts/                 # 图表图像目录（注意：目录名为 charts）
    ├── chart1.png
    ├── chart2.jpg
    └── ...
```

## 下载说明

1. **官方网站**: https://github.com/vis-nlp/ChartQA
2. **数据下载**:
   - 标注文件: 下载 train/val 的 JSON 标注文件
   - 图表图像: 下载图表图像数据

## 数据放置步骤

1. 将下载的标注文件重命名并放置：
   ```bash
   # 将官方标注文件重命名为标准格式
   cp train_augmented.json chartqa_train.json
   cp val_augmented.json chartqa_val.json
   cp test_augmented.json chartqa_test.json  # 如果有测试集
   ```

2. 将图表图像解压到 `charts/` 目录：
   ```bash
   # 解压图像文件到 charts 目录
   unzip train_images.zip -d charts/
   unzip val_images.zip -d charts/
   ```

## 标注文件格式

```json
[
  {
    "id": "sample_id",
    "question": "What is the highest value in the chart?",
    "answer": "100",
    "image": "chart1.png",
    "chart_type": "bar_chart",    // 可选，图表类型
    "ocr_tokens": [...],          // 可选，OCR 标注
    "captions": [...]             // 可选，图表描述
  }
]
```

## 图表类型

ChartQA 包含多种图表类型：
- **柱状图** (Bar Charts)
- **折线图** (Line Charts)
- **饼图** (Pie Charts)
- **散点图** (Scatter Plots)
- **面积图** (Area Charts)
- **组合图** (Combination Charts)

## 验证数据集

```bash
# 从项目根目录运行
python -c "
from data_pipeline.datasets.chartqa import ChartQADataset
from pathlib import Path
dataset = ChartQADataset(Path('data_pipeline/data/chartqa'), 'train')
print(f'Dataset loaded successfully with {len(dataset)} samples')
print('Sample:', dataset[0])
"
```

## 注意事项

- **重要**: 图像目录名必须是 `charts/`，不是 `images/`
- 图表通常包含数值信息，需要精确的数值理解能力
- 支持的图像格式：.jpg, .png, .jpeg
- 建议启用 OCR 功能以提取图表中的文本和数值