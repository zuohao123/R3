# InfoVQA Dataset

InfoVQA 是一个基于信息图表的视觉问答数据集，需要理解复杂的图表、图形和文本信息。

## 数据集结构

```
infovqa/
├── infovqa_train.json      # 训练集标注文件
├── infovqa_val.json        # 验证集标注文件
├── infovqa_test.json       # 测试集标注文件（可选）
└── images/                 # 信息图表图像目录
    ├── infographic1.png
    ├── infographic2.jpg
    └── ...
```

## 下载说明

1. **官方网站**: https://www.docvqa.org/datasets/infographicvqa
2. **数据下载**:
   - 标注文件: 下载 train/val 的 JSON 标注文件
   - 图像文件: 下载信息图表图像数据

### 一键整理（可选）
```bash
python scripts/download_datasets.py --datasets infovqa --output-root data_pipeline/data
```

## 数据放置步骤

1. 将下载的标注文件重命名并放置：
   ```bash
   # 将官方标注文件重命名为标准格式
   cp infographicsVQA_train_v1.0.json infovqa_train.json
   cp infographicsVQA_val_v1.0.json infovqa_val.json
   cp infographicsVQA_test_v1.0.json infovqa_test.json  # 如果有测试集
   ```

2. 将信息图表图像解压到 `images/` 目录：
   ```bash
   # 解压图像文件到 images 目录
   unzip infographicsVQA_images.zip -d images/
   ```

## 标注文件格式

```json
[
  {
    "id": "sample_id",
    "question": "What percentage of people prefer option A?",
    "answer": "75%",
    "image": "infographic1.png",
    "ocr_tokens": [...],        // 可选，OCR 标注
    "captions": [...],          // 可选，图表描述
    "context_evidence": [...]   // 可选，上下文信息
  }
]
```

## 数据集特点

- **复杂图表**: 包含各种类型的信息图表、统计图、流程图等
- **多模态信息**: 结合文本、图像、图表等多种信息源
- **推理要求**: 需要复杂的视觉推理和数值计算能力

## 验证数据集

```bash
# 从项目根目录运行
python -c "
from data_pipeline.datasets.infovqa import InfoVQADataset
from pathlib import Path
dataset = InfoVQADataset(Path('data_pipeline/data/infovqa'), 'train')
print(f'Dataset loaded successfully with {len(dataset)} samples')
print('Sample:', dataset[0])
"
```

## 注意事项

- 信息图表通常包含大量文本和数值信息
- 建议启用 OCR 功能以提取图表中的文本
- 支持的图像格式：.jpg, .png, .jpeg
- 图表类型多样，包括柱状图、饼图、流程图、地图等
