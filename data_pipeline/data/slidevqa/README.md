# SlideVQA Dataset

SlideVQA 是一个基于演示文稿幻灯片的视觉问答数据集，专注于理解幻灯片中的内容和结构。

## 数据集结构

```
slidevqa/
├── slidevqa_train.json     # 训练集标注文件
├── slidevqa_val.json       # 验证集标注文件
├── slidevqa_test.json      # 测试集标注文件（可选）
└── images/                 # 幻灯片图像目录
    ├── slide1.png
    ├── slide2.jpg
    └── ...
```

## 下载说明

1. **数据来源**: 根据具体的 SlideVQA 数据集来源进行下载
2. **数据下载**:
   - 标注文件: 下载 train/val 的 JSON 标注文件
   - 幻灯片图像: 下载幻灯片图像数据

## 数据放置步骤

1. 将下载的标注文件重命名并放置：
   ```bash
   # 将官方标注文件重命名为标准格式
   cp train.json slidevqa_train.json
   cp val.json slidevqa_val.json
   cp test.json slidevqa_test.json  # 如果有测试集
   ```

2. 将幻灯片图像解压到 `images/` 目录：
   ```bash
   # 解压图像文件到 images 目录
   unzip slides.zip -d images/
   ```

## 标注文件格式

```json
[
  {
    "id": "sample_id",
    "question": "What is the main topic of this slide?",
    "answer": "Machine Learning",
    "image": "slide1.png",
    "ocr_tokens": [...],        // 可选，OCR 标注
    "captions": [...],          // 可选，幻灯片描述
    "context_evidence": [...]   // 可选，上下文信息
  }
]
```

## 幻灯片特点

SlideVQA 的幻灯片通常包含：
- **标题和副标题**
- **项目符号列表**
- **图表和图形**
- **图像和插图**
- **表格数据**
- **演示文稿结构**

## 验证数据集

```bash
# 从项目根目录运行
python -c "
from data_pipeline.datasets.slidevqa import SlideVQADataset
from pathlib import Path
dataset = SlideVQADataset(Path('data_pipeline/data/slidevqa'), 'train')
print(f'Dataset loaded successfully with {len(dataset)} samples')
print('Sample:', dataset[0])
"
```

## 注意事项

- 幻灯片通常包含结构化的演示内容
- 支持的图像格式：.jpg, .png, .jpeg
- 建议启用 OCR 功能以提取幻灯片中的文本
- 幻灯片可能包含多种视觉元素（文本、图表、图像等）