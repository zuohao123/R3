# DocVQA Dataset

DocVQA 是一个基于文档图像的视觉问答数据集，专注于理解文档中的文本和布局信息。

## 数据集结构

```
docvqa/
├── docvqa_train.json       # 训练集标注文件
├── docvqa_val.json         # 验证集标注文件
├── docvqa_test.json        # 测试集标注文件（可选）
└── documents/              # 文档图像目录（注意：目录名为 documents）
    ├── document1.png
    ├── document2.jpg
    └── ...
```

## 下载说明

1. **官方网站**: https://www.docvqa.org/
2. **数据下载**:
   - 标注文件: 下载 train/val 的 JSON 标注文件
   - 文档图像: 下载文档图像数据

## 数据放置步骤

1. 将下载的标注文件重命名并放置：
   ```bash
   # 将官方标注文件重命名为标准格式
   cp train_v1.0.json docvqa_train.json
   cp val_v1.0.json docvqa_val.json
   cp test_v1.0.json docvqa_test.json  # 如果有测试集
   ```

2. 将文档图像解压到 `documents/` 目录：
   ```bash
   # 解压图像文件到 documents 目录
   unzip documents.zip -d documents/
   ```

## 标注文件格式

```json
[
  {
    "id": "sample_id",
    "question": "What is the date on this document?",
    "answer": "2023-01-01",
    "image": "document1.png",
    "ocr_tokens": [...],      // 可选，OCR 标注
    "layout": {...},          // 可选，文档布局信息
    "captions": [...]         // 可选，文档描述
  }
]
```

## 文档类型

DocVQA 包含多种文档类型：
- **表单** (Forms)
- **发票** (Invoices)
- **收据** (Receipts)
- **报告** (Reports)
- **信件** (Letters)
- **证书** (Certificates)

## 验证数据集

```bash
# 从项目根目录运行
python -c "
from data_pipeline.datasets.docvqa import DocVQADataset
from pathlib import Path
dataset = DocVQADataset(Path('data_pipeline/data/docvqa'), 'train')
print(f'Dataset loaded successfully with {len(dataset)} samples')
print('Sample:', dataset[0])
"
```

## 注意事项

- **重要**: 图像目录名必须是 `documents/`，不是 `images/`
- 文档通常包含结构化的文本信息
- 支持的图像格式：.jpg, .png, .jpeg
- 建议启用 OCR 功能以提取文档中的文本
- 文档布局信息对理解文档结构很重要