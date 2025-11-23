# MP-DocVQA Dataset

MP-DocVQA (Multi-Page Document Visual Question Answering) 是一个多页文档视觉问答数据集，支持跨页面的信息检索和推理。

## 数据集结构

```
mp_docvqa/
├── mp_docvqa_train.json    # 训练集标注文件
├── mp_docvqa_val.json      # 验证集标注文件
├── mp_docvqa_test.json     # 测试集标注文件（可选）
└── images/                 # 文档页面图像目录
    ├── doc1_page0.png
    ├── doc1_page1.png
    ├── doc2_page0.png
    └── ...
```

## 下载说明

1. **官方网站**: https://rrc.cvc.uab.es/?ch=17
2. **数据下载**:
   - 标注文件: 下载 train/val 的 JSON 标注文件
   - 图像文件: 下载多页文档图像数据

## 数据放置步骤

1. 将下载的标注文件重命名并放置：
   ```bash
   # 将官方标注文件重命名为标准格式
   cp train.json mp_docvqa_train.json
   cp val.json mp_docvqa_val.json
   cp test.json mp_docvqa_test.json  # 如果有测试集
   ```

2. 将文档页面图像解压到 `images/` 目录：
   ```bash
   # 解压图像文件到 images 目录
   unzip documents.zip -d images/
   ```

## 标注文件格式

```json
[
  {
    "id": "doc1_page0",
    "doc_id": "doc1",
    "page": 0,
    "question": "What is the title of this document?",
    "answer": "Annual Report 2023",
    "image": "doc1_page0.png",
    "ocr_tokens": [...],      // 可选，OCR 标注
    "captions": [...],        // 可选，页面描述
    "context_evidence": [...]  // 可选，跨页上下文
  }
]
```

## 特殊功能

MP-DocVQA 支持 **Page-as-Evidence** 功能：
- 自动提取相邻页面的 OCR 和描述作为上下文证据
- 支持跨页面的信息检索和推理
- 在构建伪文本时会包含邻近页面的信息

## 验证数据集

```bash
# 从项目根目录运行
python -c "
from data_pipeline.datasets.mp_docvqa import MPDocVQADataset
from pathlib import Path
dataset = MPDocVQADataset(Path('data_pipeline/data/mp_docvqa'), 'train')
print(f'Dataset loaded successfully with {len(dataset)} samples')
print('Sample:', dataset[0])
"
```

## 注意事项

- 文档页面图像通常命名为 `{doc_id}_page{page_num}.png`
- 支持自动跨页上下文提取
- 确保同一文档的页面按顺序编号