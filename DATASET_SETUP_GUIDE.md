# 数据集目录结构指南

本文档说明了如何正确组织各个数据集的目录结构，以便与 R³ 项目的数据处理工具兼容。

## 推荐的总体目录结构

```
datasets/
├── textvqa/
│   ├── textvqa_train.json
│   ├── textvqa_val.json
│   ├── textvqa_test.json
│   └── images/
│       ├── image1.jpg
│       ├── image2.png
│       └── ...
├── mp_docvqa/
│   ├── mp_docvqa_train.json
│   ├── mp_docvqa_val.json
│   └── images/
│       ├── doc1_page0.png
│       ├── doc1_page1.png
│       └── ...
├── infovqa/
│   ├── infovqa_train.json
│   ├── infovqa_val.json
│   └── images/
│       ├── infographic1.png
│       ├── infographic2.jpg
│       └── ...
├── chartqa/
│   ├── chartqa_train.json
│   ├── chartqa_val.json
│   └── charts/
│       ├── chart1.png
│       ├── chart2.jpg
│       └── ...
├── docvqa/
│   ├── docvqa_train.json
│   ├── docvqa_val.json
│   └── documents/
│       ├── document1.png
│       ├── document2.jpg
│       └── ...
└── slidevqa/
    ├── slidevqa_train.json
    ├── slidevqa_val.json
    └── images/
        ├── slide1.png
        ├── slide2.jpg
        └── ...
```

## 各数据集详细要求

### 1. TextVQA
**目录结构：**
```
textvqa/
├── textvqa_train.json      # 训练集标注文件
├── textvqa_val.json        # 验证集标注文件
├── textvqa_test.json       # 测试集标注文件（可选）
└── images/                 # 图像目录
    ├── image1.jpg
    └── ...
```

**备选图像目录：**
- `textvqa/images/` （推荐）
- `textvqa_image/` （父目录下）

**标注文件格式：**
```json
{
  "data": [
    {
      "id": "sample_id",
      "question": "What is written on the sign?",
      "answer": "Stop",
      "image": "image_filename.jpg",
      "ocr_tokens": [...],  // 可选
      "captions": [...]     // 可选
    }
  ]
}
```

### 2. MP-DocVQA
**目录结构：**
```
mp_docvqa/
├── mp_docvqa_train.json
├── mp_docvqa_val.json
└── images/
    ├── doc1_page0.png
    ├── doc1_page1.png
    └── ...
```

**标注文件格式：**
```json
[
  {
    "id": "doc1_page0",
    "doc_id": "doc1",
    "page": 0,
    "question": "What is the title?",
    "answer": "Annual Report",
    "image": "doc1_page0.png",
    "ocr_tokens": [...],
    "captions": [...]
  }
]
```

### 3. InfoVQA
**目录结构：**
```
infovqa/
├── infovqa_train.json
├── infovqa_val.json
└── images/
    ├── infographic1.png
    └── ...
```

**标注文件格式：**
```json
[
  {
    "id": "sample_id",
    "question": "What percentage is shown?",
    "answer": "75%",
    "image": "infographic1.png",
    "ocr_tokens": [...],
    "captions": [...]
  }
]
```

### 4. ChartQA
**目录结构：**
```
chartqa/
├── chartqa_train.json
├── chartqa_val.json
└── charts/                 # 注意：图像目录名为 "charts"
    ├── chart1.png
    └── ...
```

**标注文件格式：**
```json
[
  {
    "id": "sample_id",
    "question": "What is the highest value?",
    "answer": "100",
    "image": "chart1.png",
    "chart_type": "bar_chart"  // 可选
  }
]
```

### 5. DocVQA
**目录结构：**
```
docvqa/
├── docvqa_train.json
├── docvqa_val.json
└── documents/              # 注意：图像目录名为 "documents"
    ├── document1.png
    └── ...
```

**标注文件格式：**
```json
[
  {
    "id": "sample_id",
    "question": "What is the date?",
    "answer": "2023-01-01",
    "image": "document1.png",
    "ocr_tokens": [...],
    "layout": {...}           // 可选
  }
]
```

### 6. SlideVQA
**目录结构：**
```
slidevqa/
├── slidevqa_train.json
├── slidevqa_val.json
└── images/
    ├── slide1.png
    └── ...
```

**标注文件格式：**
```json
[
  {
    "id": "sample_id",
    "question": "What is the main topic?",
    "answer": "Machine Learning",
    "image": "slide1.png",
    "ocr_tokens": [...],
    "captions": [...]
  }
]
```

## 下载和组织数据集的步骤

### 1. 创建数据集根目录
```bash
mkdir -p datasets
cd datasets
```

### 2. 下载各个数据集
根据数据集的官方下载方式，将数据下载到对应目录：

```bash
# 示例：下载 TextVQA
mkdir textvqa
cd textvqa
# 下载标注文件和图像...
cd ..

# 示例：下载 MP-DocVQA
mkdir mp_docvqa
cd mp_docvqa
# 下载标注文件和图像...
cd ..
```

### 3. 验证目录结构
使用以下命令验证目录结构是否正确：

```bash
# 检查标注文件
find datasets/ -name "*.json" | sort

# 检查图像目录
find datasets/ -type d -name "images" -o -name "charts" -o -name "documents" | sort
```

### 4. 使用构建工具
验证目录结构正确后，可以使用构建工具：

```bash
# 处理单个数据集
python build_pseudo_text.py \
  --dataset_root datasets/textvqa \
  --split train \
  --output artifacts/textvqa_pseudo_text.jsonl

# 批量处理所有数据集
python build_pseudo_text.py \
  --dataset_roots datasets/textvqa datasets/mp_docvqa datasets/infovqa datasets/chartqa datasets/docvqa datasets/slidevqa \
  --split train \
  --output artifacts/combined_pseudo_text.jsonl \
  --enable_ocr
```

## 注意事项

1. **标注文件命名**：必须严格按照 `{dataset_type}_{split}.json` 格式命名
2. **图像目录名称**：不同数据集的图像目录名称可能不同（images/charts/documents）
3. **图像文件格式**：支持 .jpg, .png, .jpeg 格式
4. **路径分隔符**：使用正斜杠 `/`，工具会自动处理不同操作系统的路径差异
5. **文件编码**：JSON 文件应使用 UTF-8 编码

## 常见问题

**Q: 如果我的数据集目录结构与标准不同怎么办？**
A: 可以通过符号链接或重命名来调整目录结构，或者修改对应的数据集适配器代码。

**Q: 可以只下载部分数据集吗？**
A: 可以，工具支持处理任意数量的数据集，不需要下载全部。

**Q: 如果标注文件中缺少某些字段怎么办？**
A: 工具会自动处理缺失字段，并在需要时使用 OCR 和 Caption 模型补充信息。