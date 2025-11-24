# R³ 数据集目录结构

本目录包含了 R³ 项目支持的所有数据集的标准目录结构。

## 目录结构概览

```
data_pipeline/data/
├── textvqa/           # TextVQA 数据集
│   ├── README.md      # 数据集说明文档
│   ├── images/        # 图像文件目录
│   ├── textvqa_train.json
│   ├── textvqa_val.json
│   └── textvqa_test.json
├── mp_docvqa/         # MP-DocVQA 数据集
│   ├── README.md
│   ├── images/        # 文档页面图像
│   ├── mp_docvqa_train.json
│   └── mp_docvqa_val.json
├── infovqa/           # InfoVQA 数据集
│   ├── README.md
│   ├── images/        # 信息图表图像
│   ├── infovqa_train.json
│   └── infovqa_val.json
├── chartqa/           # ChartQA 数据集
│   ├── README.md
│   ├── charts/        # 图表图像 (注意目录名)
│   ├── chartqa_train.json
│   └── chartqa_val.json
├── docvqa/            # DocVQA 数据集
│   ├── README.md
│   ├── documents/     # 文档图像 (注意目录名)
│   ├── docvqa_train.json
│   └── docvqa_val.json
├── slidevqa/          # SlideVQA 数据集
│   ├── README.md
│   ├── images/        # 幻灯片图像
│   ├── slidevqa_train.json
│   └── slidevqa_val.json
└── mtvqa/             # MTVQA 数据集
    ├── README.md
    ├── images/        # 图像文件
    ├── videos/        # 视频文件
    ├── mtvqa_train.json
    └── mtvqa_val.json
```

## 快速开始

### 1. 下载/整理数据（推荐）
```bash
# 从项目根目录运行，默认同时准备 InfoVQA / MP-DocVQA / MTVQA
python scripts/download_datasets.py --datasets infovqa mp_docvqa mtvqa \
  --output-root data_pipeline/data \
  --prefer-symlink   # 可选：用符号链接避免大文件复制

# 仅下载 InfoVQA
python scripts/download_datasets.py --datasets infovqa --output-root data_pipeline/data
```

> 提示：MP-DocVQA 官方下载需要登录 RRC，脚本支持 `--repo-overrides mp_docvqa=<hf_repo>` 或 `--local-sources mp_docvqa=/path/to/your/manual/download` 复用本地文件。

## 数据集下载指南

每个数据集目录都包含详细的 `README.md` 文件，说明：
- 官方下载链接
- 数据放置步骤
- 文件格式要求
- 验证方法

### 重要注意事项

1. **图像目录名称**：
   - TextVQA, MP-DocVQA, InfoVQA, SlideVQA: `images/`
   - ChartQA: `charts/`
   - DocVQA: `documents/`

2. **标注文件命名**：
   - 必须严格按照 `{dataset_type}_{split}.json` 格式
   - 例如：`textvqa_train.json`, `chartqa_val.json`

3. **自动类型检测**：
   - 工具会根据文件名和目录结构自动识别数据集类型
   - 无需手动指定数据集类型

## 使用数据集

### 构建伪文本检索库

```bash
# 处理单个数据集
python build_pseudo_text.py \
  --dataset_root data_pipeline/data/textvqa \
  --split train \
  --output artifacts/textvqa_pseudo_text.jsonl \
  --enable_ocr

# 批量处理所有数据集
python build_pseudo_text.py \
  --dataset_roots data_pipeline/data/textvqa data_pipeline/data/mp_docvqa data_pipeline/data/infovqa data_pipeline/data/chartqa data_pipeline/data/docvqa data_pipeline/data/slidevqa data_pipeline/data/mtvqa \
  --split train \
  --output artifacts/combined_pseudo_text.jsonl \
  --enable_ocr \
  --caption_model Qwen/Qwen2-VL-2B-Instruct
```

### 训练模型

```bash
python train_r3.py \
  --config configs/default.yaml \
  --device cuda \
  --output_dir checkpoints/r3_lora
```

## 数据集特点

- **TextVQA**: 文本密集的自然图像，需要 OCR 能力
- **MP-DocVQA**: 多页文档，支持跨页推理
- **InfoVQA**: 复杂信息图表，需要多模态理解
- **ChartQA**: 统计图表，需要数值推理能力
- **DocVQA**: 结构化文档，需要布局理解
- **SlideVQA**: 演示文稿，需要理解演示结构
- **MTVQA**: 多模态文本-视频问答，需要处理图像和视频内容

## 故障排除

如果遇到问题，请：
1. 检查目录结构是否正确
2. 验证文件命名是否符合规范
3. 运行验证脚本检查数据集完整性
4. 查看各数据集的 README.md 文件
