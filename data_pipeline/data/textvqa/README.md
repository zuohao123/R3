# TextVQA Dataset

TextVQA 是一个需要阅读和理解图像中文本的视觉问答数据集。

## 数据集结构

```
textvqa/
├── textvqa_train.json      # 训练集标注文件
├── textvqa_val.json        # 验证集标注文件
├── textvqa_test.json       # 测试集标注文件（可选）
└── images/                 # 图像文件目录
    ├── image1.jpg
    ├── image2.png
    └── ...
```

## 下载说明

1. **官方网站**: https://textvqa.org/
2. **数据下载**: 
   - 标注文件: 下载 train/val/test 的 JSON 标注文件
   - 图像文件: 下载对应的图像数据

## 数据放置步骤

1. 将下载的标注文件重命名并放置：
   ```bash
   # 将官方标注文件重命名为标准格式
   cp train_questions_vqa_format.json textvqa_train.json
   cp val_questions_vqa_format.json textvqa_val.json
   cp test_questions_vqa_format.json textvqa_test.json  # 如果有测试集
   ```

2. 将图像文件解压到 `images/` 目录：
   ```bash
   # 解压图像文件到 images 目录
   unzip train_images.zip -d images/
   unzip val_images.zip -d images/
   ```

## 标注文件格式

```json
{
  "data": [
    {
      "id": "sample_id",
      "question": "What is written on the sign?",
      "answer": "Stop",
      "image": "image_filename.jpg",
      "ocr_tokens": [...],  // 可选，OCR 标注
      "captions": [...]     // 可选，图像描述
    }
  ]
}
```

## 验证数据集

完成数据放置后，可以运行以下命令验证：

```bash
# 从项目根目录运行
python -c "
from data_pipeline.datasets.textvqa import TextVQADataset
from pathlib import Path
dataset = TextVQADataset(Path('data_pipeline/data/textvqa'), 'train')
print(f'Dataset loaded successfully with {len(dataset)} samples')
print('Sample:', dataset[0])
"
```

## 注意事项

- 确保图像文件名与标注文件中的 `image` 字段匹配
- 支持的图像格式：.jpg, .png, .jpeg
- 如果缺少 OCR 标注，可以使用 `--enable_ocr` 参数自动生成