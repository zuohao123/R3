# MTVQA Dataset

MTVQA (Multi-modal Text-Video Question Answering) 是一个多模态文本-视频问答数据集，结合了图像、视频和文本信息。

## 数据集结构

```
mtvqa/
├── mtvqa_train.json        # 训练集标注文件
├── mtvqa_val.json          # 验证集标注文件
├── mtvqa_test.json         # 测试集标注文件（可选）
├── images/                 # 图像文件目录
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── videos/                 # 视频文件目录
    ├── video1.mp4
    ├── video2.avi
    └── ...
```

## 下载说明

1. **数据来源**: 根据具体的 MTVQA 数据集来源进行下载
2. **数据下载**:
   - 标注文件: 下载 train/val 的 JSON 标注文件
   - 图像文件: 下载相关图像数据
   - 视频文件: 下载相关视频数据

### 一键整理（可选）
```bash
python scripts/download_datasets.py --datasets mtvqa --output-root data_pipeline/data
```

## 数据放置步骤

1. 将下载的标注文件重命名并放置：
   ```bash
   # 将官方标注文件重命名为标准格式
   cp train.json mtvqa_train.json
   cp val.json mtvqa_val.json
   cp test.json mtvqa_test.json  # 如果有测试集
   ```

2. 将图像文件解压到 `images/` 目录：
   ```bash
   # 解压图像文件到 images 目录
   unzip images.zip -d images/
   ```

3. 将视频文件解压到 `videos/` 目录：
   ```bash
   # 解压视频文件到 videos 目录
   unzip videos.zip -d videos/
   ```

## 标注文件格式

```json
[
  {
    "id": "sample_id",
    "question": "What happens in this video?",
    "answer": "A person is walking",
    "image": "image1.jpg",
    "video": "video1.mp4",
    "ocr_tokens": [...],          // 可选，OCR 标注
    "captions": [...],            // 可选，图像/视频描述
    "video_frames": [...],        // 可选，视频帧信息
    "temporal_info": {            // 可选，时序信息
      "start_time": 0.0,
      "end_time": 10.0,
      "key_frames": [1.5, 3.2, 7.8]
    }
  }
]
```

## 数据集特点

MTVQA 的多模态特性：
- **图像信息**: 静态图像内容理解
- **视频信息**: 动态视频内容和时序关系
- **文本信息**: OCR 提取的文本内容
- **时序信息**: 视频中的时间序列关系
- **跨模态推理**: 需要结合多种模态信息进行推理

## 验证数据集

```bash
# 从项目根目录运行
python -c "
from data_pipeline.datasets.mtvqa import MTVQADataset
from pathlib import Path
dataset = MTVQADataset(Path('data_pipeline/data/mtvqa'), 'train')
print(f'Dataset loaded successfully with {len(dataset)} samples')
print('Sample:', dataset[0])
"
```

## 注意事项

- 支持同时处理图像和视频文件
- 视频文件格式：.mp4, .avi, .mov
- 图像文件格式：.jpg, .png, .jpeg
- 建议启用 OCR 功能以提取图像/视频中的文本
- 视频处理可能需要额外的计算资源
- 时序信息对于理解视频内容很重要

## 视频处理说明

对于包含视频的样本：
1. 可以提取关键帧进行图像处理
2. 可以使用视频理解模型进行内容分析
3. 时序信息有助于理解动作和事件序列
4. OCR 可以应用于视频帧中的文本提取
