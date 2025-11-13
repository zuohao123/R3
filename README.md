# R³: Robust Multimodal Retrieval-Reconstruction-Reasoning Framework

R³ (Retrieval, Reconstruction, Reasoning) 是一个专为**部分模态损坏(PMC)**场景设计的鲁棒多模态问答框架，通过"腐蚀感知 + 伪文本检索 + 选择性重建"实现增强的鲁棒性和可解释性。

## 🏗️ 框架架构概览

R³框架通过五个核心阶段实现鲁棒的多模态问答：

graph TB
    subgraph "阶段1: 数据流水线"
        A1[原始数据集] --> A2[腐蚀模拟器]
        A2 --> A3[伪文本生成器]
        A3 --> A4[特征构建器]
        A4 --> A5[PMC数据模块]
    end
    
    subgraph "阶段2: 知识检索"
        B1[语料库构建] --> B2[伪文本检索器]
        B2 --> B3[一致性过滤器]
    end
    
    subgraph "阶段3: 自适应重建"
        C1[自适应门控器] --> C2[三路径增强器]
        C2 --> C3[Qwen3-VL适配器]
    end
    
    subgraph "阶段4: 训练优化"
        D1[训练器] --> D2[一致性正则器]
        D2 --> D3[联合损失优化]
    end
    
    subgraph "阶段5: 评估分析"
        E1[PMC评估器] --> E2[幻觉检测]
        E2 --> E3[结果导出器]
    end
    
    A5 --> D1
    B3 --> C1
    C3 --> D1
    D3 --> E1

## 📦 环境依赖

基于 [`requirements.txt`](./requirements.txt)：

torch>=2.1          # 深度学习框架
pyyaml>=6.0         # 配置文件解析  
numpy>=1.24         # 数值计算
pillow>=10.0        # 图像处理
transformers>=4.39  # 预训练模型支持

安装依赖：
pip install -r requirements.txt

## 🚀 快速开始

### 1. 配置数据集
# 编辑配置文件
vim configs/default.yaml
# 确保 dataset.root 指向正确的数据集目录

### 2. 构建检索索引
# 生成伪文本语料库
python scripts/build_index.py --input data/samples.jsonl --output runs/index

### 3. 训练模型
# 启动训练流程
python scripts/train.py \
    --config configs/default.yaml \
    --index runs/index/corpus.jsonl \
    --device cuda \
    --epochs 10 \
    --batch-size 8

### 4. 模型评估
# PMC场景评估
python eval/evaluate_pmc.py --model_path runs/model.pt

# 完整模态基线对比
python eval/evaluate_full.py --model_path runs/model.pt

# 结果导出
python scripts/export_results.py --input runs/results.json

## 🔄 详细架构流程

### 阶段1️⃣: 数据预处理与腐蚀模拟

**核心文件**: [`data_pipeline/`](./data_pipeline/)

graph LR
    subgraph "数据加载"
        A1[TextVQA/ChartQA/DocVQA] --> A2[base_dataset.py]
        A2 --> A3["{id, question, image_path, answer, extra}"]
    end
    
    subgraph "腐蚀模拟"
        A3 --> B1[simulator.py]
        B1 --> B2["视觉: blur(30%), occlude(30%), crop(20%)"]
        B1 --> B3["文本: OCR noise(40%) → <UNK>"]
        B2 --> B4[corruption_report]
        B3 --> B4
    end
    
    subgraph "特征构建"
        A3 --> C1[simple_feature_builder.py]
        B4 --> C1
        C1 --> C2["question_tokens[batch×64×512]"]
        C1 --> C3["vision_tokens[batch×32×512]"]
    end
    
    subgraph "伪文本生成"
        A3 --> D1[tokenizer.py]
        D1 --> D2["['<OCR> text </OCR>', '<CAP> caption </CAP>']"]
    end
    
    subgraph "数据模块集成"
        C2 --> E1[pmc_datamodule.py]
        C3 --> E1
        D2 --> E1
        B4 --> E1
        E1 --> E2["{full_sample, corrupted_sample, pseudo_text, corruption_report}"]
    end

**关键实现**:
- **数据加载**: [`base_dataset.py`](./data_pipeline/datasets/base_dataset.py) 定义统一接口，[`textvqa.py`](./data_pipeline/datasets/textvqa.py) / [`chartqa.py`](./data_pipeline/datasets/chartqa.py) / [`docvqa.py`](./data_pipeline/datasets/docvqa.py) 实现具体数据集
- **腐蚀模拟**: [`simulator.py`](./data_pipeline/corruption/simulator.py) 基于概率配置实现视觉/文本腐蚀
- **特征构建**: [`simple_feature_builder.py`](./data_pipeline/features/simple_feature_builder.py) 生成固定维度的张量特征
- **伪文本生成**: [`tokenizer.py`](./data_pipeline/pseudo_text/tokenizer.py) 将多模态内容转换为文本片段
- **数据模块**: [`pmc_datamodule.py`](./data_pipeline/dataloaders/pmc_datamodule.py) 统一封装，支持批处理和数据加载

### 阶段2️⃣: 知识检索与一致性过滤

**核心文件**: [`retrieval/`](./retrieval/)

graph LR
    subgraph "语料库构建"
        A1[伪文本样本] --> A2[build_corpus.py]
        A2 --> A3[corpus.jsonl]
    end
    
    subgraph "相似度检索"
        A3 --> B1[pseudo_text_retriever.py]
        B2[question + pseudo_text + uncertainty] --> B1
        B1 --> B3["Token重叠度 + 噪声阈值"]
        B3 --> B4[Top-K候选文档]
    end
    
    subgraph "一致性过滤"
        B4 --> C1[consistency_filter.py]
        C1 --> C2["过滤矛盾和无关证据"]
        C2 --> C3[高质量检索结果]
    end

**关键实现**:
- **语料构建**: [`build_corpus.py`](./retrieval/indexer/build_corpus.py) + [`scripts/build_index.py`](./scripts/build_index.py)
- **检索引擎**: [`pseudo_text_retriever.py`](./retrieval/query/pseudo_text_retriever.py) 基于token重叠度和动态Top-K策略
- **一致性过滤**: [`consistency_filter.py`](./retrieval/filters/consistency_filter.py) 去除矛盾和低质量证据

### 阶段3️⃣: 自适应重建与推理

**核心文件**: [`model/`](./model/)

graph TB
    subgraph "输入处理"
        A1[question_tokens] --> B1[自适应门控器]
        A2[vision_tokens] --> B1
        A3[检索证据] --> B1
        A4[腐蚀报告] --> B1
    end
    
    subgraph "门控逻辑"
        B1 --> B2["text_gate = 0.65 - 0.4×uncertainty + 0.2×retrieval_quality"]
        B1 --> B3["memory_gate = 0.65 × (0.5 + retrieval_quality)"]
        B1 --> B4["imputation_gate = 0.65 + 0.5×uncertainty - 0.2×retrieval_quality"]
    end
    
    subgraph "三路径增强"
        B2 --> C1["文本前缀路径<br/>prefix_encoder(question[:64])"]
        B3 --> C2["跨模态记忆<br/>memory_adapter(question[:32])"]
        B4 --> C3["潜空间填充<br/>learnable_embeddings[16×512]"]
    end
    
    subgraph "主干推理"
        C1 --> D1[Qwen3VL适配器]
        C2 --> D1
        C3 --> D1
        D1 --> D2["answer_logits + loss + gate_values"]
    end

**关键实现**:
- **自适应门控**: [`adaptive_gate.py`](./model/modules/adaptive_gate.py) 根据腐蚀不确定性和检索质量动态调整权重
- **三路径增强**: [`tri_path_enhancer.py`](./model/modules/tri_path_enhancer.py) 实现文本前缀、跨模记忆、潜空间填充
- **模型适配器**: [`qwen3_vl_adapter.py`](./model/backbones/qwen3_vl_adapter.py) 统一封装门控器、增强器、主干网络
- **模型构建**: [`builders.py`](./model/builders.py) 提供模型实例化工厂函数

### 阶段4️⃣: 一致性约束与联合训练

**核心文件**: [`train/`](./train/) + [`model/losses/`](./model/losses/)

graph LR
    subgraph "双路径前向"
        A1[corrupted_sample] --> B1[模型前向传播]
        A2[full_sample] --> B2[模型前向传播]
        A3[检索结果] --> B1
        A3 --> B2
    end
    
    subgraph "损失计算"
        B1 --> C1[outputs_corrupted]
        B2 --> C2[outputs_full]
        C1 --> C3["QA损失<br/>(CrossEntropy)"]
        C1 --> D1[consistency_regularizer.py]
        C2 --> D1
        D1 --> D2["一致性损失<br/>(KL Divergence)"]
    end
    
    subgraph "优化更新"
        C3 --> E1["总损失 = QA + Consistency"]
        D2 --> E1
        E1 --> E2