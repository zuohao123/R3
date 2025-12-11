# R³ 训练与评测快速手册

## 0. 环境与资源
- 硬件：4×V100 32GB（可扩展 8 卡），CPU 64 核，内存 128GiB。
- 依赖：`pip install -r requirements.txt`（包含 transformers、peft、bitsandbytes 等）。
- 模型：`./models/Qwen3-VL-8B-Instruct` 本地权重。
- 数据：`./data_pipeline/data/{docvqa,infovqa,chartqa}`（train/val），伪文本语料 `./artifacts/pseudo_text_all_train.jsonl`。

## 1. 配置概览
- 默认批次：`configs/default.yaml` 设 `batch_size=2`，`grad_accum_steps=4`（4 卡全局 batch≈32），4bit+LoRA，关闭 grad checkpoint。
- Stage2/Stage3：`configs/stage2.yaml`（轻度腐蚀/一致性），`configs/stage3.yaml`（重度腐蚀，一致性更高）。

## 2. 训练命令（后台可持续，含 max_steps）
通用模板（先建目录）：
```bash
mkdir -p <out_dir>
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup torchrun --nproc_per_node=4 train_r3.py \
  --config <config_yaml> \
  --device cuda \
  --output_dir <out_dir> \
  --log_file <out_dir>/train.log \
  --max_steps 800 \
  > <out_dir>/console.log 2>&1 &
```
示例：
- Stage1（干净多任务）：
  ```bash
  mkdir -p checkpoints/stage1_clean_mp4
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nohup torchrun --nproc_per_node=4 train_r3.py \
    --config configs/default.yaml \
    --device cuda \
    --output_dir checkpoints/stage1_clean_mp4 \
    --log_file checkpoints/stage1_clean_mp4/train.log \
    --max_steps 800 \
    > checkpoints/stage1_clean_mp4/console.log 2>&1 &
  ```
- Stage2（轻 PMC）：
  ```bash
  mkdir -p checkpoints/stage2_light_pmc
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nohup torchrun --nproc_per_node=4 train_r3.py \
    --config configs/stage2.yaml \
    --device cuda \
    --output_dir checkpoints/stage2_light_pmc \
    --log_file checkpoints/stage2_light_pmc/train.log \
    --max_steps 800 \
    > checkpoints/stage2_light_pmc/console.log 2>&1 &
  ```
- Stage3（重 PMC）：
  ```bash
  mkdir -p checkpoints/stage3_heavy_pmc
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nohup torchrun --nproc_per_node=4 train_r3.py \
    --config configs/stage3.yaml \
    --device cuda \
    --output_dir checkpoints/stage3_heavy_pmc \
    --log_file checkpoints/stage3_heavy_pmc/train.log \
    --max_steps 800 \
    > checkpoints/stage3_heavy_pmc/console.log 2>&1 &
  ```

> 说明：`max_steps` 可按需调整（如 800/1000）。LossLogger 已写入 train.log，进度条等在 console.log。

## 3. 日志与监控
- TensorBoard：`tensorboard --logdir checkpoints/runs_default`（或 runs_stage2/3），浏览器查看 loss 曲线。
- 文本：`tail -f <out_dir>/train.log`（包含 loss），`tail -f <out_dir>/console.log`（进度条/警告）。
- 显存：`nvidia-smi -l 5`。

## 4. 评测（示例）
如有评测脚本 `evaluate_r3.py`，可按需运行：
```bash
torchrun --nproc_per_node=4 evaluate_r3.py \
  --config configs/default.yaml \
  --device cuda \
  --ckpt_dir checkpoints/stage1_clean_mp4 \
  --split val
```
（根据实际脚本参数调整，评测 Stage2/3 时切换对应配置与权重。）

### 4.1 评测场景设计
- **干净集性能**：验证 R³ 是否在无腐蚀情况下不退化甚至提升（对比基座 Qwen3-VL）。
- **部分缺模态/腐蚀**：测试不同缺失比例/类型（遮挡、模糊、OCR 截断、伪文本 drop）。准备对应 corruption 配置或数据版本。
- **跨模态一致性/幻觉率**：统计回答与提供证据的一致性，记录 hallucination rate。

### 4.2 基线对比
- **基座模型（无 R³）**：加载 Qwen3-VL-8B，关闭 retrieval/consistency/corruption，命令示例：
  ```bash
  torchrun --nproc_per_node=4 evaluate_r3.py \
    --config configs/default.yaml \
    --device cuda \
    --ckpt_dir ./models/Qwen3-VL-8B-Instruct \
    --split val \
    --disable_retrieval --disable_consistency --disable_corruption
  ```
  （假设评测脚本支持这些开关；否则可复制配置，手动将 `enable_*` 设为 false。）

### 4.3 缺模态/腐蚀评测
- **轻腐蚀配置**（Stage2，drop 0.1，适中遮挡/模糊）：
  ```bash
  torchrun --nproc_per_node=4 evaluate_r3.py \
    --config configs/stage2.yaml \
    --device cuda \
    --ckpt_dir checkpoints/stage2_light_pmc \
    --split val
  ```
- **重腐蚀配置**（Stage3，drop 0.35，遮挡/模糊更强，一致性权重 0.7）：
  ```bash
  torchrun --nproc_per_node=4 evaluate_r3.py \
    --config configs/stage3.yaml \
    --device cuda \
    --ckpt_dir checkpoints/stage3_heavy_pmc \
    --split val
  ```
- **基座 vs R³**：在相同腐蚀配置下，分别用基座权重和 R³ 权重评测，比较准确率/幻觉率。

### 4.4 消融实验
- **无检索**：关闭 `enable_retrieval`（配置或命令开关），评测：
  ```bash
  torchrun --nproc_per_node=4 evaluate_r3.py \
    --config configs/stage2.yaml \
    --device cuda \
    --ckpt_dir checkpoints/stage2_light_pmc \
    --split val \
    --disable_retrieval
  ```
- **无一致性**：关闭 `enable_consistency`，查看 hallucination 是否上升：
  ```bash
  torchrun --nproc_per_node=4 evaluate_r3.py \
    --config configs/stage2.yaml \
    --device cuda \
    --ckpt_dir checkpoints/stage2_light_pmc \
    --split val \
    --disable_consistency
  ```
- **无腐蚀模拟**：关闭 `enable_corruption`，检验模拟器对鲁棒性的贡献：
  ```bash
  torchrun --nproc_per_node=4 evaluate_r3.py \
    --config configs/stage2.yaml \
    --device cuda \
    --ckpt_dir checkpoints/stage2_light_pmc \
    --split val \
    --disable_corruption
  ```
- **组件组合**：可依次关闭/开启以上模块，形成 (R)、(C)、(R+C)、(全开) 四组，对比鲁棒性与幻觉率。

> 若评测脚本无命令行开关，可复制配置文件（如 `configs/ablate_no_retrieval.yaml`）并手动设置 `enable_*` 标志。

## 5. 可选调优
- 显存有余：可尝试 `batch_size=2, grad_accum_steps=6`（全局≈48），或 8 卡训练（`--nproc_per_node=8`，适当调小 grad_accum 保持全局 batch）。
- 显存不足：减小 `vision_tokens`（如 4），或在配置中添加 `max_seq_length: 768`。
- 日志频率：`training.log_interval` 控制 logging 步长，默认 10。

## 6. 常见问题
- `--max_steps` 未被识别：已在脚本支持，确保使用最新版本。
- 日志缺少 loss：LossLogger 已启用，查看 train.log；若为空，确认命令使用最新脚本并重启进程。
- 目录不存在：训练前 `mkdir -p <out_dir>`。
- dtype 报错：默认关闭 grad checkpoint，4bit 计算用 fp32；若仍异常，重启进程加载最新代码。

## 7. 命令总览大表

| 任务/场景 | 目标 | 配置/权重 | 命令 | 备注 |
| --- | --- | --- | --- | --- |
| 训练 | Stage1 干净多任务 | `configs/default.yaml` → `checkpoints/stage1_clean_mp4` | `mkdir -p checkpoints/stage1_clean_mp4 && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/stage1_clean_mp4 --log_file checkpoints/stage1_clean_mp4/train.log --max_steps 800 > checkpoints/stage1_clean_mp4/console.log 2>&1 &` | 4bit+LoRA，batch=2，accum=4，全局≈32 |
| 训练 | Stage2 轻腐蚀/一致性 | `configs/stage2.yaml` → `checkpoints/stage2_light_pmc` | `mkdir -p checkpoints/stage2_light_pmc && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 train_r3.py --config configs/stage2.yaml --device cuda --output_dir checkpoints/stage2_light_pmc --log_file checkpoints/stage2_light_pmc/train.log --max_steps 800 > checkpoints/stage2_light_pmc/console.log 2>&1 &` | 开启 corruption/retrieval/consistency |
| 训练 | Stage3 重腐蚀/一致性 | `configs/stage3.yaml` → `checkpoints/stage3_heavy_pmc` | `mkdir -p checkpoints/stage3_heavy_pmc && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 train_r3.py --config configs/stage3.yaml --device cuda --output_dir checkpoints/stage3_heavy_pmc --log_file checkpoints/stage3_heavy_pmc/train.log --max_steps 800 > checkpoints/stage3_heavy_pmc/console.log 2>&1 &` | 更高腐蚀/λ=0.7，accum=12 |
| 评测 | 干净集（R³） | `configs/default.yaml` + `ckpt=stage1` | `torchrun --nproc_per_node=4 evaluate_r3.py --config configs/default.yaml --device cuda --ckpt_dir checkpoints/stage1_clean_mp4 --split val` | 无腐蚀基线性能 |
| 评测 | 轻腐蚀（R³） | `configs/stage2.yaml` + `ckpt=stage2` | `torchrun --nproc_per_node=4 evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir checkpoints/stage2_light_pmc --split val` | 部分缺模态 |
| 评测 | 重腐蚀（R³） | `configs/stage3.yaml` + `ckpt=stage3` | `torchrun --nproc_per_node=4 evaluate_r3.py --config configs/stage3.yaml --device cuda --ckpt_dir checkpoints/stage3_heavy_pmc --split val` | 严重缺模态 |
| 基线对比 | 基座 Qwen3-VL（无 R³） | `configs/default.yaml` + `./models/Qwen3-VL-8B-Instruct` | `torchrun --nproc_per_node=4 evaluate_r3.py --config configs/default.yaml --device cuda --ckpt_dir ./models/Qwen3-VL-8B-Instruct --split val --disable_retrieval --disable_consistency --disable_corruption` | 若无命令开关，复制配置将 `enable_*` 置 false |
| 消融 | 关检索 | `configs/stage2.yaml` + `ckpt=stage2` | `torchrun --nproc_per_node=4 evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir checkpoints/stage2_light_pmc --split val --disable_retrieval` | 测检索贡献 |
| 消融 | 关一致性 | `configs/stage2.yaml` + `ckpt=stage2` | `torchrun --nproc_per_node=4 evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir checkpoints/stage2_light_pmc --split val --disable_consistency` | 观察幻觉率 |
| 消融 | 关腐蚀模拟 | `configs/stage2.yaml` + `ckpt=stage2` | `torchrun --nproc_per_node=4 evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir checkpoints/stage2_light_pmc --split val --disable_corruption` | 模拟器贡献 |
| 训练/存储 | 避免 safetensors 共享权重报错 | `train_r3.py` 默认 | `save_safetensors=False` 已在脚本设置，如遇相同报错请更新代码后重启训练 | 共享嵌入导致 safetensors 报错 |
| 资源监控 | TensorBoard | 日志目录 | `tensorboard --logdir checkpoints/runs_default` | runs_stage2/3 类似 |
| 资源监控 | 日志查看 | 输出目录 | `tail -f <out_dir>/train.log`；`tail -f <out_dir>/console.log` | LossLogger 写入 train.log |
| 评测（全精+长视觉） | 基座官方风格 | `configs/eval_base_full.yaml` | `python evaluate_r3.py --config configs/eval_base_full.yaml --device cuda --ckpt_dir ./models/Qwen3-VL-8B-Instruct --split val --use_chat_template` | vision_tokens=64，关闭量化，对齐官方能力 |
