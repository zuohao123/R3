# R³ 训练/评测大表（4×V100 32GB）

## 资源与前置
- 硬件：4×V100 32GB，CPU 64c，RAM 128GiB。
- 模型：`./models/Qwen3-VL-8B-Instruct`（本地权重）。
- 数据：`./data_pipeline/data/{docvqa,infovqa,chartqa}`（train/val），伪文本库 `./artifacts/pseudo_text_all_train.jsonl`。
- 依赖：`pip install -r requirements.txt`。
- 日志：`train.log`（loss）、`console.log`（进度），TensorBoard：`tensorboard --logdir checkpoints`.

## 训练阶段（建议按序）
- Stage1：干净多任务，对齐基座，关闭 PMC/检索/一致性。
- Stage2：轻 PMC，开启腐蚀+检索+一致性，轻度 drop/遮挡。
- Stage3：重 PMC，强化腐蚀和一致性权重。

核心超参（已写入 configs）：
- batch_per_gpu=2，grad_accum=12（全局≈24，模型并行后显存足够）。
- fp16 关闭（fp32 训练以稳定对齐基座），vision_tokens=64，全精无量化。
- lr：Stage1 1e-4；Stage2 5e-5；Stage3 3e-5；wd=0.01；warmup 2%/5%。
- device_map=auto（单进程模型并行），无量化，无 grad checkpoint（显存不足再开）。

## 命令大表

| 任务 | 目标/场景 | 配置/权重 | 命令 | 备注 |
| --- | --- | --- | --- | --- |
| 训练（快测） | Stage1 干净（模型并行，前台 200 步） | `configs/default.yaml` → `checkpoints/stage1_clean_mp` | `CUDA_VISIBLE_DEVICES=0,1,2,3 python train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/stage1_clean_mp --log_file checkpoints/stage1_clean_mp/train.log --max_steps 200 --quick_eval_every 5 2>&1 \| tee checkpoints/stage1_clean_mp/console.log` | 单进程模型并行，batch=2, accum=12，fp32 |
| 训练 | Stage1 干净多任务（正式） | `configs/default.yaml` → `checkpoints/stage1_clean` | `mkdir -p checkpoints/stage1_clean && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/stage1_clean --log_file checkpoints/stage1_clean/train.log --max_steps 1500 --quick_eval_every 5 > checkpoints/stage1_clean/console.log 2>&1 &` | 关闭 corruption/retrieval/consistency，lr=1e-4，模型并行 |
| 训练 | Stage2 轻 PMC | `configs/stage2.yaml` → `checkpoints/stage2_light_pmc` | `mkdir -p checkpoints/stage2_light_pmc && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train_r3.py --config configs/stage2.yaml --device cuda --output_dir checkpoints/stage2_light_pmc --log_file checkpoints/stage2_light_pmc/train.log --max_steps 1500 --quick_eval_every 5 > checkpoints/stage2_light_pmc/console.log 2>&1 &` | 开启 corruption/retrieval/consistency，lr=5e-5，模型并行 |
| 训练 | Stage3 重 PMC | `configs/stage3.yaml` → `checkpoints/stage3_heavy_pmc` | `mkdir -p checkpoints/stage3_heavy_pmc && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train_r3.py --config configs/stage3.yaml --device cuda --output_dir checkpoints/stage3_heavy_pmc --log_file checkpoints/stage3_heavy_pmc/train.log --max_steps 1500 --quick_eval_every 5 > checkpoints/stage3_heavy_pmc/console.log 2>&1 &` | 更强腐蚀，λ_consistency=0.5，lr=3e-5，模型并行 |
| 评测-基座 | 干净 val，对齐官方 | `configs/default.yaml` + `./models/Qwen3-VL-8B-Instruct` | `python evaluate_r3.py --config configs/default.yaml --device cuda --ckpt_dir ./models/Qwen3-VL-8B-Instruct --split val --use_chat_template --native_eval --limit 200 --log_interval 10 --log_samples 5 --errors errors_base.jsonl --predictions preds_base.jsonl` | 原生前向，batch=1 |
| 评测-Stage1 | 干净 val | `configs/default.yaml` + `checkpoints/stage1_clean` | `python evaluate_r3.py --config configs/default.yaml --device cuda --ckpt_dir checkpoints/stage1_clean --split val --use_chat_template --native_eval --limit 200 --log_interval 10 --log_samples 5 --errors errors_s1.jsonl --predictions preds_s1.jsonl` | 对比基座 |
| 评测-Stage2 | 轻腐蚀 val | `configs/stage2.yaml` + `checkpoints/stage2_light_pmc` | `python evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir checkpoints/stage2_light_pmc --split val --use_chat_template --native_eval --limit 200 --log_interval 10 --log_samples 5 --errors errors_s2.jsonl --predictions preds_s2.jsonl` | 轻缺模态 |
| 评测-Stage3 | 重腐蚀 val | `configs/stage3.yaml` + `checkpoints/stage3_heavy_pmc` | `python evaluate_r3.py --config configs/stage3.yaml --device cuda --ckpt_dir checkpoints/stage3_heavy_pmc --split val --use_chat_template --native_eval --limit 200 --log_interval 10 --log_samples 5 --errors errors_s3.jsonl --predictions preds_s3.jsonl` | 重缺模态 |
| 消融-无检索 | 轻腐蚀 | `configs/stage2.yaml` + `checkpoints/stage2_light_pmc` | `python evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir checkpoints/stage2_light_pmc --split val --use_chat_template --native_eval --disable_retrieval --limit 200 --errors errors_s2_noR.jsonl` | 检索贡献 |
| 消融-无一致性 | 轻腐蚀 | 同上 | `... --disable_consistency ...` | 幻觉率对比 |
| 消融-无腐蚀模拟 | 轻腐蚀 | 同上 | `... --disable_corruption ...` | 模拟器贡献 |

说明：
- `--max_steps` 可按数据量/时间调整（如 1500–3000）；或去掉则跑满 1 epoch。
- 需要后台运行直接使用 nohup，日志在 `<out_dir>/train.log`、`console.log`。
- 评测脚本已提供 `--predictions`（全量预测）和 `--errors`（仅错误样本）方便检查。

## 调参提示
- 初期稳定性优先：全精 fp16、关闭量化/grad checkpoint、小 lr、冻结主干（目前配置仅更新 LoRA/新模块）。
- 如显存紧张：降低 `grad_accum_steps` 或缩短 `max_seq_length`（需同步训练/评测一致）。
- 如想更高吞吐：可尝试 8 卡，适当调小 `grad_accum_steps` 保持全局 batch 不变。
