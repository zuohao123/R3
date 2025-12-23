# R³ 训练/评测大表（推荐 8×V100 32GB）

## 资源与前置
- 硬件：推荐 8×V100 32GB（单机），CPU 64c，RAM 128GiB。（4 卡同机也可跑，吞吐更低）
- 模型：`./models/Qwen3-VL-8B-Instruct`（本地权重）。
- 数据：`./data_pipeline/data/{docvqa,infovqa,chartqa}`（train/val），伪文本库 `./artifacts/pseudo_text_all_train.cleaned.jsonl`。
- 依赖：`pip install -r requirements.txt`。
- 日志：`train.log`（loss）、`console.log`（进度），TensorBoard：`tensorboard --logdir checkpoints`.

## 训练阶段（建议按序）
- Stage1：干净多任务，对齐基座，关闭 PMC/检索/一致性。
- Stage2：轻 PMC，开启腐蚀+检索+一致性，轻度 drop/遮挡。
- Stage3：重 PMC，强化腐蚀和一致性权重。

核心超参（已写入 configs）：
- 推荐 8 卡同机用 **DDP**（`torchrun --nproc_per_node=8`），每卡 batch=1，grad_accum=4（全局≈32）。
- `dtype=fp16` + `fp16: true`（AMP）：与基座/评测一致（fp16 权重），同时显存更省、速度更快；无量化。
- lr：Stage1 8e-5；Stage2 5e-5；Stage3 3e-5；wd=0.01；warmup 5%/8%。
- 分组 LR：Stage2/3 默认 `lr_r3_mult=2.0`（R³模块更快学），`lr_lora_mult=0.5/0.3`（LoRA 更稳）。
- `device_map=auto`：DDP 时会自动映射到 `LOCAL_RANK` 的单卡（不会跨卡切分）。
- Stage2/3 长序列：`max_seq_length=10240`，`retrieval_max_evidence_tokens=1024`，`retrieval_chunk_tokens=128`；伪文本 `pseudo_text_max_chars=2000`，`pseudo_text_chunk_tokens=128`。
- 建议环境变量：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（降低碎片化导致的 alloc fail）。

## 阶段目标自检（确认 R³ 真正在训练）

| 阶段 | 目标 | 关键开关（configs） | 是否走 R³ 路径 | loss 是否包含 | 你应该在日志里看到 |
| --- | --- | --- | --- | --- | --- |
| Stage1（干净） | 对齐/强化基座在 3 数据集上的 QA 能力 | `apply_corruption=false`；`enable_retrieval=false`；`enable_consistency=false` | 基本等价“只用基座+LoRA”（R³模块不参与融合） | 仅 `task_loss` | `Trainable params: R3=... LoRA=...`（R3 可能存在但梯度基本为 0）+ quick_eval 基座输出正常 |
| Stage2（轻 PMC） | 学会在轻度腐蚀下用检索补全，并保持与 clean 一致 | `apply_corruption=true`；`enable_retrieval=true`；`enable_consistency=true` | 是（retrieval→reconstruction→reasoner） | `task_loss + λ * consistency_loss` | loss 能下降；且 `Trainable params: R3=...` 非 0；腐蚀强度来自 stage2.yaml |
| Stage3（重 PMC） | 强化鲁棒性与抗幻觉 | 同 Stage2，但腐蚀更强、`lambda_consistency` 更高 | 是 | 同 Stage2 | 评测在 `apply_corruption=true` 下相对 Stage2 更稳；stage3.yaml 腐蚀更强 |

说明：`R3Trainer` 已对齐 token 布局做了标签 padding（prefix/imputation/vision 都用 `-100` 忽略），并按真实答案长度预留 token 预算，避免“伪文本/召回占满，标签全为 -100”导致 loss=0。

## 命令大表

| 任务 | 目标/场景 | 配置/权重 | 命令 | 备注 |
| --- | --- | --- | --- | --- |
| 训练（快测） | Stage1 干净（8 卡 DDP，前台 200 步） | `configs/default.yaml` → `checkpoints/stage1_clean_ddp8` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/stage1_clean_ddp8 --log_file checkpoints/stage1_clean_ddp8/train.log --max_steps 200 --quick_eval_every 200 --log_interval 10 2>&1 \| tee checkpoints/stage1_clean_ddp8/console.log` | DDP 更快；`train.log` 仅 rank0 写入 |
| 训练（脚本） | Stage1 干净（后台） | 同上 | `bash scripts/train_stage1_ddp8.sh checkpoints/stage1_clean_ddp8` | 默认 max_steps=3000，quick_eval=500 |
| 训练（脚本） | Stage2 轻 PMC（后台，初始化自 Stage1 权重，不继承优化器） | `configs/stage2.yaml` → `checkpoints/stage2_light_pmc_ddp8` | `bash scripts/train_stage2_ddp8.sh checkpoints/stage1_clean_ddp8/checkpoint-3000 checkpoints/stage2_light_pmc_ddp8` | 用 `--init_from_checkpoint` 只加载权重，重新走 Stage2 学习率/调度 |
| 训练（脚本） | Stage3 重 PMC（后台，初始化自 Stage2 权重） | `configs/stage3.yaml` → `checkpoints/stage3_heavy_pmc_ddp8` | `bash scripts/train_stage3_ddp8.sh checkpoints/stage2_light_pmc_ddp8/checkpoint-2000 checkpoints/stage3_heavy_pmc_ddp8` | λ_consistency 更高，腐蚀更强 |
| 训练（备选） | 单进程模型并行（4/8 卡，避免 DDP 环境问题） | 同上 | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/stage1_clean_mp --log_file checkpoints/stage1_clean_mp/train.log --max_steps 200 --quick_eval_every 200 2>&1 \| tee checkpoints/stage1_clean_mp/console.log` | 速度不如 DDP，但最稳；若 DDP 仍 OOM 可先用它 |
| 评测-基座 | 干净 val（基座能力） | `configs/default.yaml` + `./models/Qwen3-VL-8B-Instruct` | `python evaluate_r3.py --config configs/default.yaml --device cuda --ckpt_dir ./models/Qwen3-VL-8B-Instruct --split val --use_chat_template --native_eval --limit 200 --log_interval 10 --log_samples 5 --errors errors_base_clean.jsonl --predictions preds_base_clean.jsonl` | 原生生成式评测，batch=1 |
| 评测-基座 | Stage2 腐蚀 val（轻 PMC 基线） | `configs/stage2.yaml` + `./models/Qwen3-VL-8B-Instruct` | `python evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir ./models/Qwen3-VL-8B-Instruct --split val --use_chat_template --native_eval --apply_corruption --limit 200 --log_interval 10 --log_samples 5 --errors errors_base_s2.jsonl --predictions preds_base_s2.jsonl` | 轻缺模态（按 stage2.yaml 的腐蚀强度） |
| 评测-基座 | Stage3 腐蚀 val（重 PMC 基线） | `configs/stage3.yaml` + `./models/Qwen3-VL-8B-Instruct` | `python evaluate_r3.py --config configs/stage3.yaml --device cuda --ckpt_dir ./models/Qwen3-VL-8B-Instruct --split val --use_chat_template --native_eval --apply_corruption --limit 200 --log_interval 10 --log_samples 5 --errors errors_base_s3.jsonl --predictions preds_base_s3.jsonl` | 重缺模态（按 stage3.yaml 的腐蚀强度） |
| 评测-R³ | Stage2 checkpoint（teacher forcing 评估） | `configs/stage2.yaml` + `checkpoints/stage2_light_pmc/checkpoint-1000` | `CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir checkpoints/stage2_light_pmc/checkpoint-1000 --split val --use_chat_template --limit 200 --log_interval 10 --log_samples 5 --errors errors_r3_s2.jsonl --predictions preds_r3_s2.jsonl` | 非 `--native_eval`，会加载 R³ 权重；输出为 teacher-forcing argmax（便于 debug） |
| 消融-无检索 | Stage2 checkpoint | 同上 | `CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_r3.py --config configs/stage2.yaml --device cuda --ckpt_dir checkpoints/stage2_light_pmc/checkpoint-1000 --split val --use_chat_template --limit 200 --disable_retrieval --errors errors_r3_s2_noR.jsonl` | 关闭检索模块 |
| 消融-无一致性 | Stage2 checkpoint | 同上 | `... --disable_consistency ...` | 关闭一致性项（eval 默认不算 consistency loss，但可用于禁用相关模块路径） |
| 消融-无腐蚀模拟 | Stage2 checkpoint | 同上 | `... --disable_corruption ...` | 评估时不施加腐蚀（对比 clean） |

说明：
- `--max_steps` 可按数据量/时间调整（如 1500–3000）；或去掉则跑满 1 epoch。
- 分阶段训练建议用 `--init_from_checkpoint checkpoints/.../checkpoint-XXXX`（只继承权重，不继承优化器/调度），更符合 “阶段性策略切换”。
- 需要后台运行直接使用 nohup，日志在 `<out_dir>/train.log`、`console.log`。
- 评测脚本已提供 `--predictions`（全量预测）和 `--errors`（仅错误样本）方便检查。

## 调参提示
- 初期稳定性优先：fp32（当前默认）、关闭量化/grad checkpoint、小 lr、分组 LR（R³ 模块更快、LoRA 更稳）。
- 如显存紧张：降低 `grad_accum_steps` 或缩短 `max_seq_length`（需同步训练/评测一致）。
- 如想更高吞吐：可尝试 8 卡，适当调小 `grad_accum_steps` 保持全局 batch 不变。
