# CODEX_HANDOFF — TokenGS 实例分割 / Absolute-Unit 研究

生成日期：2026-09-04
仓库根目录：`/space0/mawb/tokengs`（git 仓库）
运行环境：
```bash
export PATH=/space0/mawb/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 HF_HUB_OFFLINE=1
```

> 本文件面向 240 服务器上的新 Codex：只读本文件 + 代码即可继续。任何数值结论都标注来源实验，避免把“旧评测路径产生的 20.12 伪值”当作真实结果。

---

## 1. 研究目标与当前 Pipeline

**目标**：以 TokenGS（1024 decoder tokens × 64 GS/token，基于 DUSt3R-like 多视图编码）为骨架，逼近/超过 InstOk3D 在 ScanNet LSM-40 上的一步前向 class-agnostic instance segmentation（InstOk3D 论文 AP50=0.438、AP25=0.564；ObjectGS per-scene 0.337、Gaussian Grouping 0.288）。随后再做 semantic（C3G/Uni3R 风格）与 instance 的统一评测。

**当前结论（截至交接）**：
- 冻结表示 + 后置 head 的 one-shot 上限 ≈ **AP50 0.324**（wide7l@8000 语义化特征 + DINO + 8 local units + InfoNCE + Agglomerative eps=0.5），PSNR 19.81。
- 纯 RGB 重建特征（tokengs_re10k 或 base8k recon-only）上同 head 只有 ~0.09，且是欠分割（pred/gt≈0.8）；说明 **wide7l 语义化 token 特征才是 0.324 的来源**，不是重建质量。
- 因此现阶段的“主实验”改为 **token-aligned absolute student**：`Token → 8 Shared Local Units → 每 unit 8 GS → 完整 Student GS`，reconstruction 先 bootstrap 到接近冻结 teacher（base8k@8000，PSNR≈20.12）的 19.1–19.6，之后才允许打开 instance 监督。**在此之前不要加 semantic / DINO / Mutual Benefit / 更多 instance head。**

**正在跑的 pipeline（完整数据 recon-only，尚未跑）**：
```text
base8k@8000 (frozen teacher + backbone)
        │ encoder + token transformer（frozen）
        ▼
   gs_token_hidden ──► AbsoluteUnitDecoder（唯一可训练）
                          Token → 8 Shared Local Units → 8 GS/unit
        │
        ├─ RGB render（student）── RGB loss（真实 GT）
        ├─ RGB render（student）── teacher RGB distill（frozen teacher no_grad）
        └─ per-token Hungarian GS pair ── low-weight GS distill
```
Teacher 由 base8k@8000 的旧 activation head 提供，只做监督；student 输出完全不依赖旧 GS 参数。

---

## 2. 结构定义（Shared Local Unit / Absolute Student / Teacher）

### 2.1 Shared Local Units（token 内共享 query 先验 + token content readout）
- 每 token 8 个 unit；unit query = `unit_queries(K, D)`（跨 token 共享参数）+ token content readout，unit 特征经共享 MLP 解码。
- 实例侧旧分支（`TokenLocalUnitGrouping`）里的 8 units 是对 64 个已存在 GS 做 soft k-means 分组的“3D-local units”；absolute 学生侧的 8 units 是**先形成 unit、再由 unit 解码 8 GS**（固定 slot，不依赖旧 GS 做分组）。两套结构都保留，用途不同。

### 2.2 Absolute Student（`AbsoluteUnitDecoder`，见 `tokengs/models/absolute_unit_decoder.py`）
- 输入：`gs_token_hidden [B,T,1024]`（decoder blocks 输出，不含 activation head）；
- 输出：完整 14 维 Gaussian（position/opacity/log-scale/quaternion/color），`[B,T*64,14]`；
- 完全不叠加/不读取旧 GS head 输出；旧 head 只出现在 teacher no_grad 路径。
- 24 个参数 key，~675K 可训练参数（unit queries/readout + slot emb + decoder + center MLP）。

### 2.3 Teacher（base8k@8000 冻结旧 GS head）
- 权重来源：`workspace/scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors`（TokenGS-only 提取：395 keys，含 activation head；patch_embed 保持 re10k 初值，其余为该 recon fine-tune 已训练权重）。
- 使用方式：`no_grad`，每步产生 Teacher GS + Teacher RGB；GS 蒸馏按 **per-token Hungarian** 配对（64×64 位置代价），scale 权重 10、颜色不蒸馏（RGB distill 覆盖）。
- Teacher 调度（主 abs 实验 config）：0–600 步 eff=1、600–1000 线性衰减、1000 后关闭并**完全跳过旧 head 前向**。
- recon-only 阶段（`..._recon_continue` / `..._recon_full3`）：teacher 恒为 1，永不衰减。

### 2.4 保留的对照结构
- residual generative units（`generative_units=True`，`semantic_v6_generative_units_teacher_train`）：`Final GS = 冻结 Old GS + unit residual`，作为与 absolute 的对照，不改。
- 纯 RE10K absolute 消融：`semantic_v6_absolute_units_teacher_re10k_train`（teacher=re10k）。

---

## 3. 关键实验、指标与结论（按时间线）

| 实验 | AP50 | PSNR | 结论 |
|---|---|---|---|
| pgr3df2（wide7l frozen） | 0.232 | 19.81 | 早期 best，per-GS residual head |
| token-level GroupToken | 0.219 | — | 固定全局 query 事后绑定不足 |
| GroupToken+per-GS refine | 0.208 | — | 同上 |
| unit embedding + Agglomerative | 0.21–0.28 | — | 向 0.324 演进 |
| +patch feature + soft InfoNCE | 0.279 | 19.81 | 判别力提升 |
| +DINOv2 dense feature（**0.324 baseline**） | **0.324** | 19.81 | 冻结表示 one-shot 实际上限 |
| DINO GT-prototype oracle | 0.534 | — | 特征空间上界（用 GT 中心） |
| GS-level GT oracle | 0.789 | — | 几何上界 |
| learned 8-unit + GT identity oracle | ~0.70 | — | unit 形成足够，binding 是断点 |
| B0 SIC（scene-conditioned unit queries, generic trainer） | 0.321 | 19.81 | 结构改动无增益；B 线关闭 |
| base8k@8000 纯重建 + 0.324 recipe | 0.09 | 20.12 | 纯重建特征缺 instance identity（欠分割） |
| absolute student 3000 步（**旧评测路径**） | 0.044–0.101 | “20.12 恒定” | **评测 bug**：未启用 abs 分支，RGB 走旧 head |
| absolute student 3000 步（修复后单场景） | — | 17.09（scene0686_01） | 真实 student-only；PSNR 随 ckpt 变化 |
| recon continuation（从 step1000 fork, 4000 步） | — | validation 16.00→16.93，best 16.911 | 训练量不足，未到 teacher |
| single-sample overfit（450/500 步） | — | student 21.79 / 21.52 vs teacher 22.16 | **容量基本足够** |
| single-sample overfit step475 | — | 17.97（3 次复现一致） | 优化震荡（abs_grad 尖峰 65k），非结构问题；已用 LR 3e-5 缓解 |

评测口径注意：40 场景 LSM eval 的 `mean_psnr/ssim/lpips` 只对 student GS 有意义；旧 `lsm_abs_base8k_*` 五份结果全部作废（eval 曾缺 `instance_branch_abs_units` 透传）。真实 student 早期 PSNR 低于 20.12 属正常。

---

## 4. 已证伪的方向（避免重复）

- 冻结 wide7l 表示上的 embedding/clustering 变体（InfoNCE 变体、center push、rendered pull/push、soft-proto+margin）：全部 ≤0.325。
- 固定/动态 100 slots、scene assignment、DPG、seeded/peak-extraction、GT-count prune/maxclust：全部失败（0.03–0.18，slot collapse/碎片）。
- 仅加 scene-conditioned query 到 unit formation（B0 SIC）：≈0.321，无增益 → **不要重启 B0/B1**。
- base8k 纯 RGB recon fine-tune + 旧 head 复跑：~0.09；**纯重建不会产生 instance identity**，需要 instance 监督参与表示学习（joint/生成式）才有意义。
- generative units v1/v2（无 teacher 直接 joint）：v2 rendered-mask 结果差、且当时跑在错误的 trainer/backbone 口径上；若要复用必须用“residual/absolute + teacher bootstrap”路径。
- “直接堆正式训练步数”：在 0.324 pipeline 上已证无收益；现在 recon 阶段只允许 3 个完整 epoch + LR 3e-5，若仍 <19 需先分析，不许盲加 instance。

---

## 5. 当前有效配置与关键代码

### Options（`tokengs/options.py`）
- `semantic_v6_absolute_units_teacher_train`（主 base8k abs 实验，含 600/1000/1800 调度）
- `semantic_v6_absolute_units_teacher_re10k_train`（RE10K 消融）
- `semantic_v6_absolute_units_recon_continue`（从 abs@1000 fork 的 recon-only 续训，teacher 恒 1，20 epoch×200）
- `semantic_v6_absolute_units_recon_full3`（**新完整数据 recon 方案**，3 epoch×~5680，LR 3e-5，abs_ckpt_every=1000）
- `semantic_v6_generative_units_teacher_train`（residual 对照）
- 关键 flag：`instance_branch_abs_units`、`abs_bootstrap_steps/abs_teacher_decay_steps/abs_instance_warmup_steps`、`abs_teacher_gs_weight/rgb_weight`、`abs_freeze_instance`、`abs_ckpt_every`、`gen_teacher_distill/decay_steps`、`instance_branch_sic_*`（已废弃验证）、`lambda_rgb`（abs/recon 用 200）。

### 模型/评测代码
- `tokengs/models/absolute_unit_decoder.py`：absolute student + Hungarian teacher pair/distill。
- `tokengs/models/semantic_tokengs_v4.py`：abs 分支接入（teacher-on/off、semantic 关闭、instance stage eff、best PSNR 相关 helper）。
- `tokengs/models/semantic_tokengs_v6.py`：abs decoder 构造、semantic/prompt 硬冻结、`abs_freeze_instance`。
- `tokengs/models/instance_group_head.py`：原有 Token→8 units→Group/渲染与 `abs_render_grad`。
- `tokengs/train.py`：generic trainer；abs resume tolerant 分支、validation mean PSNR、eval 语义跳过、intra-epoch ckpt、`[abs-train]` 日志。
- `scripts/eval_instance_lsm_protocol.py`：LSM-40 eval；abs 字段透传、`gaussians_source`/`abs_loaded`/`old_gs_head_calls` 输出。
- 诊断脚本：`scripts/smoke_absolute_units_teacher.py`、`scripts/audit_abs_eval_chain.py`、`scripts/train_abs_single_overfit.py`、`scripts/audit_overfit_checkpoints.py`。

### 关键约定
- 训练一律用通用 trainer：`python -u -m tokengs.train <config> ...`（不要用 `python tokengs/train.py`，路径 sys.path 不对）。
- resume 用新 workspace + `--resume`：只加载权重（epoch_start=0、optimizer 全新），abs 走 tolerant 分支，backbone 从 `backbone_resume` 加载。
- 有旧 semantic loss 污染的 workspace 禁止复用；分叉全部用新目录。

---

## 6. 需要迁移的 Checkpoint（均在 `/space0/mawb/tokengs/workspace/`）

| 角色 | 路径 | 用途/备注 |
|---|---|---|
| wide7l@8000 | `semantic_v6_open_vocab_full_ce2_wide7l_train_8000/checkpoints/model_step_008000.safetensors` | 0.324 冻结 backbone（语义化） |
| tokengs_re10k | `tokengs_re10k/model.safetensors` | 纯重建基座 |
| token-units@6000 | `semantic_v6_token_units_train_12000/checkpoints/model_step_006000.safetensors` | 旧 unit 分支 warm start |
| 0.324 instance | `semantic_v6_unit_shaping_img_dino_train_3000/checkpoints/model_step_003000.safetensors` | 当前 best head（DINO/patch/unit-shaping） |
| base8k recon@8000（完整） | `scannet_recon_finetune_base_8k/checkpoints/model_step_008000.safetensors` | recon fine-tune 全量（teacher 权重来源之一） |
| base8k TokenGS-only | `scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors` | abs/recon 的 prompt/backbone 加载文件（含 activation head） |
| abs 主实验 | `semantic_v6_absolute_units_teacher_base8k_train_3000/checkpoints/model_step_{000600,001000,001800,002600,003000}.safetensors` | 已跑完；**step1000 是 recon fork 点** |
| recon continuation | `semantic_v6_absolute_units_recon_continue_4000/model_best.safetensors`（step2200 选 best）及 `checkpoints/*` | recon-only 4k 步产物，`model_best`=full-data 训练 resume 源 |
| overfit 诊断 | `abs_single_overfit_500/checkpoints/model_step_{000450,000475,000500}.safetensors` | 仅诊断，**禁止用于正式续训** |

迁移时保留目录名，评测/训练都依赖绝对路径 `backbone_resume` 与 metadata。

---

## 7. 当前完整数据 Reconstruction 训练方案（待正式运行）

Config：`semantic_v6_absolute_units_recon_full3`

- resume：`workspace/semantic_v6_absolute_units_recon_continue_4000/model_best.safetensors`
- 数据：`scannet_prompt_small`（full wide 8×7 manifest，train ~5680）
- 3 个完整 epoch × 5680 = 17040 步；`max_iters_per_epoch=6000`（不截断）
- 可训练：仅 `absolute_gs_head`（unit formation + decoder）；teacher/backbone/instance/semantic 全冻结
- Teacher 恒开；instance_stage_eff 恒 0；loss = RGB + teacher RGB + GS distill
- LR=3e-5（475 震荡修复依据），AdamW weight_decay=0.05，grad clip=1.0，bf16
- 每 1000 步存 head ckpt（`checkpoints/model_step_*.safetensors`），epoch 末存 5680/11360/17040 并更新 `model_best`（mean validation PSNR）

命令见第 9 节。

---

## 8. 后续 Instance / Mutual Benefit 计划（仅计划，禁止现在实现）

1. 前置条件：full3 的 student-only mean PSNR 达 **19.1–19.6**（否则先分析 Hungarian GS 蒸馏权重、decoder 参数化/scale 初始化与泛化，不直接加 instance）。
2. Instance 阶段：从 recon 最佳 ckpt fork 新 config（等价 abs 主实验的 instance 调度：先纯重建/teacher，再 600–1000 衰减 teacher，1000–1800 渐开 instance）。instance 侧沿用 unit-level group tokens → student GS 渲染 mask + Hungarian BCE/Dice；instance loss 会通过 student GS 反传（`abs_render_grad`），使 instance 监督直接塑造 unit decoder。
3. Mutual Benefit Mechanism（MBM）：尚无任何实现/验证。方向性定义：reconstruction 与 instance 在同一 unit 表示上互相增强——instance-pure units 产生更锐利的边界（boundary-aware RGB 项），RGB 质量反过来稳定 instance grouping；跨任务梯度平衡与收敛协调优先于任何新 head。实现前必须先重读第 4 节已证伪清单。
4. Semantic：后续再对 C3G8/Uni3R-LSM 协议；不要在 instance 阶段混入 semantic。
5. 全程 one-shot feed-forward，禁止 TTT（仅可作 oracle 诊断）。

---

## 9. 训练/评测命令与阈值

### 完整数据 recon（正式）
```bash
cd /space0/mawb/tokengs
CUDA_VISIBLE_DEVICES=0 python -u -m tokengs.train \
  semantic_v6_absolute_units_recon_full3 \
  --workspace workspace/semantic_v6_absolute_units_recon_full3 \
  --resume workspace/semantic_v6_absolute_units_recon_continue_4000/model_best.safetensors
```

### student-only LSM-40 阶段评测（跑在 student GS 上）
```bash
for ck in $(ls workspace/semantic_v6_absolute_units_recon_full3/checkpoints/model_step_*.safetensors \
            | sed 's/.*model_step_//;s/\.safetensors//' | sort -n); do
  CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_instance_lsm_protocol.py \
    --resume workspace/semantic_v6_absolute_units_recon_full3/checkpoints/model_step_${ck}.safetensors \
    --workspace workspace/lsm_recon_full3_${ck} --label recon_full3_${ck}
done
```
至少评 001000 / 005680 / 011360 / 017040 与 `model_best`。每个 JSON 必须出现 `gaussians_source=absolute_student`、`old_gs_head_calls=0`。

### 单样本 overfit 诊断（用于定位容量 vs 训练量）
```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_abs_single_overfit.py \
  --workspace workspace/abs_single_overfit_500 \
  --resume workspace/semantic_v6_absolute_units_recon_continue_4000/model_best.safetensors \
  --steps 500 --eval-every 25 --save-every 25 --sample-index 0
```

### 阈值与注意事项
- full3 目标：student-only LSM mean PSNR ≥ **19.1–19.6**；若最后两 epoch 增益 <0.1 dB 且仍明显低于 19 → 先分析，不加 instance。
- 训练日志：`[abs-train]` 看 `loss_rgb / teacher_gs / teacher_rgb / teacher_eff / inst / inst_eff / teacher_called / abs_grad / sem_grad / abs_norm`；recon 阶段要求 teacher_eff=1、inst=0、sem_grad=0、student eval old_calls=0（训练中 teacher 监督会合法调用旧 head 1 次/step）。
- checkpoint 命名：full3 新 workspace 从 0 计，保存步=累计步（1000/2000/…/5680/…）；recon_continue 的保存步 N 对应绝对步 1000+N。
- 不要用旧 `lsm_abs_base8k_*` 的 20.12 作为参照；旧评测有 abs 透传 bug。
- LSM 评测相机为 ScanNet `.sens` poses，论文为 scene-level COLMAP，官方实现未开源：跨论文数字只能“部分对齐”引用。
- GPU 空闲查看 `nvidia-smi`；评测/训练都带 `CUDA_VISIBLE_DEVICES`。

---

## 10. Git 状态（未提交的重要修改）

仓库 `/space0/mawb/tokengs` 大量未提交；不要用 `git checkout --` 或 `git reset --hard`。

- **已跟踪但修改**：`tokengs/options.py`（+7453 行，新增大量 instance/abs/sic config 与选项）、`tokengs/train.py`（+1090 行，abs resume/eval/ckpt 逻辑）、`configs/semantic/scannet_c3g8.yaml`、`tokengs/data/*`、`tokengs/models/{tokengs,prompt_*,semantic_adapter_v2,semantic_tokengs_v2,conditional_prompt_tokengs,__init__}.py`、`tokengs/rendering/gs.py` 等（多为既有未提交工作，与本 instance 线部分相关）。
- **未跟踪的新文件（关键，勿删）**：
  - `tokengs/models/absolute_unit_decoder.py`
  - `tokengs/models/semantic_tokengs_v4.py`、`semantic_tokengs_v6.py`、`instance_group_head.py`
  - `scripts/eval_instance_lsm_protocol.py`、`train_unit_joint.py`
  - `scripts/smoke_absolute_units_teacher.py`、`smoke_b0_sic.py`、`smoke_generative_units_teacher.py`
  - `scripts/audit_abs_eval_chain.py`、`audit_overfit_checkpoints.py`、`train_abs_single_overfit.py`
  - 大量诊断/可视化/oracle 脚本（`diagnose_*`、`dino_*`、`ablate_*`、`seeded_*`、`oracle_*`、`probe_*`、`tta_gtfree.py` 等）
  - 数据/产物目录：`data/ImageNet`、`data/scannetpp_processed/`、`logs/`、`workspace/*`（experiment results/checkpoints）
- 建议：开始长训练前先提交一次快照（或至少 `git add -N` 关键源码文件并保存 diff），并单独用 `rsync` 备份 `workspace/` 与 `data/`。

---

## 附：一份“最短起步检查单”
1. `nvidia-smi` 找空卡；2. 确认 `workspace/scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors` 与 `workspace/semantic_v6_absolute_units_recon_continue_4000/model_best.safetensors` 存在；3. 跑第 9 节 full3 训练命令；4. 每 1000/epoch 用 LSM eval 只读 `mean_psnr/ssim/lpips`（student）；5. 达标前不加 instance/semantic/MBM。
