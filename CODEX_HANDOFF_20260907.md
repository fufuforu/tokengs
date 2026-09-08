# CODEX_HANDOFF — TokenGS True Shared Unit / MBM / PGSR（2026-09-07 交接）

生成日期：2026-09-07
仓库：`/space/mawb/tokengs`（git，branch `main`，HEAD `68435e5ca129dd70953eabe1a75e634c27dbb9e8`）
本文件是**只读核对产物**：本轮没有修改模型、loss、配置、checkpoint，没有启动训练或正式评测。所有数值均从实际 JSON / checkpoint metadata / config 读取；找不到的旧结果会明确标注“仅历史记录、240 上无 JSON 可复核”，不会猜测。

> 旧的 `CODEX_HANDOFF.md` 保留未动（内容是 2024-09-04 的早期路线，含 0.324 wide7l 与 base8k 的历史数字）。新旧两份文件并存，不要互相覆盖。

---

## 1. Executive Summary

当前主线的真实数据流（8 context → 15 view 中的 7 个 target view 渲染）：

```text
8 context views
  → TokenGS encoder / token transformer（encoder 冻结；decoder tail 可训，LR=3e-6）
  → gs_token_hidden [B,1024,1024]
  → AbsoluteUnitDecoder → q_abs [B,1024,8,256]（唯一 Shared Local Unit formation）
      ├─ q_abs → center_mlp / GS decoder → 8 GS per unit → 65536 Student GS
      └─ q_abs → SharedUnitInstanceHead（100 group queries）→ unit_logits
             → broadcast to per-GS logits（PGSR 基线门=0 时严格相同）
  → 同一批 Student GS 渲染 RGB 与 instance masks（mask render 中 GS geometry detach）
  → RGB(λ=200) + instance BCE/Dice(λ=0.05, unit→q_abs 梯度 ×4) + SIU3R 式 U→R(W=10)
```

研究目标：ScanNet **class-agnostic instance segmentation**（LSM-40，40 scenes），要求保持 feed-forward 重建质量（当前正式 mean PSNR≈20.3）。当前最佳实例 checkpoint 为 MBM Both@1420，正式 AP50=0.1503；PGSR 40 场景正式评测没有超过它；下一步只做只读 representation-ceiling 审计，不训练。

---

## 2. 原始 TokenGS 与当前 True Shared 结构

### 2.1 原始 TokenGS
- 图像 encoder（ViT-like patch embed + Plücker 条件）+ scene tokens + 12 层 decoder blocks；每 token 最终产出 64 个 GS（activation head / 旧 GS head），GS/token=64，共 1024 tokens。
- 冻结 backbone（`enc_dec_backbone.encoder_blocks`、patch embed、activation head、gs_tokens）在 MBM/PGSR 实验中保持不变。

### 2.2 AbsoluteUnitDecoder（`tokengs/models/absolute_unit_decoder.py`）
- 输入 `gs_token_hidden [B,1024,1024]`；1024 tokens → 8 units/token → 8 GS/unit → **65536 Student GS**，14 维完整参数。
- 24 个参数 key（tok_norm/tok_proj/unit_queries/unit_readout/center_mlp/gs_decoder/slot_emb）。

### 2.3 True Shared Unit
- 唯一 `q_abs [B,1024,8,256]`：GS decoder 和 instance head 消费**同一个 tensor**（v4 的 `_forward_tsh_instance_branch` 直接接收 q_abs）。没有第二套 `unit_queries/unit_readout`。
- 旧 dual-unit 分支（`instance_branch=TokenLocalUnitGrouping`）在 tsh 配置下不实例化；legacy 结构仅作为历史配置保留。

### 2.4 instance head / mask renderer
- `SharedUnitInstanceHead`（100 group queries）在 q_abs 上做 cross-attention → unit_logits `[B,8192,101]` → pi_unit；baseline（无 PGSR）pi_gs = broadcast(pi_unit) 到 8 GS。
- mask 渲染：`render_feature_channels(render_gs=student_gaussians.detach(), pi_gs, ...)`，再除以 alpha 并跨通道归一化，得到 `[B,G+1,V,1,H,W]`；V=7 个 target view。
- 实例 BCE/Dice→q_abs 的 ×4 gradient edge 由 `grad_scale(q_gated, tsh_unit_gradient_multiplier_max=4.0)` 实现（只放大 instance→q_abs producer 路径，不放大 RGB/teacher）。

### 2.5 当前梯度边界（已验证，见 `audit_pgsr_step0.json` 等）

| 路径 | 更新 |
|---|---|
| GT RGB reconstruction | abs unit formation、GS decoder、decoder tail；不改 instance head |
| teacher RGB/GS | 旧 GS head 已永久关闭（`abs_teacher_decay_steps=0`），不产生 forward |
| instance BCE/Dice | tsh head、abs unit formation（×4）、refine head（PGSR 时）；**center_mlp/gs_decoder/slot_emb=0**（render 用 detach geometry） |
| U→R depth smoothness | abs unit formation、GS decoder、decoder tail；tsh/refine=0（mask 已 detach） |

冻结：image encoder、patch embed、activation head/旧 GS head、gs_tokens、semantic/DINO/CLIP/LSeg 相关模块。可训练：`absolute_gs_head`（LR 1e-5）、`tsh_instance_head`（LR 1e-4）、`enc_dec_backbone.decoder_blocks`（LR 3e-6）、PGSR 的 `tsh_slot_refine_head`（归 instance LR 组）。

---

## 3. Checkpoint lineage（实际核对）

以下均以 2026-09-07 文件系统为准（safetensors 只读 key 统计）：

| 阶段 | 路径 | 存在 | abs keys | tsh keys | refine | tail | 说明 |
|---|---|---|---|---|---|---|---|
| RE10K TokenGS | `/space0/.../tokengs_re10k` 或本地 `checkpoints/tokengs_re10k` | **本地缺失** | — | — | — | — | 旧 108 路径；240 上只有 base8k 提取版，RE10K 原文件未迁移 |
| base8k teacher | `workspace/scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors` | 存在 | 0 | 0 | 0 | 0 | backbone+activation teacher，无 tsh/abs |
| recon continuation | `workspace/semantic_v6_absolute_units_recon_continue_4000/model_best.safetensors` | 存在 | 是（未逐 key 重算） | 0 | 0 | 0 | 只有 model_best，无 checkpoints 目录 |
| recon_full3 | `workspace/semantic_v6_absolute_units_recon_full3/model_best.safetensors` | 存在 | **24** | 0 | 0 | 0 | strict abs；无 backbone keys（构造时从 base8k 载入） |
| m0 head-only DDP8 | `workspace/semantic_v6_absolute_units_true_shared_head_only_m0_ddp8/checkpoints/model_step_001420.safetensors` | 存在 | 24 | **50** | 0 | 0 | abs+tsh strict，无 tail |
| m4 DDP8 | `workspace/semantic_v6_absolute_units_true_shared_joint_m4_ddp8/checkpoints/model_step_001420.safetensors` | 存在 | 24 | 50 | 0 | 0 | 同上 |
| **MBM Both@1420（当前最佳）** | `workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors` | 存在 | **24** | **50** | 0 | **324** | abs+tsh strict、tail 可训练恢复 |
| PGSR 710 | `workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_w10_t3e6_pgsr_ddp8/checkpoints/model_step_000710.safetensors` | 存在 | 24 | 50 | 18 | 324 | 新增 `tsh_slot_refine_head` 18 个 tensor |

- 加载规则（`tokengs/train.py` guarded branch）：首分叉（无 tsh keys）只加载 abs 24/24、tsh fresh reset；tsh checkpoint 存在时 abs+tsh 全载、绝不 fresh reset；tail 若 ckpt 含 keys 则恢复，否则保留 base8k 构造值；PGSR refine keys 若存在则恢复，否则保持零初始化。
- 当前最佳必须写为 **MBM Both@1420**（不要写 m4、不要写 PGSR）。

---

## 4. 有效实验结果表（来自实际 instance_ap.json，全部 40 scenes，gaussians_source=absolute_student，old_gs_head_calls=0）

| 模型/step | mean AP25 | mean AP50 | mean AP75 | pooled AP50 | PSNR | 来源 JSON |
|---|---|---|---|---|---|---|
| recon_full3 step1000 | 0.4053 | 0.0856 | 0.0047 | 0.0584 | 17.32 | `lsm_240_eval/full/recon_full3_001000` |
| recon_full3 model_best | 0.3910 | 0.0707 | 0.0065 | 0.0446 | 18.89 | `.../recon_full3_model_best` |
| m0 step125/1420 | 0.410/0.483 | 0.1040/**0.1319** | 0.0089/0.0147 | 0.0625/0.0991 | 19.81/19.92 | `tsh_m0ddp8` |
| m4 step125/1420 | 0.382/0.473 | 0.0934/0.1263 | 0.0051/0.0151 | 0.0632/0.0951 | 19.72/19.96 | `tsh_m4ddp8` |
| **MBM Both@710/1420/2130** | 0.479/0.489/0.491 | **0.1409 / 0.1503 / 0.1448** | 0.0161/0.0219/0.0172 | 0.1098/**0.1163**/0.1127 | 20.105/**20.304**/20.336 | `lsm_eval_mbm_both_w10/stage` |
| PGSR@125/355/710 | 0.491/0.485/0.497 | 0.1438/0.1497/0.1398 | 0.0252/0.0214/0.0190 | 0.1125/0.1228/0.1122 | 20.078/20.053/20.238 | `lsm_eval_pgsr/stage` |

历史参考（**240 仓库无对应 instance_ap.json，只能当作旧交接记录**，不是本轮可复核结果）：wide7l frozen + DINO/unit pipeline 约 AP50=0.324 / PSNR=19.81；base8k 纯重建 post-hoc 约 AP50≈0.09 / PSNR≈20.12（旧评测 bug 已作废的 20.12 伪值不要与真实 student PSNR 混淆）。

重要判定：
- **当前最佳正式实例 checkpoint = MBM Both@1420（AP50=0.1503，pooled=0.1163，PSNR=20.304）**。
- PGSR 40 场景正式结果（125/355/710）均未超过 Both@1420 的 0.1503；PGSR 4-scene smoke 的提升没有推广到 40 场景。
- 旧 `lsm_abs_base8k_*`（abs 透传 bug）作废；`audit_teacher_smoke` 是 teacher 只读审计。
- 不要混用 LSM-40（40 scenes）与 `tsh_mbm_lsm4/*`（4 scenes smoke）的指标。

---

## 5. SIU3R MBM 映射

- 官方 SIU3R 代码在 `/space/mawb/SIU3R`（只读）。官方 **R→U“Multi-View Mask Aggregation”是推理期 2D mask → per-GS logits lift → splat 聚合**，无可训练参数；TokenGS 的 unit 级 3D assignment 已天然实现该目标，未新增第二套 units。
- 官方 **U→R“Mask-Guided Geometry Refinement”**：`pipeline.py:249-265`，rendered depth 差分 + 预测 mask 内部（hard/detach）L1。TokenGS 版实现在 `semantic_tokengs_v4._tsh_mbm_u2r_loss`：rendered Student-GS depth，predicted refined masks（conf/alpha gate），mask detach。
- U→R 最大权重 10.0：来自 16-batch 实测（unit/GS-decoder median grad ratio 目标 0.5–2%）；decoder tail LR 3e-6 来自真实 Adam update/param 测量和 100 步多场景对比（1e-6 过弱、1e-5 无额外增益）。
- U→R 调度（正式 Both）：step≥710 后从 0 线性 ramp 到 10（710–1420 到满）。
- 当前没有任何 DINO/CLIP/LSeg/semantic 训练损失（`lambda_feat/semantic=0`，semantic 模块 hard frozen）。

---

## 6. PGSR 结果与结论

PGSR 结构：`final_gs_logits = broadcast(unit_logits) + gate_eff × alpha × tanh(MLP(q_abs, slot_emb, detached GS attrs, group context))`，`tsh_slot_refine_head` 零初始化，gate 0→1 ramp 125 步。

验证过的事实：
- step0 identity（gate=0）：RGB/unit logits/pi_gs/rendered masks 与无 PGSR 模型 allclose（`audit_pgsr_step0.json`）。
- instance 梯度只到 refine head + unit formation；`center_mlp/gs_decoder/slot_emb`=0；U→R 不更新 refine head；RGB 路径不受 PGSR 影响。
- 150 步固定样本 overfit：AP50 0.06→0.67，PSNR 22.35→25.05，alpha≈1，unit 内 logit 差异上升（表示层可学）。
- 4-scene smoke（100 步 DDP8）：mean AP50 0.129（起点 0.085），PSNR 无下降。
- **正式 40 场景：PGSR 125/355/710 没有超过 Both@1420**（见第 4 节）。正式无增益。

结论：**不要再继续 PGSR、不要直接增大 alpha**。PGSR 的 per-GS 表示自由度没有转化为正式收益，说明“8 GS 共享 identity”不是当前主要瓶颈。

---

## 7. Multi-view matching 审计与 oracle

### 7.1 matching 事实（16 真实 sample，`audit_multiview_query_binding.json`）
- sample：8 context + 7 target，输入 `[1,15,9,256,256]`、GT label `[1,7,256,256]`。
- ScanNet instance ID 跨 view 保留（instance-filt 全局 ID，仅 crop/interp；同一 GT 常见于 5–7 个 target view）。
- 当前正式 Both 配置 `instance_group_scene_level_matching=false`，即 **逐 target view 独立 Hungarian**（每 view `_hungarian_matches` 一次）。
- 代码中已有 scene-level 实现 `_scene_hungarian_matches`（`tokengs/models/instance_group_loss.py`），仅未被当前配置启用。

| 指标（16 sample） | median |
|---|---|
| cross-view query consistency | 0.32（max 0.64） |
| fragmented GT ratio | 0.66 |
| distinct queries per GT | 2（max 6） |
| query collision ratio | 0.40 |
| target-view pairwise gradient cosine（tsh / unit） | 0.47 / 0.43（min −0.85/−0.88） |

### 7.2 oracle 诊断（4 validation scenes，`oracle_multiview_query_audit.json`）
- best-GT IoU median ≈0.29、Recall@IoU50≈0.20；
- query 合并 / 去重 / 只保留 best 跨视图 query **均无 headroom**（merge oracle IoU 与 best-IoU 相同）；
- same-view 重复 pred（IoU>0.5）≈0。

解读：scene-level Hungarian 从方法完整性上是必要修复（跨视图 query 绑定明显不一致、存在梯度冲突），但它不是当前 AP50 低的首要瓶颈——因为即使 oracle 把 query 绑定修好，pred 的边界质量仍不足。

---

## 8. 当前最关键的待办任务：representation-ceiling 审计（尚未完成）

**下一步不是训练。** 使用冻结 Both@1420 Student GS，优化临时 oracle logits：

- **Unit oracle** `[1024,8,K+1]`：unit softmax → 广播 8 GS → 正式 renderer 渲染 7 views；
- **Per-GS oracle** `[65536,K+1]`：每 GS 独立；
- K = scene 内全局 GT instance ID 数；固定通道，**不用 Hungarian**；
- 同时审计 foreground alpha 覆盖、boundary F@1/2/4/8、interior/boundary IoU、mixed-unit ratio/purity；
- 只写临时参数，**不写回 checkpoint**；优化前后模型 hash 必须一致；
- 输出 learned / unit / per-GS 指标与 oracle 收敛曲线。

判读规则：
- A：per-GS 显著 > unit → 8 GS 共享 identity 是瓶颈（PGSR 结构性失败需换 per-GS/16×4）；
- B：两个 oracle 都远高于 learned → instance head/loss/优化问题；
- C：两个 oracle 都低但 foreground coverage 高 → GS footprint 不适合边界；
- D：foreground coverage 低 → 重建几何问题。

> 目前**没有任何 oracle 证据**可以断言 8×8 unit 粒度一定瓶颈，也没有证据说明 scene-level Hungarian 一定能提升 AP50。正式 710 步训练不要启动。

---

## 9. 代码与配置索引

关键源码：
- `tokengs/models/tokengs.py`：encoder/token transformer/旧 GS head。
- `tokengs/models/absolute_unit_decoder.py`：q_abs/Student GS。
- `tokengs/models/shared_unit_instance_head.py`：TSH group queries。
- `tokengs/models/per_gs_slot_refine_head.py`：PGSR（当前不作为正式主线）。
- `tokengs/models/semantic_tokengs_v4.py`：abs forward、TSH、U→R、schedule。
- `tokengs/models/semantic_tokengs_v6.py`：v6 wrapper/head build/decoder tail 解冻。
- `tokengs/models/instance_group_loss.py`：Hungarian/scene-level 已有实现。
- `tokengs/options.py`：所有 config（含新增 MBM/PGSR 字段）。
- `tokengs/train.py`：DDP、resume、调度、checkpoint。
- `tokengs/rendering/gs.py`：renderer（render_feature_channels / alpha / depth）。
- `scripts/eval_instance_lsm_protocol.py`、`scripts/eval_lsm_240_parallel.sh`、`eval_lsm_240_task.sh`：LSM 评测。
- 关键审计脚本：`audit_shared_unit_structure.py`、`audit_multiview_query_binding.py`、`oracle_multiview_query_audit.py`、`audit_unit_gs_representation_ceiling.py`、`audit_pgsr_step0.py`、`calibrate_mbm_u2r_weight.py`、`measure_mbm_optimizer_updates.py`、`reliability_true_shared.py`、`overfit_true_shared_mbm.py`、`fork_true_shared_from_warmup.sh`。
- 240 remap：`240_shims/`、`_240_path_remap.py`。

配置继承与“当前配置”：
- `semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8`（当前最佳正式配置：tsh warm 125、unit ramp 710、multiplier 4、λ_inst=0.05、U→R W=10、tail 3e-6、3 epochs×710）。
- PGSR 配置是独立 fork（`..._mbm_w10_t3e6_pgsr_ddp8`），不要再作为推进方向。
- m0/m4/full3/causal-fork 都是历史对照，不能覆盖。

---

## 10. 数据与评测协议

- ScanNet 数据：`/space/mawb/tokengs/data/scannet_prompt/`（训练 manifest：`scannet_c3g8_train_provisional.json`、wide_8x7：`scannet_prompt_full_wide_8x7.json`）。
- 训练 loader：`scannet_prompt_small`，train size 5680 samples，test（eval）24 samples；256×256。
- LSM-40：`data/scannet_prompt/lsm_instance_eval_manifest.json`，40 scenes、8 context + 7 target、stride-10 interleaved。
- 隔离：训练 manifest 的 `excluded_eval_scenes`=LSM-40 40 scenes，交集=0（已核对）。
- AP：每 scene target views 汇成一个匹配池，COCO 式 101-point interpolated AP；`mean_*` 是 scene-macro 平均，`pooled_*` 是全场景 pooled；预测以 masked group argmax + mean-group-prob score。
- 与 InstOk3D 公开协议差异：当前用 ScanNet `.sens` 相机位姿，而论文用 COCO colmap；评测代码中已记录该差异（`PROTOCOL NOTE`）。**不允许用 LSM-40 调超参数**；开发用训练排除后的固定 validation（`tsh_mbm_lsm4` 等 4-scene smoke 只作机制验证）。

---

## 11. 集群环境与命令

240 实际环境（多次 smoke 验证）：
- repo：`/space/mawb/tokengs`
- python：`/space/mawb/anaconda3/envs/tokengs/bin/python`（Python 3.11、torch 2.7.0+cu126、gsplat 1.5.3、safetensors 0.8.0、tyro 1.0.16）
- `CUDA_HOME=/space/mawb/.cuda126`
- `CC=/space/mawb/anaconda3/envs/tokengs/bin/x86_64-conda-linux-gnu-gcc`；`CXX=...x86_64-conda-linux-gnu-g++`
- `TORCH_CUDA_ARCH_LIST=8.6 FAST_COMPILE=1 MAX_JOBS=8`（3090 与 A6000 均 sm_86）
- `LD_LIBRARY_PATH` 含 conda lib + `site-packages/nvidia/{cuda_runtime,cudnn,cublas}/lib`
- `PYTHONPATH=/space/mawb/tokengs/240_shims`
- gsplat JIT：必须先在 torchrun 前单进程执行 `python -c "from gsplat.cuda._backend import _C; print(_C)"`，否则 8 rank 并发编译会崩。
- Slurm：`srun -p a6000/3090 ... bash -lc '...'`；单节点 8 卡用 `torchrun --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=<端口>`；**所有命令加 `--exclude=3dimage-11,3dimage-12`**；3090 分区实际可分配节点是 3dimage-13（11/12 禁用）。

历史参考命令（不要再执行正式 710/2130）：
- Both 正式：`formal_mbm_both_w10_t3e6_train.sh`（`workspace/tsh_mbm_delivery/`）。
- PGSR 正式：`formal_mbm_pgsr_train.sh`（同样在 delivery，不要跑）。
- LSM 并行评测：`eval_mbm_both_w10_t3e6_stage.sh` / `eval_mbm_pgsr_stage.sh`。

---

## 12. DDP 与 checkpoint 语义

- DDP8 每卡 batch=1；dataset 全量 5680 samples/epoch → 每 rank 710 optimizer steps/epoch；3 epochs = 2130 optimizer steps = 17040 global samples。
- schedule 单卡→8卡：tsh warm/ramp 与 u2r warm 都以 **DDP optimizer step** 表达（m4/m0：125/710/1420/2130；PGSR：125/355/710）。
- rank 同步、保存（rank0）、resume 规则：Accelerate 管理；`.ddp_trace` 记录各 rank 数据/参数 hash；中断后重跑同一命令自动 resume workspace（model.safetensors+optimizer.pth+scheduler.pth），有 `tsh_fork_continue_step` 支持公共 warm-up fork（保留未正式运行）。
- 正式 workspace（不要覆盖）：`..._joint_m4_ddp8`、`..._head_only_m0_ddp8`、`..._mbm_both_w10_t3e6_ddp8`、`..._mbm_w10_t3e6_pgsr_ddp8`；`tsh_*` 前缀目录多为 smoke/audit，可读但不要当成正式结果。
- 新增正式实验必须先给新 workspace；不得在已存在 workspace 上误启动。

---

## 13. Git 与工作区状态

- branch：`main`；HEAD：`68435e5ca129dd70953eabe1a75e634c27dbb9e8`；remote：`origin git@gitee.com:fufuforu/tokengs_init.git`。
- `git status --short`：**121 行 modified/untracked**（本轮之前的长期未提交工作树）。
- tracked modified：`options.py`、`train.py`、多个 `tokengs/models/*`、data 与 scripts（大量为历史/本系列修改）。
- untracked：`240_shims/`、`_240_path_remap.py`、大量新 scripts、`data/scannetpp_processed/`、workspace 下文件等。
- 备份已生成：
  - `workspace/codex_handoff_20260907/working_tree.patch`（566,568 bytes）
  - `workspace/codex_handoff_20260907/untracked_manifest.txt`（184,170 行，11,127,603 bytes）
- **不要**执行 `git reset --hard`、`git checkout --`、`git clean`，也不要清理未跟踪文件。

---

## 14. Facts / Hypotheses / Invalidated

### Confirmed facts
- True Shared Unit（唯一 q_abs 同时生成 GS 与 instance assignment）已被结构与梯度审计验证。
- MBM Both@1420 正式 AP50=0.1503、PSNR=20.304，是当前最佳实例 checkpoint。
- PGSR 正式 40 场景无增益。
- 跨视图 matching 不一致：consistency median 0.32、fragmented 0.66、query collision 0.40。
- 查询合并/去重 oracle 无 headroom：best-GT IoU median≈0.29、Recall@50≈0.20。
- ScanNet instance ID 场景级全局、跨 view 保留；训练数据与 LSM-40 无重叠。

### Current hypotheses
- AP50 瓶颈可能在 instance head/loss/优化（而非 unit 粒度或 GS 覆盖）；representation ceiling 尚未正式量化。
- scene-level Hungarian 应改善跨视图一致性，但不一定显著提升 AP50。

### Invalidated
- PGSR（per-GS slot refinement 正式增益）。
- “8 GS 共享 identity 一定限制边界”的强假设（无 oracle 支撑）。
- 早期 abs 评测的 20.12 恒定 PSNR 伪结果。

### Open questions
- Unit/GS representation ceiling（需运行 oracle audit 回答 A/B/C/D）。
- 边界 F-score 的几何上限。
- instance head 容量/训练是否达到 oracle 上限。

### Immediate next actions（只读）
1. 运行 `scripts/audit_unit_gs_representation_ceiling.py`（8 个固定 validation scenes，steps≈120，已 smoke 通过 1 scene）。
2. 对照 `audit_multiview_query_binding.json` / `oracle_multiview_query_audit.json` 汇总报告。
3. 不要启动正式训练；先按 A/B/C/D 判读。

### Do-not-do list
- 不覆盖/删除 full3、m0、m4、Both、PGSR、causal-fork 的 workspace/config/checkpoint。
- 不继续 PGSR、不增大 alpha、不启动 710 步正式训练。
- 不用 LSM-40 调 oracle/hyperparameter；不用单场景 overfit 代替多场景验证。
- 不添加 DINO/CLIP/LSeg semantic/16×4/第二套 units。
- 不执行任何 `git reset --hard` / `checkout` / 清理 untracked。

---

## 给下一位 Agent 的第一条只读命令

```bash
cd /space/mawb/tokengs
# 1) 确认文件与 git 状态
git status --short | head; git rev-parse HEAD
# 2) 确认当前最佳 checkpoint 存在且 key 组成正确
ls -la workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors
# 3) 先读一份已生成的 oracle 结果（勿重跑训练）
head -c 2000 workspace/tsh_mv_binding_audit16_v2/audit_multiview_query_binding.json
head -c 2000 workspace/tsh_mv_oracle/oracle_multiview_query_audit.json
# 4) 下一步只跑只读 ceiling audit（smoke 已通过）：
#    srun -p 3090 --exclude=3dimage-11,3dimage-12 --nodes=1 --ntasks=1 --gpus-per-task=1 \
#        bash workspace/tsh_mbm_run/run_gs_ceiling8.sh
```
