# Geometry-Anchored Instance-Decoupled Units（GA-IDU v2）

## 1. 设计结论

本版本修正 v1 的结构表述和三个实现阻断点。第一版不应声称已经实现完整的 geometry/appearance/instance 三流，因为当前 `AbsoluteUnitDecoder` 仍从同一 hidden 联合产生 14D GS，appearance 没有独立参与 forward，实际 appearance 仍是 Both@1420 的 reconstruction 路径。

因此正式名称改为：**Geometry-Anchored Instance-Decoupled Units（GA-IDU）**。它的 v1 实现目标是：保留 geometry/appearance reconstruction base，增加与 `(T,K)` unit 一一对应的 instance stream，并通过受控 shared-unit adapter 让 instance 监督有限影响 geometry。完整 appearance feature decoupling 留作后续工作，不作为本版贡献。

```text
RECOMMENDED_ROUTE: A
ACTUAL_DECOUPLING_IN_V1: 仅有 q_abs 对齐的 instance-side 解耦；没有完整 appearance 解耦
CORE_SHARED_TENSOR: q_abs [B,1024,8,256]
CONSUMED_BY_GEOMETRY: YES（AbsoluteUnitDecoder/base GS path）
CONSUMED_BY_INSTANCE: YES（TSH/GA-IDU instance decoder）
DEAD_GRADIENT_AT_FIRST_ACTIVE_STEP: NO（主要 adapter/instance branch 非 dead；见初始化分析）
MEMORY_ALIGNMENT_DEFINED: YES（Perceiver-style resampler，不假设 token-to-patch 对齐）
PERMUTATION_LOSS_DIMENSIONALLY_VALID: YES（共同 target-view 上的 100×100 query matching）
STEP0_IDENTITY: YES（浮点执行误差范围内）
READY_FOR_IMPLEMENTATION: YES（仅表示设计信息完整，本轮未实现）
```

## 2. 代码事实和实验约束

### 2.1 当前真实路径

事实来自以下实现：

* `tokengs/models/semantic_tokengs_v4.py`：`_forward_abs_hidden`、absolute student forward、`_forward_tsh_instance_branch`、RGB loss 和总 loss。
* `tokengs/models/semantic_tokengs_v6.py`：`AbsoluteUnitDecoder`、TSH 构造和冻结边界。
* `tokengs/models/absolute_unit_decoder.py`：hidden→unit→GS。
* `tokengs/models/shared_unit_instance_head.py`：TSH unit adapter、100 queries、assignment 和 unit→GS 广播。
* `tokengs/models/instance_group_loss.py`：detached matching cost、BCE/Dice/void/unmatched 和 scene-level 分支。
* `tokengs/models/tokengs.py`：`render_reconstruction` 和 GS feature-channel renderer。
* `tokengs/train.py`：每 step 一次 forward、一次 backward、optimizer step。
* `tokengs/options.py`：Both/MBM/scene-Hungarian 和 query-memory 配置。
* `scripts/eval_instance_lsm_protocol.py`：mask、score 和 AP 计算。

当前 absolute True Shared 数据流为：

| 阶段 | 真实函数/模块 | 输入 | 输出 | 监督与梯度 |
|---|---|---|---|---|
| RGB+Plücker patch embedding | `TokenGS.forward_encoder`、`patch_embed`、`patch_plucker_embed` | context RGB `[B,8,3,256,256]`、Plücker `[B,8,6,256,256]` | encoder latent `keys/values`，序列长度 M 由实际 encoder 展开决定 | encoder 在 Both recipe 冻结 |
| scene/GS token transformer | `_forward_abs_hidden` | `get_gs_tokens` 输出 `[B,T,1024]` 与 encoder keys/values | `H=gs_token_hidden [B,1024,1024]` | decoder tail 可由 RGB/授权路径更新，encoder/static GS tokens 冻结 |
| absolute local-unit decode | `AbsoluteUnitDecoder.forward` | H `[B,1024,1024]` | GS `[B,65536,14]`、`q_abs [B,1024,8,256]`、center `[B,1024,8,3]` | RGB 可回传；历史 teacher 只在 teacher-on 阶段使用 |
| RGB reconstruction | `TokenGS.render_reconstruction` | GS `[B,65536,14]`、target cameras | RGB/depth/alpha | RGB MSE 进入 reconstruction/授权 decoder |
| TSH unit flatten | `SharedUnitInstanceHead.forward` | q_abs `[B,1024,8,256]` | units `[B,8192,256]`、z、groups `[B,100,256]` | instance loss 进入 TSH；q_abs 是否可回传由 gate/multiplier控制 |
| assignment | TSH projections + void head | z/groups | logits `[B,8192,101]`、`pi_unit [B,1024,8,101]` | per-view/scene Hungarian BCE+Dice等 |
| unit→GS mapping | `pi_unit.unsqueeze(-2).expand` | `[B,1024,8,101]` | `pi_gs [B,65536,101]` | 每 unit 的 8 GS 完全相同 assignment |
| instance render | `_forward_tsh_instance_branch` | `student_gaussians.detach()`、pi_gs | rendered probability `[B,101,7,1,H,W]` | mask loss 只回传 assignment/q_abs，不回传 GS renderer geometry |
| AP | `masks_from_group_probs`、`instance_ap` | rendered probabilities | binary masks、mean group probability score、AP | inference only |

当前固定数量为 `T=1024` tokens、`K=8` units/token、`P=8` GS/unit、`N=65536` GS、`F=256` unit dim、`G=100` object groups。14D 字段为：position `0:3`、opacity `3:4`、scale `4:7`、quaternion `7:11`、color `11:14`。

当前 True Shared 实际共享：`(T,K)` 空间索引、q_abs feature、同一 GS 数值和 renderer；不共享独立 geometry/appearance decoder。TSH instance render 对 GS detach，所以 instance loss 没有进入 absolute GS decoder。它是“unit-aligned shared input”，不是完整三流模型。

### 2.2 失败实验带来的约束

Both@1420 8-scene baseline 为 mean AP50 0.0891、pooled AP50 0.0716、best-GT IoU 0.3313、Recall@50 0.1865、PSNR 18.6082。

* representation ceiling 表明 8×8 unit/GS 粒度、per-GS identity 和 GS foreground coverage 不是主要瓶颈。
* query-memory-refiner fixed batch 可强拟合，但 Joint@125 的 8-scene native AP50 0.0787、oracle AP50 0.1415，低于 Both oracle 0.1459。
* ranking audit 表明下降由少数场景主导，oracle mask 上界也略降，不能假定只是 score calibration。
* 完全解冻联合训练会损害 PSNR；冻结重建则保持 PSNR，但 instance 只训练末端 head。
* 历史 scene-Hungarian 版本的 True Shared flag 透传不完整，不能把它作为新结构证据。

因此下一版只增加一个可归因的 shared geometry adapter，不重做 reconstruction，不加 query-memory refiner，不增加 per-GS assignment，不加入 DINO/VGGT/CLIP/LSeg。

## 3. 唯一计算图

GA-IDU 必须实现以下唯一图，不能另建没有 `(T,K)` 对应关系的 instance units：

```text
H = _forward_abs_hidden(input)                         [B,T,1024]
q_abs, G_base = AbsoluteUnitDecoder(H)                 [B,T,K,256], [B,N,14]
M_inst = MemoryResampler(F_encoder)                    [B,M',256]
delta_z = SharedGeometryAdapter(q_abs, M_inst)          [B,T,K,256]
z_shared = q_abs + s_geo(t) * delta_z                  [B,T,K,256]

z_inst = InstanceDecoder(z_shared, M_inst)              [B,T,K,256]
delta_G_geo = GeometryResidualHead(z_shared)            [B,T,K,P,7]
G_student = G_base + s_geo(t) * bound(delta_G_geo)      [B,N,14]

unit_logits = BaseTSH(q_abs)
             + s_inst(t) * InstanceAssignmentResidual(z_inst)
pi_unit = softmax(unit_logits + void, dim=-1)           [B,T,K,101]
pi_gs = repeat_each_unit_over_P(pi_unit)                [B,N,101]
mask = render_feature_channels(G_student.detach_for_instance, pi_gs)
```

`z_shared` 必须同时被 `GeometryResidualHead` 和 `InstanceDecoder` 消费。instance loss 通过 `z_inst→z_shared→SharedGeometryAdapter` 回传；reconstruction preservation 通过 `G_student→GeometryResidualHead→z_shared→SharedGeometryAdapter` 形成另一条受控梯度。`q_abs`、`G_base`、base appearance 和原 absolute decoder 在 GA-IDU-1/2/3 中冻结。

实例分支不进入 `G_base` 的 color/appearance；geometry residual 只输出 position/scale/rotation 的 bounded residual，第一版不输出 opacity residual。

### 3.1 GeometryResidualHead 的兼容替代

当前 `AbsoluteUnitDecoder` 已从 unit feature 和 slot embedding 合法地产生完整 14D GS，因此 geometry residual 可以合法地从 q_abs-derived `z_shared` 产生。唯一兼容方案是保留 `G_base` 并新增 unit/slot 对齐的 bounded residual：

```text
delta_xyz  = a_xyz * tanh(W_xyz [z_shared, slot_emb])
delta_logS = a_scale * tanh(W_scale [z_shared, slot_emb])
delta_rot  = a_rot * tanh(W_rot [z_shared, slot_emb])
```

其输出为 `[B,T,K,P,7]`，再按原 8 个 slot 展平到 `[B,65536,7]`。不得把新的 residual head 当成另一套 GS 或重新预测 opacity/color。若实现不能保持上述 unit/slot 形状，则停止实现，不退回到从最终 mask logits 预测几何。

## 4. 初始化：避免 double-zero dead branch

唯一采用方案 **A：residual projection zero-init，gate 使用外部非学习 schedule**。

* `SharedGeometryAdapter` 使用一个 zero-init 的 `Linear(256,256)`，而不是带 zero-init 最后一层的多层 MLP；这样第一 active step adapter weight/bias 本身能得到非零梯度。
* `GeometryResidualHead` 的三个输出 projection zero-init；在 step0 residual 为零。其 reconstruction preservation 梯度在精确 identity 点可以为零，这是预期的保护状态，不是 instance branch dead；先由 adapter/instance path更新，再产生 geometry residual信号。
* `s_geo(t)` 是 trainer 设置的外部 scalar：step0/Stage1 为 0，Stage2 从 0 ramp 到 0.1。它不是 Parameter，不存在 gate 乘 residual projection 的 double-zero。
* `s_inst(t)` 在 GA-IDU-1 起设为 1（或 0→1 的外部 5-step ramp），新 InstanceDecoder/assignment residual 的输出 projection 不同时 zero-gate；至少有一条 assignment path 在首个 active step 有梯度。

在 Stage2 第一个 active step：

* forward 仍为 `z_shared=q_abs`，`G_student=G_base`，因此 identity 保持；
* instance loss 对 InstanceDecoder、assignment residual 和 object queries 有非零梯度；
* instance loss 对 zero-init `SharedGeometryAdapter` 的 weight/bias 有非零梯度，因为其导数是 upstream gradient 与 q_abs 输入的乘积；
* `GeometryResidualHead` 的 preservation 梯度若 base 与 student 完全相同则为零，属于“identity anchor 尚未被移动”的正常行为，不能用它判断整个 branch dead；
* external gate 无梯度，因为它不是 learnable；其 schedule 只控制梯度幅度；
* base q_abs producer、AbsoluteUnitDecoder、base color/opacity 和 appearance branch 仍无 instance gradient。

## 5. Encoder memory：真实对齐方式

不能假设 1024 scene tokens 与 image patches 一一对应，也不能凭空构造 `[B,1024,256]` 的 pooled memory。当前真实 encoder 输出是 `encoder_latent.keys/values`，记为：

```text
F_encoder = encoder_latent.values [B,M,C]
```

RGB 和 Plücker 已在 `patch_embed + patch_plucker_embed` 阶段融合；第一版只读取实际 `values`，不引用不存在的 `plucker_memory` 字段。

### 5.1 MemoryResampler

采用 Perceiver-style learnable memory latents：

```text
L0        [B,M'=256,256]       learnable latent table, broadcast over batch
F_proj    Linear(C,256)(F_encoder) [B,M,256]
L1        = L0 + CrossAttn(LN(L0), LN(F_proj), LN(F_proj)) [B,256,256]
M_inst    = L1 + FFN(LN(L1))                               [B,256,256]
```

建议 `M'=256`、8 heads、1 cross-attention layer + 1 FFN layer。`M'=128` 是显存受限时的 ablation，不是默认设计。resampler 读取多视角 fused RGB+Plücker encoder values；由于 values 的序列保留 view×patch 排列，resampler 通过 learned latents 汇聚所有 context views，而不要求每个 latent 对应某个 scene token。

### 5.2 View/camera identity

当前 encoder values 本身按输入 view 展开，但 GA-IDU-1 不新增 camera-specific appearance feature。建议给每个 view 的 encoder value 加一个已存在/可复用的 view index embedding或由 Plücker提供的 camera ray evidence；不能用 target-view GT。若代码无法取得独立 view index，保留 fused Plücker values 作为几何相机信息，并在文档/实验中明确“不额外注入 view ID”。

### 5.3 Unit queries 读取 memory

由 8192 个 unit query 对 `M_inst` 做 cross-attention：

```text
Q_unit = Linear(z_shared.reshape(B,8192,256)) [B,8192,256]
M_read = CrossAttn(Q_unit, M_inst, M_inst)     [B,8192,256]
z_inst = z_shared + P_inst(M_read).reshape(B,T,K,256)
```

推荐一层、8 heads。attention score 数量为 `B×8192×256`；若 B=1，为 2,097,152 个 scalar，BF16 score/temporary 约 4 MiB，Q/K/V 和 projection activation 约 10–20 MiB，训练反向保存量约 30–60 MiB，不包括 renderer。相比直接对 `[8192, M]` patch memory 做 attention，resampler 将 key/value 长度固定为 256，显存和计算更稳定。

`M_inst` 只提供 evidence；它不生成新的 units。所有输出仍 reshape 回原 `(T,K)`。

## 6. GA-IDU-0/1/2/3 严格嵌套版本

三者必须是同一实现的配置开关，不得写三套模型。

### GA-IDU-0

* Both@1420 原样：不创建 MemoryResampler、InstanceDecoder、adapter 或新 residual。
* 用于 checkpoint/step0 identity reference。
* 训练/评测结果应与 Both@1420 完全相同。

### GA-IDU-1

在 GA-IDU-0 上创建 MemoryResampler、unit-aligned InstanceDecoder 和 assignment residual；`s_geo=0`，`delta_G_geo` 不影响 GS；不加入跨视角一致性。

* base TSH 50 keys 可加载并作为 base assignment shortcut；新 residual 从 zero-init 开始。
* 只训练 instance pathway、assignment residual 和 object queries；base geometry/appearance 冻结。
* 测试独立 encoder evidence 是否改善跨场景 instance 泛化。

### GA-IDU-2

在 GA-IDU-1 上开启 `SharedGeometryAdapter` 和 `GeometryResidualHead`，并加入 geometry/RGB preservation；`s_geo` 外部 ramp 0→0.1。

* base `q_abs/G_base/color/opacity` 仍冻结；只有 adapter/residual 可训练。
* instance loss 通过 z_inst 进入 adapter；preservation 通过 G_student 进入 adapter/residual。
* appearance 不接收 instance gradient。

### GA-IDU-3

在 GA-IDU-2 上加入共同 target-view 的 permutation-aware consistency。除该 loss 外，结构和冻结边界完全不变。

## 7. 正确的 permutation-aware consistency

v1 的 `A_a A_b^T` 是 `[8192,8192]`，不能用于 100 个 object query 的 permutation matching。GA-IDU-3 改为 mask-space matching：

1. context subset a、b 分别 forward，得到各自 `pi_unit^a/pi_unit^b`；
2. 两次 forward 都在相同的 1–2 个 target cameras 上渲染 100 个 soft query masks；
3. 每次 target view 的 soft mask 为 `S^a_v [B,100,H,W]`、`S^b_v [B,100,H,W]`；
4. 聚合 target views 得到 query cost `C [B,100,100]`：

```text
Dice(S_i^a,S_j^b) = 2 <S_i^a,S_j^b> /
                    (sum(S_i^a)+sum(S_j^b)+eps)
C_ij = 1 - mean_v Dice(S^a_{v,i}, S^b_{v,j})
```

5. 对 `detach(C)` 做 100×100 Hungarian，得到 permutation `P*`；
6. 对齐后计算：

```text
L_mask-xview = mean_v,i (1 - Dice(S^a_{v,i}, S^b_{v,P*(i)}))
L_KL-xview   = mean_v,i,pixel KL(S^a_{v,i} || S^b_{v,P*(i)})
L_xview      = L_mask-xview + 0.1 L_KL-xview
```

`P*` stop-gradient。两次 forward 的 unit index可以不同，因为 matching发生在共同 target-view 的 rendered query masks上，而不是假设 unit/query index相等；只要两个 forward 使用相同 target cameras，mask-space matching 就是合法的。unit→GS 映射在每个 forward 内仍严格保持自己的 `(T,K)` 对应。

额外成本为 2 次 subset forward、2×1–2 target-view feature-channel renders，以及每 scene 一次 100×100 Hungarian。建议只在每 4 个 optimizer steps 中启用一次，或只在每步随机抽 1 个 target view；正式比较中必须报告启用频率。

matching 仍选择 **per-view Hungarian + cross-view consistency**，不把历史 scene-Hungarian 直接改名复用。历史 scene-Hungarian 的失败/不可信原因是：旧版本 True Shared flag 透传不完整，且它只改变 matching scope，没有 unit-aligned memory、geometry adapter 或 permutation-aware rendered-mask consistency。GA-IDU-3 的新增点是后者，而不是再次声称 scene-level matching 本身有效。

## 8. Loss 与梯度表

第一版只保留：RGB reconstruction、BCE/Dice instance、void/unmatched、geometry/RGB preservation、unit/assignment consistency 和 GA-IDU-3 的 rendered query consistency。不加入 quality head、objectness score、新 confidence、appearance residual、DINO/VGGT/CLIP/LSeg、per-GS assignment 或 coarse-to-fine。

| Loss | Backbone | q_abs/base geometry | SharedGeometryAdapter | Geometry residual | Appearance/base color | Instance decoder/assignment | Object queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB reconstruction | 0 | 0 | 可选仅adapter | 可选 residual | 0 | 0 | 0 |
| BCE/Dice/void/unmatched | 0 | 0 | Stage2 非零 | 0（不经 mask renderer） | 0 | 非零 | 非零 |
| GS/RGB preservation | 0 | 0 | 非零 | 非零 | 0 | 0 | 0 |
| unit consistency | 0 | 0 | Stage2 可非零 | 0 | 0 | 非零 | permutation-invariant |
| rendered query consistency | 0 | 0 | Stage2 可非零 | 0 | 0 | 非零 | 非零 |

instance render 使用 `G_student.detach()`；这样 instance loss 不进入 base color/appearance 或 GeometryResidualHead 的 GS renderer path。instance loss 进入 geometry 的唯一入口是 `z_inst→z_shared→SharedGeometryAdapter`。preservation loss 不应解冻 base absolute decoder。

## 9. Matching 和监督细节

当前 BCE/Dice matching cost 在 `instance_group_loss.py` 中 detached，matched loss 对 rendered probabilities保留梯度。GA-IDU-1/2/3 保持 per-view Hungarian 作为 primary assignment，以避免将历史 scene-Hungarian 的 scope 变化误认为新结构收益。

跨视角 consistency 必须：

* 用相同 target cameras；
* 在 100 query mask 上做 100×100 matching；
* 对 permutation assignment stop-gradient；
* 不对未对齐 query index 做 L2；
* 记录 matching cost、permutation stability、soft Dice/KL 和 query collision。

Objectness、mask-quality、新 score head 全部关闭。评测继续使用原定义：每个 predicted mask 的 confidence 是 mask 内非 void group probability 的均值，使用原 `instance_ap` 排序；不引入新的 calibration 变量。

## 10. Checkpoint 与 identity

从 Both@1420 加载：

```text
absolute_gs_head: 24/24 strict
base tsh_instance_head: 50/50 strict
decoder tail: 324/324 strict
tsh query-memory refiner: 不创建、不加载，42 keys absent
PGSR/refine head: 不创建、不加载
```

GA-IDU 新模块为 fresh：MemoryResampler、InstanceDecoder、assignment residual、SharedGeometryAdapter、bounded GeometryResidualHead。其新增 residual 在 step0 必须关闭/zero-init。

各版本 step0：

| 量 | GA-IDU-0 | GA-IDU-1 | GA-IDU-2 | GA-IDU-3 |
|---|---:|---:|---:|---:|
| q_abs | = Both | = Both | = Both | = Both |
| G_base | = Both | = Both | = Both | = Both |
| G_student | = Both | = Both | = Both | = Both |
| base unit logits | = Both | = Both | = Both | = Both |
| final unit logits | = Both | = Both（residual gate 0） | = Both（s_inst 0） | = Both（s_inst 0） |
| rendered masks | = Both | = Both | = Both | = Both |
| RGB/PSNR | = Both | = Both | = Both | = Both |
| permutation loss | off | off | off | 0 at identity/reference |

如果新 instance path 在 step0 仍参与 logits，必须使用显式 base shortcut `final_logits=base_logits`；不能依靠随机新分支“恰好接近零”。严格 bitwise identity 只有在复用相同 forward、dtype、renderer execution order 时可承诺；否则以 max abs diff 的浮点误差阈值验收。

## 11. 训练预算和选择规则

100–200 steps 只能叫工程 probe。所有预算同时报告 optimizer steps、world size、per-rank batch、global batch 和 equivalent samples：

```text
equivalent_global_samples = optimizer_steps × world_size × per_rank_batch
```

建议最小可判别预算：

| 阶段 | optimizer steps | DDP8、batch/rank=1 时 global samples | 保存 |
|---|---:|---:|---|
| GA-IDU-0 audit | 0 | 0 | identity JSON |
| GA-IDU-1 warm-up | 710 | 5680 | 0/50/100/.../710 |
| GA-IDU-2 adapter | 710 | 5680 | 0/50/100/.../710 |
| GA-IDU-3 consistency | 710 | 5680 | 0/50/100/.../710 |

总设计预算为 2130 optimizer steps、17040 global samples；这是三个严格嵌套版本各自的判别预算，不应把三阶段 checkpoint 混成一个无归因长训。使用独立 selection scenes 画泛化曲线；最终固定 8-scene 只做里程碑比较，不选择最佳 step。两次评测的 step checkpoint 必须每 25 或 50 steps保存。不能凭 fixed-batch AP 启动长训。

建议 LR 初值：Stage1 新 instance/memory/assignment `3e-5`；Stage2 geometry adapter/residual `1e-5`、instance `3e-5`；base absolute head、decoder tail、backbone 和 appearance `0`。这些是设计初值，不构成正式训练授权。

## 12. 成功/停止判据

GA-IDU-1 成功条件：相对 GA-IDU-0 的 8-scene native 与 oracle mask 指标均无明显退化，且至少 mean AP50 或 best-GT IoU/Recall@50 出现跨场景正向趋势；fixed-batch 不能单独作为成功证据。

GA-IDU-2 成功条件：在 GA-IDU-1 之上 oracle AP50、best-GT IoU 或 Recall@50 提高，native AP50 同步改善，PSNR 下降不超过 0.1 dB，且 geometry adapter 的 update/gradient 有限。

GA-IDU-3 成功条件：在 GA-IDU-2 之上 cross-view query-mask consistency 降低、query permutation 稳定，且 8-scene native/oracle 都没有少数场景主导的退化。

进入 LSM-40 的最低条件：mean AP50 相对 0.0891 提高至少 0.02、pooled AP50 同步提高、best-GT IoU 与 Recall@50 同步提高、至少 5/8 场景 AP50 不下降、PSNR 下降不超过 0.1 dB、无 query/void collapse，且提升不只来自 ranking。

任何一个版本若 fixed batch 提升而 selection/8-scene 不提升、oracle AP50 不提升、PSNR 持续下降、improvement 只集中于少数场景，或 instance path 仍只是读取 q_abs 而没有有效 adapter update，都停止该版本并转向监督/matching 分析，不增加模块。

## 13. 实现文件与 preflight

获批后最小实现范围：

1. 新增 `tokengs/models/geometry_anchored_instance_units.py`：MemoryResampler、unit-aligned InstanceDecoder、assignment residual、SharedGeometryAdapter、bounded geometry residual。
2. 小范围扩展 `semantic_tokengs_v4.py`/`semantic_tokengs_v6.py`：挂接唯一计算图、schedule 和 loss；保留 GA-IDU-0 base shortcut。
3. `options.py` 增加独立配置和 GA-IDU-0/1/2/3 开关，默认关闭；不修改旧配置语义。
4. 新增 identity/gradient preflight 和 permutation audit；不修改正式 AP evaluator 口径。

必须检查：24/24、50/50、324/324；42 refiner keys absent；PGSR absent；q_abs/GS/RGB/logits/masks identity；shape `[B,T,K,256]`、`[B,N,14]`、`[B,T,K,101]`、`[B,N,101]`；instance-only/RGB-only gradient boundary；single backward、finite、DDP hash、无 unused-parameter/deadlock；unit 内 8 GS assignment 恒等；共同 target-view 的 100×100 permutation matching 维度正确。

## 14. 与相关工作的定位

GA-IDU 不是简单把 QuerySplat appearance branch 改成 semantic branch：它的 object grouping 在与 reconstruction GS 一一对应的 `(T,K)` 3D unit 上发生，geometry/instance 使用同一 unit slot 但不同 feature projection 和 loss path。

与 GlobalSplat 的关系是借鉴 align-first/decode-later 和受控 geometry/appearance 分流；不复制其完整 scene latent 或 coarse-to-fine Gaussian capacity。与 QuerySplat 的关系是 geometry anchor 先形成、instance evidence 后读取，而不是两个无对应 query 集合。与 InstOk3D/InstSplat 的差异是 assignment 可追溯到同一个生成 GS 的 local unit，而不只是独立 group token 的 mask 解码。

论文贡献只能声称：token-aligned 3D units、instance-specific memory readout、受控 instance-to-geometry adapter 和 rendered-mask permutation consistency。必须用 parameter-matched head control、GA-IDU-1/2/3 nested ablation、zero-init 对照、preservation on/off、per-view/query consistency 对照证明不是单纯增加参数。

主要风险是：memory evidence 仍不能跨场景泛化；adapter 重新造成 geometry/RGB 冲突；共同 target-view consistency 成本过高；或 unit assignment 虽然对齐但 object semantics 仍不稳定。若失败，应得出“当前监督/matching不能把 q_abs 变成跨场景 object-aware unit representation”的结论，而不是继续堆 query/refiner。
