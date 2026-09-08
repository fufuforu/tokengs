# GA-IDU v3：actual geometry path、channel anchoring 与 matched-step 归因设计

本版只修正 v2 的计算图和实验归因问题，不改变路线 A，不实现代码，不训练，不评测，也不修改 v1/v2 文档。

## 1. 最终定位与结论

GA-IDU v1/v2 不能称为完整 geometry/appearance/instance 三流。当前 appearance 仍由 Both@1420 的 absolute reconstruction path 产生，新增部分只是 unit-aligned instance stream 以及受控 shared-unit adaptation。

本版的唯一目标是让 instance mask loss 真正可导到“新增的实际 GS geometry residual”，同时保证 base reconstruction 完全冻结：

```text
BASE_SPATIAL_ANCHOR: q_abs [B,1024,8,256]
ADAPTED_SHARED_TENSOR: z_shared [B,1024,8,256]
INSTANCE_GRAD_REACHES_ACTUAL_GS_GEOMETRY: YES
BASE_TSH_GROUP_CHANNELS_ANCHORED: YES
EMPTY_QUERIES_EXCLUDED_OR_DUSTBINNED: YES
CONSISTENCY_PROBABILITY_LOSS_VALID: YES
MATCHED_STEP_CONTROLS_DEFINED: YES
STEP0_IDENTITY: YES
DEAD_GRADIENT_AT_FIRST_ACTIVE_STEP: NO（允许 GeometryResidualHead 内部早层首步为 0）
READY_FOR_IMPLEMENTATION: YES
```

“YES”表示设计已经消除已知的计算图歧义；不表示本轮已经实现或获得实验收益。

## 2. v3 唯一 forward graph

固定 `T=1024`、`K=8`、`P=8`、`N=T×K×P=65536`、`F=256`、`G=100`。完整 graph 为：

```text
H = _forward_abs_hidden(input)                         [B,T,1024]
q_abs, G_base = AbsoluteUnitDecoder(H)                 [B,T,K,F], [B,N,14]

F_encoder = encoder_latent.values                      [B,M,C]
M_inst = MemoryResampler(F_encoder)                    [B,256,256]
delta_z = SharedGeometryAdapter(q_abs, M_inst)          [B,T,K,F]
z_shared = q_abs + s_geo(t) * delta_z                  [B,T,K,F]
                         /\
                        /  \
                       /    \
              InstanceDecoder  GeometryResidualHead
                 [B,T,K,F]          [B,T,K,P,9]
                       |                    |
               assignment residual   bounded xyz/scale/rot residual
                       |                    |
base TSH group logits + residual       G_student geometry
                       |                    |
                 pi_unit [B,T,K,101]     |
                       |                    |
                 pi_gs [B,N,101] ---------+
                                             |
                             render_feature_channels(G_student, pi_gs)
```

`z_shared` 是唯一 adapted shared tensor。InstanceDecoder 和 GeometryResidualHead 必须消费同一个 `z_shared`，不能各自重新建立 unit table。两者输出都保持 `[T,K]` 索引；每个 unit 的 assignment 继续广播给原来的 P=8 个 Gaussians。

### 2.1 Base GS 与 student GS 的组装

当前 14D 顺序为：position `0:3`、opacity `3:4`、scale `4:7`、quaternion `7:11`、color `11:14`。v3 明确拆开梯度边界：

```text
G_base_geo   = stopgrad(concat(
                  G_base[..., 0:3],   # position
                  G_base[..., 4:7],   # scale
                  G_base[..., 7:11],  # quaternion
              ))
G_base_alpha = stopgrad(G_base[..., 3:4])
G_base_color = stopgrad(G_base[..., 11:14])

delta_G_geo  = GeometryResidualHead(z_shared, slot_emb)
              # [B,T,K,P,9] = xyz(3)+scale/log-scale(3)+rotation-tangent(3)

G_student = assemble(
    G_base_geo + s_geo(t) * bound(delta_G_geo),
    G_base_alpha,
    G_base_color,
)
```

rotation residual应使用对 base quaternion 的 3D 局部切空间/axis-angle bounded update，再 normalize；scale residual 作用于 log-scale 并 clamp；position residual使用固定幅度 `a_xyz*tanh`。不新增 opacity residual，不新增 color residual。

关键修正是：instance renderer **不能**再使用 `G_student.detach()`。它必须使用上面的 `G_student`，其中只有 geometry residual 是可导的，base geometry/alpha/color 已 stop-gradient。base absolute decoder producer 也保持 `requires_grad=False`，两层保护同时存在。

## 3. instance loss 到各张量的精确梯度状态

设 `L_inst` 是 rendered probability 上的 BCE/Dice/void/unmatched loss；`pi_gs` 与 `G_student` 都是 instance renderer 的输入。

| 张量/模块 | `L_inst` 梯度 | 原因 |
|---|---:|---|
| `pi_gs` / `pi_unit` | 非零 | assignment probability 保留 computation graph；matching cost 才 detached |
| InstanceDecoder / AssignmentResidual | 非零 | `L_inst → pi_gs → unit logits → z_inst` |
| `z_shared` | 非零 | 同时有 `z_inst` 分支；geometry 分支在 GeometryResidualHead 已有权重后也非零 |
| SharedGeometryAdapter | 非零 | `L_inst → z_inst → z_shared → adapter`；第一 active step 已有 feature path |
| GeometryResidualHead | 非零输出层 | `L_inst → rendered mask → G_student geometry → delta_G_geo`；首步 output projection可非零 |
| GeometryResidualHead 早期层 | 首个 active step 可为 0，随后非零 | zero-init final projection 阻断首步到更早层；output 更新后第二步开启完整 path |
| `G_base` / AbsoluteUnitDecoder | 0 | base fields stop-gradient，producer frozen |
| opacity | 0 | 使用 `G_base_alpha=stopgrad(G_base[...,3:4])`，无 opacity residual |
| color | 0 | 使用 `G_base_color=stopgrad(G_base[...,11:14])`，无 color residual |
| base appearance path | 0 | instance render 只通过 geometry residual 使用 base GS 的空间数值 |

RGB/GS preservation loss 使用同一个 `delta_G_geo`/`G_student`，因此它不会只约束一个未被 mask loss 使用的 phantom geometry branch。

## 4. 首个和第二个 active optimizer step

采用：external `s_geo>0`；SharedGeometryAdapter output projection zero-init；GeometryResidualHead output projection zero-init；base GS fields detached。

### 4.1 Stage 2 第一个 active step

因为两个 residual projection 的输出为 0：

```text
delta_z = 0
z_shared = q_abs
delta_G_geo = 0
G_student = G_base
```

forward 仍是 identity，但 backward 不全是 zero：

1. GeometryResidualHead 的最后输出层从 `L_inst→mask renderer→G_student geometry` 获得非零梯度；zero output 不等于 zero Jacobian 对参数。
2. 如果 GeometryResidualHead 是“非零 trunk + zero-init final projection”，trunk 的梯度首步暂为 0，因为 final weight 为 0；这只是延迟一阶更新，不是永久 dead branch。
3. SharedGeometryAdapter 从 `L_inst→InstanceDecoder(z_shared)→z_shared` 获得非零梯度；adapter zero-init 只使 forward residual 为 0，不会使其 weight gradient 为 0。
4. SharedGeometryAdapter 经 geometry path 的梯度首步为 0，因为 GeometryResidualHead 的 final projection为 0；其 instance path仍非零。
5. base q_abs producer、AbsoluteUnitDecoder、base GS fields、opacity/color保持 0 gradient。

### 4.2 第二个 active step

第一个 optimizer step 后：

* SharedGeometryAdapter output projection 通常非零，`delta_z` 非零；
* GeometryResidualHead final projection 已被 instance geometry gradient 更新，`delta_G_geo` 可非零；
* GeometryResidualHead trunk 开始获得来自 final projection 的梯度；
* `L_inst` 同时通过 `z_inst` 和 `G_student` 两条路径进入 adapter；
* preservation/RGB anchor 也通过 `G_student` 约束相同 residual；
* base q_abs、absolute decoder、alpha 和 color仍不接收 instance gradient。

因此首步允许“GeometryResidualHead 只有输出层启动”，但不能把首步 trunk=0误报为 dead branch；第二步开始完整 geometry residual path 可训练。

## 5. BaseTSH channel anchoring

禁止创建一套无锚定的 100 query table，再把其 logits 任意加到 frozen BaseTSH logits。新分支必须使用 BaseTSH 的 batch-specific group states。

### 5.1 现有代码能提供什么

`SharedUnitInstanceHead.forward` 当前已经产生：

```text
head_out["groups"]          [B,100,256]  # group_norm后的batch-specific states
head_out["base_unit_logits"] [B,8192,101]
head_out["unit_logits"]       [B,8192,101]
```

在 v3 中需要一个最小接口/forward flag，使它明确返回 `groups` 和 `base_unit_logits`，并不运行 `query_memory_refiner`。如果当前 public output 已被下游丢弃，只需增加 `return_base_states=True` 或等价的内部返回字段，不改变 old 50-key state_dict。

### 5.2 新的 channel-aligned residual

原 TSH 50 keys 全部冻结，包括 `group_tokens`、self-attention/cross-attention、projections、temperature 和 void head。新分支不创建新的 learnable `[100,256]` table：

```text
base_group_states = stopgrad(head_out["groups"])       [B,100,256]
base_group_logits = head_out["base_unit_logits"]       [B,8192,101]
base_void_logit   = base_group_logits[...,100:101]      [B,8192,1]

new_group_states = base_group_states + GroupResidualDecoder(
    base_group_states, z_inst
)                                                       [B,100,256]

delta_group_logits = AssignmentResidual(
    z_inst.reshape(B,8192,256), new_group_states
)                                                       [B,8192,100]

final_group_logits = base_group_logits[..., :100] \
                   + s_inst(t) * delta_group_logits
final_logits = concat(final_group_logits, base_void_logit) [B,8192,101]
```

第一版不修改 base void logit；不新增 objectness、quality 或 score head。`s_inst(0)=0` 时 final logits 与 Both base logits严格一致。新 residual 的第 100 个 channel 永远对应原 base 的 void，不存在 channel permutation ambiguity。

每个 unit 的 100 group logits reshape 为 `[B,T,K,100]`，softmax 后 repeat-interleave P=8 得到 `[B,N,101]`。assignment 与原 GS block mapping完全一致。

## 6. MemoryResampler 与 unit alignment

当前真实 encoder memory 是 `encoder_latent.values [B,M,C]`；不能假设 M 与 T=1024一一对应，也不能凭空 pool 成 `[B,1024,256]`。

建议唯一实现：Perceiver-style resampler，`M'=256`，latent dim 256，8 heads、1层 cross-attention + 1层 FFN：

```text
F_proj = Linear(C,256)(F_encoder)                    [B,M,256]
L0     = learned_latents[256,256].expand(B,-1,-1)   [B,256,256]
L1     = L0 + CrossAttn(LN(L0), LN(F_proj), LN(F_proj))
M_inst = L1 + FFN(LN(L1))                            [B,256,256]
```

当前 RGB patch 与 Plücker patch embedding 已在 encoder 前融合；因此 v3 只读取真实 `values`，不引用不存在的 `plucker_memory`。view/camera identity由 encoder sequence中的多视角排列与已融合 Plücker evidence保留；不使用 target-view GT。若未来加入显式 view embedding，必须作为独立 ablation。

8192 unit queries对 `M_inst` 做一层8-head cross-attention：

```text
Q_unit = Linear(z_shared.reshape(B,8192,256))
M_read = CrossAttn(Q_unit, M_inst, M_inst)             [B,8192,256]
z_inst = InstanceDecoder(z_shared, M_read).reshape(B,T,K,256)
```

attention score规模为 `B×8192×256`，B=1时约 2.1M BF16 scalars、约 4 MiB；保存 Q/K/V、projection 和反向临时量约 30–60 MiB，取决于 PyTorch attention kernel。renderer 和 65536 GS activation仍是主要显存项。M'=128可作为显存 ablation，不作为默认设计。

## 7. Cross-view active-query / dustbin consistency

共同 target-view 的 100×100 mask matching保留，但空 query必须显式处理。每个 subset在相同 target cameras `V_c=1或2`上渲染：

```text
S^a, S^b [B,100,V_c,H,W] ∈ [0,1]
m_i^a = mean_{v,h,w}(S^a_i)
m_j^b = mean_{v,h,w}(S^b_j)
active_i^a = (m_i^a >= τ_active)
active_j^b = (m_j^b >= τ_active)
```

建议 `τ_active=0.01`，与当前 active-query 统计的概率质量阈值一致；它不是 learned objectness。

* 双方 inactive：不进入 Hungarian，也不进入 consistency loss。
* 一侧 active、另一侧 inactive：进入 dustbin，产生 `λ_miss * m_active` missing-query penalty；不强行匹配到噪声 query。
* 双方 active：使用 multi-view soft Dice cost。

对 active sets `A_a,A_b`：

```text
D_{ij} = mean_v 2 sum(S^a_{v,i} S^b_{v,j}) /
              (sum(S^a_{v,i}) + sum(S^b_{v,j}) + eps)
C_{ij} = 1 - D_{ij}
eps = 1e-6
```

对 detached `C` 加入 dustbin row/column 后做 Hungarian。真实 active-active pair 的 cost范围 `[0,1]`；active-dustbin cost固定为 `λ_miss`，建议初始 `λ_miss=0.5`；double-empty不生成 pair。

一致性损失只选择一种概率损失：本设计选择 **symmetric pixelwise BCE**，不使用 KL/JS：

```text
BCE_sym(p,q) = 0.5 * [
   -p log(clamp(q,eps,1-eps))
   -(1-p) log(clamp(1-q,eps,1-eps))
   -q log(clamp(p,eps,1-eps))
   -(1-q) log(clamp(1-p,eps,1-eps))
]
```

`p,q` 是 `[0,1]` soft mask probabilities，`eps=1e-6`。对 Hungarian 匹配的 active pair：

```text
w_i = (m_i^a + m_{P*(i)}^b)/2
L_pair = sum_i w_i * mean_{v,h,w} BCE_sym(S^a_{v,i}, S^b_{v,P*(i)})
         / (sum_i w_i + eps)
L_missing = λ_miss * [sum active mass sent to dustbin] /
             (sum active mass + eps)
L_xview = L_pair + L_missing
```

Hungarian permutation `P*` stop-gradient。两个 forward 的 unit index可以发生变化，因为匹配对象是相同 target cameras 下的 rendered query masks；不假设 query index或unit index跨 subset天然相同。额外成本为两次 subset forward、1–2次 target feature-channel render和一次最多100×100 Hungarian。建议每4个 optimizer steps启用一次。

## 8. Matching 选择与历史 scene-Hungarian

GA-IDU-3 正式选择：**保持 per-view Hungarian，只增加上述 cross-view rendered-mask consistency**。

不选择历史 scene-level Hungarian，原因有二：

1. 历史 step20 运行时 True Shared scene-level flag未可靠透传，因此不能作为严格的 scene-Hungarian结论；
2. 即使 flag正确，它只改变 matching scope，没有 channel-anchored BaseTSH residual、MemoryResampler、actual geometry gradient 或空 query dustbin处理，和 GA-IDU-3 的实质目标不同。

因此 v3 不把 scene-level matching重复命名为新贡献，也不直接复用旧实现。

## 9. Matched-step 训练设计

不能把 GA-IDU-1/2/3 各自独立训练710步后直接比较。使用共同 checkpoint 和 matched-step controls：

### Phase A

从 Both@1420 分叉，训练 GA-IDU-1 710 optimizer steps。DDP8、per-rank batch=1 时：

```text
global batch = 8
equivalent global samples = 710 × 8 = 5680
```

每25或50 steps保存。selection scenes用于决定是否进入 Phase B；固定最终8-scene不用于选 checkpoint或调权重。

继续条件：selection mean AP50 相对 Both 至少 +0.01，且 oracle AP50 或 best-GT IoU/Recall@50 至少一项同步提升，多数 selection scenes不下降。训练 loss/fixed-batch不能单独触发继续。

### Phase B

从同一个 GA-IDU-1@710 checkpoint 分叉：

* Control-B：保持 GA-IDU-1，继续710 steps；
* GA-IDU-2：只开启 SharedGeometryAdapter、GeometryResidualHead 和 preservation，继续710 steps。

两者总量均为1420 optimizer steps、11360 global samples。两者必须使用同一 scene manifest、相同 seed、相同 sample order（尽可能通过保存/恢复 dataloader sampler 和每-rank RNG实现）。

### Phase C

从同一个 GA-IDU-2@1420 checkpoint 分叉：

* Control-C：继续 GA-IDU-2，不启用 cross-view consistency，710 steps；
* GA-IDU-3：只加入 active/dustbin rendered-mask consistency，710 steps。

两者总量均为2130 optimizer steps、17040 global samples。

### Optimizer state

每次分叉复制 checkpoint 的 model state、optimizer state、scheduler state、每-rank RNG 和 sampler state。Control 与 treatment 从完全相同的 optimizer moment 开始；新解冻的 GA-IDU-2 geometry parameters 若在 GA-IDU-1 中已存在但 frozen，则其 Adam moment保持为 zero；GA-IDU-1-only 与 geometry group 的参数组定义必须记录。Phase C 新增 consistency 只有 loss path，不新增参数，因此 optimizer state可完全复制。

Control 与 treatment 的唯一差异分别是 geometry adapter/preservation 或 cross-view consistency；不得同时改变 LR、matching、scene order、checkpoint selection 或 evaluator。

## 10. 参数量、显存与计算量

相对 Both@1420 保留 24 absolute keys、50 base TSH keys、324 decoder-tail keys，删除/不创建 query-memory-refiner 42 keys。

新增参数估算：

* MemoryResampler（C=1024→256、256 latents、8 heads、1 layer）：约 1–2M；
* 8192 unit→256 memory cross-attention + InstanceDecoder/FFN：约 2–4M；
* GroupResidualDecoder + channel-aligned AssignmentResidual：约 1–2M；
* SharedGeometryAdapter：约 0.07M；
* GeometryResidualHead + slot projection：约 0.5–1.5M。

总新增约 5–10M 参数，取决于 FFN expansion 和 projection 是否共享；不增加 GS 数量。B=1时 unit-memory attention score约4 MiB BF16，训练 activation约30–60 MiB；主要显存仍来自 65536 GS renderer、7 target views和保存的 backward tensors。GA-IDU-3 额外约两次 subset mask render，每4 step一次以控制平均成本。

## 11. 最小接口修改清单

只读设计确定的最小代码接口，不代表本轮执行：

1. `SharedUnitInstanceHead.forward` 增加明确的 `return_base_states`/等价返回：`groups`、`base_unit_logits`；关闭 query-memory-refiner，原 50 keys 全冻结。
2. 新增 `geometry_anchored_instance_units.py`：MemoryResampler、SharedGeometryAdapter、InstanceDecoder、GroupResidualDecoder、channel-aligned AssignmentResidual、GeometryResidualHead。
3. `semantic_tokengs_v4.py`：
   * 组装 detached base geometry/alpha/color 与 differentiable geometry residual；
   * instance renderer 接收 `G_student`，不再对整个 student GS detach；
   * 单独处理 RGB path和instance path的 gradient boundary。
4. `semantic_tokengs_v6.py`/`options.py`：GA-IDU-0/1/2/3配置开关、schedule 和 independent workspace。
5. 新增 preflight：strict load 24/50/324、42 refiner absent、step0 identity、梯度路径、active/dustbin shape、single backward/DDP hash。
6. 不修改正式 AP score定义；不新增 score/quality head；oracle 诊断保持独立。

## 12. Implementation preflight

### Checkpoint/identity

* `absolute_gs_head 24/24`；base TSH 50/50；decoder tail 324/324。
* query-memory-refiner 42 keys不创建；PGSR absent；fresh reset=false。
* `q_abs`、`G_base`、`G_student`、base logits、final logits、rendered masks、RGB、PSNR全部与 Both@1420 相同（浮点误差范围）。
* step0 `s_geo=0`、`s_inst=0`，base void logit不变。

### Gradient audit

使用三个独立 fresh forward，不能 retain graph：

1. instance-only：assignment/InstanceDecoder、SharedGeometryAdapter、GeometryResidualHead output layer grad按预期非零；base q_abs/absolute decoder/alpha/color/appearance grad=0；
2. RGB-only：base reconstruction path按配置，instance modules grad=0；
3. total-loss：mask renderer实际使用 differentiable `G_student`，一次 backward、clip、optimizer.step，所有 loss/grad/parameter finite。

额外逐边断言：instance loss对 `G_base[...,0:3]`、`G_base[...,4:11]`、opacity、color的 grad为0；对 geometry residual position/scale/rotation为非零；assignment 对 `pi_gs`为非零。

### Matching/consistency

* `S^a/S^b` shape `[B,100,V_c,H,W]`、值域 `[0,1]`；
* active threshold只由 foreground mass得到；无 learned objectness；
* 双空 query跳过；单侧 active进入 dustbin；双侧 active进入 detached 100×100 Hungarian；
* BCE_sym 的 eps、权重和 missing penalty finite；
* unit→P=8 GS block identity恒成立。

## 13. 成功与停止条件

GA-IDU-1 只有在 selection scenes 上相对 Both 至少 AP50 +0.01，且 oracle AP50 或 best-GT IoU/Recall@50同步改善时，才进入 Phase B。

GA-IDU-2 必须相对 matched-step Control-B 改善，而不是只相对 Both 改善；其 actual geometry residual 非零但 base geometry、alpha/color梯度保持 0，PSNR 下降不超过0.1 dB。

GA-IDU-3 必须相对 matched-step Control-C 改善，并降低 query-mask inconsistency；不能用更多训练 steps解释收益。

最终只有满足以下全部条件才运行 LSM-40：

* 固定8-scene mean AP50 ≥ 0.1091；
* pooled AP50 同步提高；
* best-GT IoU 和 Recall@50 同步提高；
* 至少5/8场景AP50不下降；
* PSNR下降≤0.1 dB；
* oracle AP50同步提高；
* 无 query/void collapse，且提升不是只来自 score ranking。

任何阶段若 fixed-batch提升但 selection/8-scene不提升、oracle不提高、actual geometry residual不更新、PSNR下降或改善集中在少数场景，则停止该版本并返回监督/matching诊断，不增加新的 query/refiner 结构。
