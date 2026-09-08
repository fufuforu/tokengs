# Geometry-Anchored Reconstruction–Instance Decoupled Units

## 1. 结论摘要

本设计针对 TokenGS 当前的核心矛盾：重建产生的 3D 表示已经足以支撑实例分组，但实例监督要么只训练末端 head，要么无约束地改变重建表示并损害 PSNR。

推荐第一版采用路线 A（最小兼容改造），暂命名为 **Geometry-Anchored Reconstruction–Instance Decoupled Units**（GAR-IDU）。路线 A 保留 Both@1420 的 absolute student 和 decoder，新增一个与现有 unit 一一对应的 instance pathway，并用零初始化、门控的 geometry adapter 让 instance 梯度在后续阶段有限地进入共享 unit 几何。正式结构不保留 `GroupQueryMemoryRefiner`。

关键原则：

* `q_abs` 是唯一的 token-aligned 3D unit anchor，不重新从最终 GS 或 mask 建立另一套 units。
* 每个 `(token, unit)` 同时拥有 `z_geo`、`z_app`、`z_inst` 三个槽位；三者有相同的 `[B,T,K]` 索引，但不是同一个可任意回传的 feature tensor。
* geometry 负责空间支撑和可见性；appearance 负责颜色/纹理；instance 负责 unit-to-object assignment、objectness 和 mask quality。
* RGB 梯度只进入 reconstruction/geometry 的允许路径；instance loss 只进入 instance pathway，并通过一个零初始化 geometry adapter 受控进入 `z_geo`，不进入 appearance。
* step 0 保留 Both@1420 的原始 forward，新增分支为零残差；因此 q_abs、GS、RGB 和 base instance masks 可以保持数值一致。

最终选择：

```text
RECOMMENDED_ROUTE: A
PROPOSED_MODEL_NAME: Geometry-Anchored Reconstruction–Instance Decoupled Units (GAR-IDU)
CORE_SHARED_REPRESENTATION: q_abs-derived local units [B, 1024, 8, 256]
DECOUPLED_BRANCHES: geometry / appearance / instance, same unit index, separate projections
INSTANCE_TO_GEOMETRY_GRADIENT: zero-init gated GeometryAdapter, staged and low-LR
STEP0_RECON_IDENTITY: YES, up to floating-point execution order
REUSE_BOTH_1420: YES
REMOVE_QUERY_MEMORY_REFINER: YES
READY_FOR_IMPLEMENTATION: YES, after the read-only design is approved
```

“READY_FOR_IMPLEMENTATION” 只表示设计信息足够开始单独实现；本轮没有实现，也没有训练或评测。

## 2. 当前代码的真实结构审计

### 2.1 真实代码路径

主要路径如下：

* `tokengs/models/semantic_tokengs_v4.py`：共享训练/forward、absolute-student forward、True Shared instance path、RGB loss、Hungarian loss 接入和梯度门控。
* `tokengs/models/semantic_tokengs_v6.py`：模型构造、`AbsoluteUnitDecoder`、`SharedUnitInstanceHead` 的实例化和模块冻结/解冻策略。
* `tokengs/models/absolute_unit_decoder.py`：hidden token 到 8 local units、每 unit 8 个 Gaussian、14 维 GS 属性。
* `tokengs/models/shared_unit_instance_head.py`：q_abs 到 8192 units、100 group queries、unit assignment 和 unit→GS 广播。
* `tokengs/models/instance_group_loss.py`：detached matching cost、BCE/Dice/void/unmatched loss、per-view 或 scene-level Hungarian。
* `tokengs/models/tokengs.py`：GS 重建与 RGB/depth/feature renderer。
* `tokengs/train.py`：单次 forward、总 loss backward、optimizer step、各类 schedule 和 checkpoint。
* `tokengs/options.py`：Both、TSH、MBM、scene-Hungarian 和 refiner 配置。
* `scripts/eval_instance_lsm_protocol.py`：`rendered_instance_group_probability` 到 mask、confidence、AP 的评测链路。

### 2.2 当前真实 tensor 流

下表针对当前 absolute True Shared 路径；B 为 batch，T=1024，K=8，P=8，N=T×K×P=65536，F=256，G=100。正式 Both@1420 的输入为 8 context views 和 7 target views，RGB 图像配置为 256×256，patch size 为 8。

| 阶段 | 模块/函数 | 输入 shape | 输出 shape | 监督 | 梯度来源 |
|---|---|---|---|---|---|
| RGB/Plücker 编码 | `TokenGS.forward_encoder`、`patch_embed`、`patch_plucker_embed` | RGB `[B,8,3,256,256]`、Plücker `[B,8,6,256,256]` | encoder keys/values，view×patch memory；具体序列长度由 `forward_encoder` 的 patch 展开决定 | 无直接实例监督 | Both@1420 中 decoder tail 可由 RGB/相关 loss 更新；encoder、patch embed 冻结 |
| scene token 初始化 | `get_gs_tokens` | encoder latent + decoder camera input | `[B,T,1024]`，T=`num_gs_tokens`=1024，token dim=1024 | RGB 间接监督 | 由 decoder tail/TokenGS 路径决定；static `gs_tokens` 在 MBM recipe 冻结 |
| token transformer | `_forward_abs_hidden` | scene tokens、encoder keys/values | `gs_token_hidden [B,1024,1024]` | RGB；特定配置下 instance/u2r | Both@1420 的 decoder tail 是低 LR；instance 是否进入此处由上游 gate/adapter 决定 |
| local-unit/GS decode | `AbsoluteUnitDecoder.forward` | hidden `[B,T,1024]` | `new_gaussians [B,65536,14]`、`q_abs [B,T,K,F]`、centers `[B,T,K,3]` | RGB；历史 teacher GS distillation | RGB 进入 absolute decoder；当前 TSH mask renderer 对 GS 使用 detach，所以 instance loss不进入该 GS decoder |
| 14D GS字段 | `AbsoluteUnitDecoder` | unit feature + center + slot embedding | 每个 Gaussian `[x,y,z,opacity,sx,sy,sz,qw,qx,qy,qz,r,g,b]` | RGB renderer | RGB 可更新 decoder；instance mask路径使用同一数值 GS，但 renderer 输入被 detach |
| RGB重建 | `TokenGS.render_reconstruction` / `self.gs.render` | GS `[B,N,14]`、target cameras | RGB `[B,7,3,H,W]`、alpha/depth 等 | `lambda_rgb * MSE`；Both 主配置 `lambda_rgb=200` | 进入 GS/允许的 decoder tail；不进入 TSH，因为 TSH 不在 RGB graph 中 |
| unit flatten | `SharedUnitInstanceHead.forward` | `q_abs [B,T,K,256]` | `q [B,8192,256]` | instance loss 间接 | `q_abs` 通过 q adapter 和 instance head；实际传播受 `tsh_unit_grad_eff` 与 multiplier 控制 |
| unit adapter | `q + q_adapter(LN(q))` | `[B,8192,256]` | `z [B,8192,256]` | Hungarian instance loss | TSH 训练时可更新；这不是新的 unit formation，仍使用 q_abs 的原始索引 |
| group queries | `_TSHCrossBlock` ×2 | group queries `[B,100,256]`，units `[B,8192,256]` | groups `[B,100,256]` | instance loss 间接 | TSH 参数可训练；query self-attention + cross-attention |
| unit assignment | `unit_assignment_proj`、`group_assignment_proj`、temperature、`void_head` | z/groups | logits `[B,8192,101]`、`pi_unit [B,T,K,101]` | BCE/Dice/void/unmatched | instance loss；最后一维为 100 groups + void |
| unit→GS assignment | `pi_unit.unsqueeze(-2).expand` | `[B,T,K,101]` | `pi_gs [B,65536,101]` | 无新增监督 | 对 assignment 保持梯度；同一个 unit 的 8 个 GS 使用完全相同 probability |
| instance render | `semantic_tokengs_v4._forward_tsh_instance_branch` + `gs.render_feature_channels` | `student_gaussians.detach()`、`pi_gs`、7 target cameras | rendered probability `[B,101,7,1,H,W]`、alpha `[B,7,1,H,W]` | rendered mask Hungarian loss | 只沿 `pi_gs` 回到 TSH/q_abs；GS 参数路径被 detach |
| Hungarian matching | `hungarian_instance_group_loss` | rendered probability + GT label maps | detached assignment + BCE/Dice/void/unmatched scalar | per-view 或 scene-level 配置 | matching cost detached；matched loss 对 probability 保留梯度 |
| AP评测 | `masks_from_group_probs`、`instance_ap` | `[G+1,H,W]` probability | binary masks、mean probability score、AP25/50/75 | 无训练监督 | inference only；score 是 mask 内非 void 最大 group probability 的平均 |

### 2.3 q_abs、unit、Gaussian 和 query 的真实数量/关系

* `T=1024` scene/GS decoder tokens。
* `K=8` local units per token。
* `P=8` Gaussians per local unit。
* 因此 unit 数为 `8192`，Gaussian 数为 `65536`，且 `8192×8=65536`。
* `q_abs` 为 `[B,1024,8,256]`；它是 `AbsoluteUnitDecoder` 内部 unit feature `q`，不是旧 activation head 的输出。
* 当前 14D 字段是：`0:3` position，`3:4` opacity，`4:7` scale，`7:11` quaternion rotation，`11:14` RGB color。
* TSH 有 100 个可学习 group queries；输出 101 个 assignment channel，其中最后一个是 void。
* 每个 unit 的 assignment 被精确复制给该 unit 的 8 个 GS；当前没有 unit 内 8 GS 的不同 instance assignment。

### 2.4 “True Shared Units”到底共享了什么

当前实现不是完全三流共享：

| 对象 | 当前是否共享 | 说明 |
|---|---|---|
| 空间索引 | 是 | TSH 使用同一个 `(T,K)` unit 索引，unit→GS 是固定 block mapping |
| unit feature | 是，但只限 q_abs 输入 | TSH 直接消费 q_abs；其内部另有 q_adapter，尚不是独立 instance 3D stream |
| decoder | 否 | absolute GS decoder 和 TSH group/query decoder 是两个模块 |
| Gaussian 参数 | forward数值上是 | TSH mask render 使用 absolute student GS；但 instance renderer 对 GS detach |
| renderer | 是 | RGB 和 instance mask 都调用同一 GS renderer 家族，mask 用 feature-channel render |
| 输入 encoder | 部分是 | q_abs/GS 和 TSH 追溯到同一 hidden；当前 TSH 没有独立读取 encoder memory 的 projection |
| instance→geometry梯度 | 当前关闭 | `student_gaussians.detach()` 使实例 loss不能改变 GS decoder；q_abs 梯度还可受 multiplier/gate 控制 |

因此当前结构已经验证了“同一 unit ID 对齐”，但没有充分验证“instance-aware geometry”。这正是下一版要补的最小缺口。

## 3. 已有实验对设计的约束

当前最佳 Both@1420 的 8-scene baseline 为 AP50 0.0891、pooled AP50 0.0716、best-GT IoU 0.3313、Recall@50 0.1865、PSNR 18.6082。

已有结果给出以下硬约束：

1. representation ceiling 已显示 unit 粒度、per-GS identity 和 GS foreground coverage 不是主瓶颈；因此不应把第一版做成更细的 16×4、per-GS identity 或新的 renderer。
2. head+query-memory-refiner fixed batch 可以强烈拟合，但 8-scene Joint@125 的 native AP50 为 0.0787，oracle AP50 0.1415，仍低于 Both oracle 0.1459。说明固定样本可学习不等于跨场景 unit 语义泛化。
3. Joint@125 的 PSNR/SSIM/LPIPS 与 baseline 完全一致，说明 frozen reconstruction 是可行的；但实例路径本身没有带来泛化提升。
4. ranking audit 显示下降集中在少数场景，且 oracle mask 上界也略下降；不能把问题简单归为 confidence calibration。
5. 早期 per-view Hungarian 允许同一 query 在不同 view 产生不同匹配；历史 scene-Hungarian 实验不能直接复用，因为当时 True Shared flag 透传不完整，且正式结果并未证明 scene-level matching 能解决泛化。
6. 过去无保护的联合解冻会破坏 PSNR；所以第一版必须把 appearance 与 instance 梯度隔离，并将进入 geometry 的实例梯度放在零初始化 adapter 后面。
7. 当前 `instance→q_abs` 在相关 probe 中为 0，`U→R` 也为 0；实例监督实际上主要训练 TSH 末端。下一版必须允许一条可测量但受控的 instance→geometry adapter 梯度。

## 4. GlobalSplat / QuerySplat 可迁移原则

只迁移结构原则，不复制代码：

* GlobalSplat 的 “align first, decode later” 对应于先把多视角证据汇聚到 scene/geometry token，再固定 `(token,unit)` 空间槽位，最后分别解码 GS 属性和实例属性。
* GlobalSplat 的 geometry/appearance 双流对应 `z_geo` 与 `z_app`，二者共享 unit index，但不共用同一 projection 和 loss path。
* QuerySplat 的 geometry query → geometry attributes、appearance query 读取 RGB/Plücker memory，可转化为 unit-local geometry anchor 先产生，再由 appearance/instance projections 各自读取。
* QuerySplat 的 slot correspondence 在本设计中不是通过两个独立 query 集合保证，而是由一个显式 `(T,K)` unit table 保证；每个 unit 的所有 GS 共享该 ID。
* coarse-to-fine `1→2→4→8` 暂不加入第一版。当前 Both checkpoint 的 `K=8,P=8` 已有清晰 identity mapping；改变容量会同时改变 checkpoint、renderer 统计和表示上限，无法隔离共享表示假设。

本设计不是“重建分支 + 语义分支”：instance 分支使用的是与重建 GS 一一对应的 3D local units，而不是从最终 mask logits 或独立 semantic token 集合再建对象表示。

## 5. 路线 A/B 比较与唯一选择

| 维度 | 路线 A：最小兼容改造 | 路线 B：完整三流重构 |
|---|---|---|
| 验证共享 local-unit 假设 | 能；在同一 q_abs unit 上加入独立 instance feature 和受控 geometry adapter | 能，但同时改变 token→unit→GS 全链路，难以归因 |
| 复用 Both@1420 | 高；absolute head、GS decoder、decoder tail、TSH 50 keys 均可复用 | 只能部分复用，三流 decoder 大量新参数 |
| step0 identity | 可实现：新增分支为 zero residual/gate=0 | 很难；重新解码 geometry/appearance 会引入执行差异 |
| PSNR 风险 | 低；appearance 冻结，geometry adapter staged 开启并受 reconstruction consistency 保护 | 高；三个 stream 的共同训练会复现历史 geometry/RGB 冲突 |
| instance 影响空间表示 | 可通过唯一 geometry adapter 测量 | 理论上强，但难判断提升来自新表示容量还是共享机制 |
| 实现复杂度 | 中等，新增 unit-aligned projection、adapter、objectness/quality | 高，需要重写 decoder、checkpoint、训练 schedule、评测校验 |
| 显存/计算 | 增量可控，主要是 unit↔encoder-memory projection 和 100 queries | 三流 memory/decoder 全量增加，attention 与 activation 显著增加 |
| 训练成本 | 可从 1420 warm-start，先做少量可判别阶段 | 基本需要重训 reconstruction 或长期蒸馏 |
| GlobalSplat/QuerySplat 差异 | 采用“对齐 unit 后解耦读取”的核心思想 | 更接近完整重构，但并不一定更有科学辨识度 |
| 论文清晰度 | 贡献边界清楚：token-aligned 3D units + controlled instance-to-geometry | 容易被批评为大规模架构堆叠 |

当前实验长期没有跨场景提升，最低成本且最可证伪的选择是路线 A。路线 B 作为后续独立工作，不是当前下一步。

## 6. 推荐模型的 tensor-level 设计

### 6.1 Geometry anchor

复用当前 absolute path：

```text
H = _forward_abs_hidden(model_input)              [B,T,1024]
q_abs, gs_base, c_base = AbsoluteUnitDecoder(H)
q_abs  [B,T,K,256]
gs_base[B,T*K*P,14]
c_base [B,T,K,3]
```

新增 geometry adapter 只在 `q_abs` 的 unit feature 上工作：

```text
z_geo = q_abs + g_geo(s) * A_geo(LN(q_abs))
```

其中 `A_geo: 256→256` 为两层 residual MLP，最后一层 weight 和 bias zero-init；`g_geo(0)=0`，并在 Stage 2 由 0 线性 ramp 到推荐上限 0.1，再保持。初始 `z_geo=q_abs`，所以 base decoder 可以继续产生 `gs_base`。正式实现有两种兼容方式，第一版选用后一种：

```text
gs = gs_base + g_geo * Δgeo(z_geo, slot_emb)
```

`Δgeo` 输出 position/scale/rotation 的受限 residual，zero-init；不直接重写 base 14D。这样 old absolute decoder 的 24 keys 保持 base，新增 adapter 只负责可控 shared-unit adaptation。

Geometry memory 的来源是已有 encoder 的 RGB+Plücker context memory，经独立 `P_geo` 投影后可选地作为 adapter 的输入；第一版默认不增加 geometry cross-attention，以保持 step0 和显存最小。geometry anchor 的主要输入仍为 `q_abs`。

### 6.2 Appearance representation

定义与 geometry 相同索引的 appearance stream：

```text
F_app = concat(
    P_app_rgb(encoder.values),
    P_app_plucker(encoder.keys or Plücker branch)
)
z_app = D_app(q_abs, F_app, camera_memory)       [B,T,K,256]
```

路线 A 的第一版不让 instance loss使用 `z_app`，也不让 `z_app`改变 reconstruction。为了兼容 Both，保留现有 `AbsoluteUnitDecoder` 产生的 base color，并把 appearance residual 定义为关闭的 optional branch：

```text
gs_app = gs_base + g_app * Δapp(z_app, camera)  
g_app = 0 in all instance adaptation stages
```

如果后续需要 RGB-only appearance refinement，`Δapp` 只能由 RGB loss 更新；第一版不把它加入正式结构，以免把本次 instance 结论与 appearance 容量混淆。概念上 `z_app` 是明确的 unit-aligned appearance representation，工程上先以 frozen/base decoder 作为它的 compatibility implementation。

### 6.3 Instance representation

第一版禁止 DINO、VGGT、CLIP、LSeg、新 VGM 和新的 image foundation encoder。选用已有 encoder 的 RGB+Plücker patch memory 的独立 projection：

```text
F_inst = P_inst(encoder.values)                         [B,M,256]
z_inst0 = LN(q_abs)                                       [B,T,K,256]
z_inst1 = CrossAttn(unit_queries=z_inst0, memory=F_inst)  [B,T,K,256]
z_inst = z_inst0 + g_inst * P_res(z_inst1)                [B,T,K,256]
```

`P_inst` 是独立于 reconstruction 的 1024→256 projection。当前代码中 Plücker 已在 `patch_plucker_embed` 与 RGB patch embedding 相加后进入 encoder，因此第一版不假设存在一个名为 `encoder.plucker_memory` 的独立 tensor；使用实际 `encoder_latent.values` 即可获得已融合的 RGB+Plücker memory。若实现需要保留独立 Plücker证据，必须显式新增并单独审计该 projection，而不是引用不存在的字段。unit query 只按 `(T,K)` 展开，不能重新聚类 GS。`g_inst` step0 为 0，Stage 1 可开启但只更新 instance pathway；它不直接回传到 RGB reconstruction。

为避免对 8192 units 做过大的全局 attention，第一版的 memory 读取按 token 所属的 context patch 区域做 block-local cross-attention，或先将 encoder memory pool 到 `[B,T,256]` 再 broadcast 到 K units。二者必须保持 `(T,K)` 显式索引；建议先采用 pooled `[B,T,256]` 版本，计算量更稳定。

### 6.4 Object/group queries 与 assignment

保留 100 个 object queries 作为初始化兼容，但移除 `GroupQueryMemoryRefiner`。新 assignment path 为：

```text
Q_obj:       [100,256]
Q_obj'       = SelfAttn(Q_obj)                           [B,100,256]
A_inst       = CrossAttn(Q_obj', z_inst.reshape(B,8192,256))
O_obj        = Linear(Q_obj')                            [B,100,1]
Q_obj_quality= Linear(Q_obj')                            [B,100,1]
L_unit       = (W_u z_inst) · (W_q Q_obj') / τ           [B,8192,100]
L_void       = VoidMLP(z_inst)                           [B,8192,1]
π_unit       = softmax([L_unit+L_obj_gate, L_void], -1)  [B,8192,101]
```

更具体地，objectness 是 query-level 的 non-empty prior，不能替代 unit assignment；它只作为训练/诊断项和可选的 score prior。mask quality 是 query-level 或 unit-pool-level 的 IoU estimate，训练目标来自 detached matched IoU，不能反过来改变 matching。

重建 unit 到 Gaussian 的广播严格为：

```text
π_unit  [B,T,K,101]
π_gs    = repeat each unit probability over its P=8 GS slots
        [B,T,K,P,101] -> [B,65536,101]
```

mask renderer 仍使用当前 `gs.render_feature_channels`，输入 target-view cameras，输出 `[B,101,7,1,H,W]`。同一 unit 的 8 个 GS 仍有相同 assignment；object queries 不再从最终 mask logits构造 memory residual，也不保留旧 query-memory refinement。

### 6.5 Opacity 归属

第一版明确选择 **A：opacity 属于 geometry/visibility**。

理由：renderer 的 alpha 决定 GS 是否参与可见性、RGB 合成和 instance feature-channel 合成。当前 `AbsoluteUnitDecoder` 的 14D 输出中 opacity 位于 index 3，`render_feature_channels` 直接使用 GS opacity 做 alpha compositing。把 opacity 归入 appearance 会让一个只为颜色/纹理服务的分支改变 instance mask coverage，并重新引入 RGB/instance 冲突。

因此：

* base opacity 由 geometry/visibility decoder 产生并由 RGB/geometry supervision 保护；
* instance branch 不预测 opacity，也不改变 GS opacity；
* appearance branch 第一版不输出 opacity residual；
* 如果将来需要 view-dependent visibility，只能增加独立、RGB-supervised、bounded 的 geometry visibility adapter，并单独 ablate。

## 7. Stream interaction 与梯度流

选择 **零初始化 gated residual interaction**，不采用全双向 mixer。

允许的前向/反向关系：

| 关系 | 第一版是否允许 | 说明 |
|---|---:|---|
| geometry → appearance | 仅 forward 的 unit index/受控 conditioning | `z_app`可读取 geometry anchor；instance loss不穿过 appearance |
| geometry → instance | 是 | `z_inst`读取 `q_abs/z_geo` 的空间 unit 信息 |
| appearance → geometry | 否 | 防止 RGB texture evidence 无约束改变空间位置/scale |
| instance → geometry | 受控允许 | 仅经 zero-init `A_geo`，Stage 2 才开启，base GS 分支保留 |
| instance → appearance | 否 | instance loss 对 appearance branch stop-gradient |
| RGB → instance | 否 | RGB loss不进入 instance pathway |

梯度表：

| Loss | Backbone | q_abs/base geometry | Geometry adapter | Appearance | Instance branch | Object queries |
|---|---:|---:|---:|---:|---:|---:|
| RGB reconstruction | Stage 0/1 0；Stage 2 0 | base frozen；可选保持 0 | 0 | RGB-only 时才允许 | 0 | 0 |
| Instance BCE/Dice | 0 | 0 in Stage 1；Stage 2 仅经 `A_geo` gate | Stage 2 nonzero, low LR | 0 | nonzero | nonzero |
| Objectness | 0 | 0 | 0 | 0 | nonzero | nonzero |
| Mask quality/IoU | 0 | 0 | 0 | 0 | nonzero | nonzero |
| Geometry/RGB preservation | 0 | 0 or RGB-controlled | nonzero but constrained | 0 | 0 | 0 |
| Unit consistency | 0 | optional adapter-only | low | 0 | nonzero | permutation-invariant |

这里的 RGB reconstruction loss 对 `q_abs/base geometry` 的训练边界默认是冻结的；如果 Stage 2 需要让 geometry adapter 接受 RGB anchor，使用独立 preservation loss，而不是让 instance loss直接更新原 absolute decoder。这样能够测量 adapter 的实例收益而不重演“完全解冻联合训练”。

## 8. 监督、matching 和跨视角一致性

### 8.1 监督组成

第一版使用以下监督：

1. **RGB reconstruction**：保留现有 full-image RGB MSE，仅用于保护 base reconstruction；不把它接入 instance branch。
2. **Instance BCE + Dice**：保留当前 soft rendered probability 的主要 mask 监督，但 matching cost detached，matched loss 保持 differentiable。
3. **Void/objectness**：void 监督背景；objectness 监督 query 是否获得至少一个 matched GT，使用 soft target 或 detached assignment，避免空 query 互相竞争不稳定。
4. **Mask quality**：对 matched query 使用 detached `IoU(pred_mask, GT)` 作为 quality target；quality 只用于 score/诊断，不能参与 matching cost 的反向图。
5. **Geometry preservation**：base-vs-new GS 的 position/scale/rotation/opacity consistency，以及 rendered RGB consistency；只约束 geometry adapter 输出。
6. **Local-unit consistency**：同一 unit 的 unit embedding 在不同 context subset 下保持相似；损失作用于 `z_inst`，不直接比较未对齐的 object query index。
7. **Cross-view subset consistency**：两个不同 context subsets 各自 forward 后，以 unit-to-unit affinity/assignment matrix 做对齐，而不是直接对 100 query index 做 L2。

第一版不加新的类别语义、DINO、CLIP、LSeg 或 VGM loss。

### 8.2 matching policy

不直接重跑旧 scene-Hungarian 作为“新方法”。已有 scene-level Hungarian 是一个 matching-scope 改变，但历史版本的 True Shared 透传不完整，不能视为本结构验证。

推荐 matching 为两级：

* **同一 forward 内**：保持当前 per-view Hungarian 作为兼容 baseline，确保 step0 和 Both@1420 assignment 一致。
* **跨 view consistency loss**：不对 query index 做匹配；先计算两次 forward 的 unit affinity matrix `A=[8192,100]`，用 Sinkhorn/soft assignment 或 Hungarian 只在 detached affinity 上获得 permutation，再对 unit-level assignment distributions 做一致性。
* **可选 scene-level supervision**：只有在独立实现中显式保证场景内 query permutation、输入/target view 混合和相同 unit index 对齐后才能加入。它不能被描述为新颖结构贡献，也不能直接复制历史 scene-Hungarian 配置。

形式上，两个 context subsets `a,b` 的一致性可写为：

```text
A_a = softmax(z_inst^a W_u (Q_obj^a W_q)^T)
A_b = softmax(z_inst^b W_u (Q_obj^b W_q)^T)
P*  = Hungarian(detach(A_a A_b^T))
L_aff = || A_a - P*(A_b) ||_1  (P* stop-gradient)
```

若 object queries 被独立初始化或发生强 permutation，使用 assignment matrix 的 permutation-invariant matching；禁止未经对齐直接比较 query 0 与 query 0。

## 9. Checkpoint 兼容与 step0 identity

### 9.1 Both@1420 映射

源 checkpoint：

```text
workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors
```

加载规则：

| 旧内容 | GAR-IDU处理 |
|---|---|
| `absolute_gs_head.*` | 严格加载 24/24 |
| 原 `tsh_instance_head` | 严格加载原始 50/50，作为 base assignment/readout 初始化 |
| decoder tail `enc_dec_backbone.decoder_blocks.*` | 严格加载 324/324 |
| `tsh_instance_head.query_memory_refiner.*` | 不加载；正式模型删除该模块及其 42 keys |
| PGSR / `tsh_slot_refine_head` | absent；不创建、不加载 |
| new `GeometryAdapter` | 新建，最后 residual projection zero-init，gate=0 |
| new `InstanceProjection/CrossAttention` | 新建；其 residual 对 base assignment 的输出 zero-init 或 gate=0 |
| new objectness/quality heads | 新建；初始化为不影响 base mask score 的状态 |
| optional appearance residual | 第一版新建但关闭，或延后到独立 ablation；不改变 base GS |

最终 saved checkpoint 会比源 checkpoint 多新模块 keys；“50/50”指旧 TSH base keys 的加载，不包括新模块。

### 9.2 identity 可达到的程度

目标：在第一个 optimizer step 前，将同一 batch 输入两个 forward：

* `q_abs`：相同，若前向路径和 dtype 完全相同可 bitwise；否则要求 max diff 在浮点误差范围。
* base GS：相同；新增 `Δgeo=0` 且 gate=0。
* RGB：相同；使用相同 GS、camera、renderer。
* base unit logits/masks：相同；新 instance branch 只产生 zero residual，或 base TSH 作为显式 shortcut。
* PSNR/SSIM/LPIPS：相同或仅有 deterministic reduction 的浮点误差。

不能承诺新模型在开启 `g_inst` 后仍 bitwise 相同；那是设计要测量的适应阶段。任何新增 decoder layer、attention 或不同的 execution order 都必须通过 `max_abs_diff` 记录，而不是口头宣称 identity。

## 10. 分阶段训练方案

### Stage 0：identity 与梯度审计（不训练）

* 加载 24/24、50/50、324/324；确认 refiner keys 不存在。
* 两次 fresh forward 比较 q_abs、GS、RGB、base/final logits、rendered masks、instance loss 和 PSNR。
* 运行独立 fresh forward 的 gradient audit：instance-only、RGB-only、total-loss 各自只 backward 一次。
* 检查 instance-only 不进入 base GS/appearance；Stage 2 关闭时 geometry adapter grad 为 0；RGB-only 不进入 instance branch。

### Stage 1：instance pathway warm-up（建议 100–200 optimizer steps）

* 冻结 backbone、absolute GS decoder、decoder tail、base q_abs/GS、appearance。
* 训练新 `InstanceProjection/CrossAttention`、object queries、objectness/quality heads；旧 TSH 50 keys 可先冻结作为 base assignment anchor。
* `g_geo=0`，`g_inst` 只在 instance branch 内开启；assignment 使用当前 per-view Hungarian。
* 使用多 scene DDP 数据；fixed batch 只做 smoke，不能作为泛化成功证据。
* 记录 native/oracle AP、unit assignment consistency、query permutation、void、pred/gt 和 per-scene 指标。

### Stage 2：受控 shared-unit adaptation（建议再 200–500 steps）

* 只解冻 zero-init `GeometryAdapter`；base absolute GS decoder 和 appearance 保持冻结。
* `g_geo` 从 0 ramp 到 0.1；geometry adapter LR 建议 `1e-5`，instance pathway LR `3e-5`，object queries `3e-5`。
* instance loss 对 adapter 的系数建议从 `0.0→0.25` ramp；不能一开始用历史 `unit multiplier=4/32` 直接放大。
* 每步同时加入小权重 geometry preservation：position/scale/rotation/opacity consistency + rendered RGB consistency。该 preservation loss 只约束 adapter，不解冻 base decoder。
* RGB path 继续用 frozen base/固定 RGB anchor；若需要训练 adapter 的 geometry，RGB loss 可只更新 adapter，且必须单独审计 instance/RGB gradient cosine。

推荐初始损失形式：

```text
L = L_rgb_base
  + λ_inst(t) (L_BCE + L_Dice + 0.1 L_void + 0.1 L_unmatched)
  + 0.25 λ_quality L_quality
  + λ_preserve (L_GS_preserve + L_RGB_preserve)
  + 0.05 L_affinity_consistency
```

其中 `L_rgb_base` 在 reconstruction freeze 时是 anchor/metric，不应通过 instance branch 回传；`λ_inst` 不能与旧 probe 的无条件 head stacking 混用。具体数值必须在 Stage 0/1 梯度审计后确认，不在本设计文档中授权正式训练。

### Stage 3：扩大训练的条件

只有 8-scene 正式比较稳定改善且 PSNR 守恒后，才考虑更大规模训练或 LSM-40。第一版不加入 coarse-to-fine `1→2→4→8`；它应是未来独立 ablation。

## 11. 参数量、显存和计算量估算

下面是 implementation planning estimate，不是当前代码运行统计。

* 保留部分：absolute head 24 keys、旧 TSH 50 keys、decoder tail 324 keys、已有 1024×64 GS 和 renderer，参数量与 Both@1420 基本相同。
* 一个 1024→256 projection 约 `0.26M` 参数；若 RGB/Plücker 各一层约 `0.52M`。
* 一个 256→256 两层 residual adapter 约 `0.13M` 参数。
* 100 queries×256 约 `0.026M`，2 层 query self/cross attention + FFN 约 `2–4M`，取决于是否复用旧 TSH block。
* objectness/quality head 小于 `0.1M`。
* 总新增参数预计约 `3–6M`，远小于 218M 级 TokenGS backbone；新增 activation 主要是 `[B,8192,256]` instance unit tensor 和 100-query attention。
* pooled memory 版本的 instance cross-attention 复杂度约 `O(B·8192·T·256)`，block-local 版本约 `O(B·8192·M_local·256)`；不建议第一版对全部 view×patch memory 做无约束全局 attention。
* 保存 inference masks 时仍是 65536 GS 的 renderer 计算；新 branch 不增加 GS 数量、不改变 8×8 unit/GS 粒度。

## 12. 必要 preflight 与验证矩阵

所有梯度审计都必须使用独立 fresh forward，不能 `retain_graph=True`：

1. checkpoint strict load：24/24、50/50、324/324；refiner 42 keys absent；PGSR absent；fresh reset=false。
2. step0 equivalence：q_abs、base GS、RGB、unit logits、rendered masks、instance loss、PSNR。
3. forward shape：`q_abs [B,1024,8,256]`、`z_geo/z_app/z_inst [B,1024,8,256]`、`π_unit [B,1024,8,101]`、`π_gs [B,65536,101]`、render `[B,101,7,1,H,W]`。
4. unit block identity：每个 unit 的 8 个 GS assignment 完全相同；不能出现隐式 per-GS query。
5. instance-only backward：instance/objectness/quality 梯度非零；base absolute GS、appearance、backbone 梯度为 0；Stage 1 geometry adapter 为 0，Stage 2 仅 adapter 非零。
6. RGB-only backward：只允许被授权的 reconstruction/adapter 参数非零；instance branch/object queries 为 0。
7. total backward：一次 backward、finite、optimizer step 后参数 hash 在 DDP ranks 一致；无 second-backward、unused-parameter、collective hang。
8. cross-view consistency：先做 permutation matching；报告 unit affinity consistency，不报告未对齐 query index L2。
9. no-collapse：active/non-empty query、effective query count、void、pred/gt、assignment entropy、GT coverage 全部记录。
10. no-hidden path：`teacher_called=0`（正式 instance adaptation）、`old_gs_head_calls=0`、`gaussians_source=absolute_student`，且没有旧 query-memory refiner forward。

## 13. 8-scene 与 LSM-40 判据

正式 8-scene 比较必须固定 Both@1420、相同 manifest、相同 8+7 views 和相同 mask 后处理。至少输出：

* mean/pooled AP25、AP50、AP75；
* best-GT IoU、Recall@25/50/75；
* pred/gt、void、non-empty queries、effective query count；
* assignment entropy、unit consistency、query permutation；
* PSNR/SSIM/LPIPS；
* 8 场景逐场景差值、场景胜负数；
* native score 与 oracle ranking 的差值，防止把 ranking 偶然提升当作 mask 提升。

建议进入 LSM-40 的最低条件：

* mean AP50 至少相对 0.0891 提高 0.02；
* pooled AP50 同步提高；
* best-GT IoU 与 Recall@50 同步提高；
* 至少 5/8 场景 AP50 不下降；
* PSNR 下降不超过 0.1 dB，SSIM/LPIPS 无异常；
* 无 query/void collapse；
* oracle AP50 不低于 baseline，且提升不能只来自 score ranking；
* geometry adapter 的 instance gradient 与 RGB preservation gradient 没有持续严重冲突。

停止条件包括：fixed batch 提升而未见场景不提升、oracle AP50 不提高、PSNR 持续下降、改进只来自少数场景、instance stream 仍只是重新读取固定 q_abs，或 unit-to-GS 对应关系被破坏。

## 14. 与相关工作的差异和论文定位

### TokenGS

TokenGS 的强项是多视图输入到 Gaussian reconstruction；当前版本的 q_abs 是 reconstruction-shaped local unit。GAR-IDU 将其变为明确的 token-aligned 3D anchor，并在相同 unit ID 上派生 instance representation，同时保留重建保护边界。

### GlobalSplat

GAR-IDU 采用 GlobalSplat 的先对齐后解码、geometry/appearance 解耦和受控交互原则，但不复制其完整 scene latent 或 coarse-to-fine Gaussian capacity。我们的核心 slot 是 TokenGS 的 `(token, local-unit)`，并且每个 slot 可追溯到固定的 8 GS。

### QuerySplat

GAR-IDU 不是简单把 QuerySplat 的 appearance branch 改名为 semantic branch。QuerySplat 的 query 主要用于 geometry/appearance 解码；这里的 `z_inst` 是与每个 reconstruction unit 同 index 的 instance feature，object query 只在其上做 grouping。geometry 先提供空间 anchor，instance 不能任意重写 appearance。

### InstOk3D / InstSplat

它们的 anchor/group token 主要是实例分组或 mask 解码 token；GAR-IDU 的核心对应关系在显式 3D local unit：同一个 unit 同时产生 GS 属性、instance embedding 和 assignment，assignment 再无歧义地广播到该 unit 的 8 GS。区别不是“有无 group query”，而是 GS、空间 unit 和 instance assignment 是否共享可追溯的 3D slot。

### 当前 Both/TSH

Both/TSH 已有 q_abs、100 queries 和 unit→GS 广播，但 instance render 对 GS detach，且没有独立 encoder-memory instance projection。因此它验证了 index/renderer 对齐，没有验证 instance-conditioned shared geometry。GAR-IDU 保留其 checkpoint-compatible base，同时补上独立 instance stream 和唯一受控 geometry adapter。

### 已失败的 query-memory refiner

旧 refiner 在已有 assignment 上做 query-memory residual。它在 fixed batch 可学，却没有建立新的跨场景 unit evidence，且 8-scene oracle AP50 仍下降。GAR-IDU 不在最终 mask logits后再做 memory refinement，而是在 unit feature 层提供独立的 instance evidence，并用跨-view unit affinity consistency 处理 permutation。

### 可发表贡献与必要 ablation

潜在贡献是：

1. token-aligned local 3D units 作为 reconstruction/instance 的共同空间索引；
2. geometry、appearance、instance feature 的同-slot解耦，而不是两套无对应 token；
3. zero-init gated instance-to-geometry adapter，在不破坏 reconstruction identity 的前提下让实例监督塑造空间 anchor；
4. permutation-aware unit consistency，避免 query identity 随 view 置换。

必要 ablation：

* base TSH head vs 新 instance stream；
* 新 instance stream 但 `g_geo=0` vs 开启 geometry adapter；
* geometry adapter zero-init vs random-init；
* RGB preservation on/off；
* pooled encoder memory vs q_abs-only；
* per-view Hungarian vs permutation-aware cross-view consistency；
* appearance branch完全冻结 vs RGB-only appearance residual；
* unit→8 GS strict broadcast vs 允许 per-GS assignment；
* 参数量匹配的末端 head control。

最可能的审稿人质疑：

1. 提升是否只是增加了参数量，而不是 unit alignment：需要 parameter-matched controls 和严格 unit broadcast ablation。
2. zero-init/gate 是否只是训练技巧：需要报告 instance→adapter 梯度、geometry preservation、gate schedule 和去掉 gate 的失败对照。
3. query permutation、ScanNet camera/protocol 和 8-scene 选择是否造成结果偏差：需要 permutation-invariant consistency、固定 manifest、LSM-40 后续验证和 raw prediction audit。

## 15. 最小实现文件清单

设计获批后，建议只新增/修改以下最小范围：

1. 新增 `tokengs/models/geometry_anchored_instance_units.py`：geometry adapter、instance projection、unit-aligned object grouping、quality/objectness。
2. 新增或小范围扩展 `AbsoluteUnitDecoder`：保留 base 14D 输出，添加 zero-init bounded geometry residual接口；不得改变旧字段顺序。
3. 小范围扩展 `semantic_tokengs_v4.py`：挂接三流、gradient gate、loss 输出和 cache；保留 old path 作为 identity reference。
4. 小范围扩展 `semantic_tokengs_v6.py`：新增配置开关和模块构造；正式模式不创建 query-memory refiner。
5. 小范围扩展 `options.py`：独立配置名、训练阶段 schedule、所有新开关默认关闭。
6. 新增独立 Stage 0/preflight 脚本和 fixed 8-scene evaluator diagnostics；默认 evaluator 口径不变。

不应修改 `tokengs/utils/instance_ap.py` 的正式 AP 语义；oracle/ranking 仅使用独立诊断路径。

## 16. 主要风险与失败解释

* 如果 Stage 1 instance stream 仍 fixed batch 学得好、8-scene 不提升，说明 encoder memory projection 没有提供跨场景可泛化的对象证据，不能继续堆 attention。
* 如果 Stage 2 只提升 best-IoU 但 PSNR 下降，说明 geometry adapter 仍过强；应停止并分析 preservation/gradient conflict，而不是解冻更多 decoder。
* 如果 geometry adapter 开启后 oracle AP50 仍低于 Both，核心 shared-unit 假设没有被当前监督验证，不能以 confidence calibration 掩盖。
* 如果 query/objectness collapse，先检查 permutation、void 和 unmatched supervision；不要增加 query 数量或再次引入 query-memory refiner。
* 如果 appearance branch 需要 instance loss 才有提升，说明 opacity/visibility 与 RGB 纠缠；第一版应保持 opacity 归属 geometry/visibility，不扩大 appearance 解冻边界。
* 如果新模型无法满足 step0 identity，应回退到 base shortcut，修复 checkpoint/forward wiring 后再讨论训练；不能用重新训练解释 identity 失败。

本设计的失败也应产生明确结论：若在严格 identity、受控梯度和多场景验证下仍不提升，则更可能需要改变 instance supervision/matching 与跨视角约束，而不是继续增加 reconstruction 或 query head 容量。
