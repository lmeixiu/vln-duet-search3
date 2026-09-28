# 第一版：DUET 自主采样与奖励加权回放

本仓库基于官方 [VLN-DUET](https://github.com/cshizhe/VLN-DUET)，保留其提交历史。
本版实现 R2R、单 GPU、离线固定策略采集，再进行奖励加权行为克隆。
测试时执行单条贪心轨迹，不需要多候选搜索，不增加模型参数。
这是待实验验证的实现，不代表已经取得 SR/SPL 提升。

## 方法

参考 [A Gallery of Methods Beyond RL — Part I](https://shengyu-feng.github.io/blog/2026/08/27/beyond-RL/) 的分布重加权思路：

```text
p*(trajectory | instruction) ∝ p0(trajectory | instruction) exp(R / beta)
R = success_weight * Success + ndtw_weight * nDTW + spl_weight * SPL
w_i = softmax_i(R_i / beta)        # 同一指令的 K 个候选之间归一化
L_replay = -sum_i w_i sum_t log pi_student(action_i,t | history_i,t)
L_total = L_original_DUET + replay_weight * L_replay
```

三个奖励指标均使用 `[0, 1]`，默认系数均为 1，默认 `beta=1.0`、`replay_weight=0.2`。
采集使用固定 checkpoint、`eval()`、`no_grad()`、temperature=1 的完整 categorical 分布。
没有 top-k、调温或混合探索，因此 proposal 就是固定策略 p0，无须 p0/q 修正。
有限 K 的归一化权重仅是自归一化重要性采样近似，不能保证精确采到目标分布。

训练时均匀抽取指令，再在该指令的 K 个候选中均匀抽取一条，乘以 `K*w_i`。
因此每次更新只回放一个 minibatch，期望梯度对应上述候选加权目标。
动作损失按轨迹求和、按 batch 平均，不做轨迹长度归一化。
原版 IL/DAgger 先 backward，再累积 replay 梯度，最后共同裁剪并执行一次 optimizer.step。

## 前置条件

沿用原版 Linux/CUDA/MatterSim 环境与数据准备，见仓库根目录 README：

- R2R annotations、connectivity、ViT-base 图像特征；
- 可以正常运行原版 DUET 的 Python 环境；
- **已微调的导航 checkpoint**，即原版 `save()` 生成、含 `vln_bert` 与 `critic` 的文件；
- 原版模型初始化需要可访问或已缓存的 BERT 配置。

`BASE_CKPT` 不是预训练阶段的裸 state_dict；后者对应原版 `--bert_ckpt_file`。
先确认基线能运行，再采集和训练。脚本默认架构与官方 `run_r2r.sh` 一致。
若你的基线配置不同，修改脚本 `common` 参数，并对采集、训练、评估保持一致。
v1 不支持 REVERIE/R4R、DDP 或 `--aug` 与 replay 混用，会直接报错。

## 运行

从仓库根目录：

```bash
export BASE_CKPT=/absolute/path/to/R2R/trained_models/best_val_unseen
export DATA_ROOT=/absolute/path/to/datasets
export OUT_DIR=/absolute/path/to/experiments/duet-replay-v1
export CUDA_VISIBLE_DEVICES=0

# 1. 小规模冒烟测试：32 条训练指令，每条 4 个候选。
LIMIT=32 bash map_nav_src/scripts/run_r2r_weighted_replay.sh collect
ITERS=20 LOG_EVERY=10 bash map_nav_src/scripts/run_r2r_weighted_replay.sh train

# 2. 训练前后都用自主 STOP，单次贪心执行，保存预测。
bash map_nav_src/scripts/run_r2r_weighted_replay.sh eval
```

运行脚本内部会切换到 `map_nav_src`，建议所有路径使用绝对路径。
上述 `train` 的 20 步仅用于确认可以运行，不用于判断效果。
`eval` 沿用原版 `--submit`，同时生成验证集和 test 预测；需准备相应 annotations。
同一 eval 输出目录中已有预测会被原版入口跳过，重评请使用新的 `OUT_DIR` 并设置 `STUDENT_CKPT`。

正式采集使用新的输出目录，避免混用小样本缓存：

```bash
export OUT_DIR=/absolute/path/to/experiments/duet-replay-full
K=4 bash map_nav_src/scripts/run_r2r_weighted_replay.sh collect
ITERS=10000 BETA=1.0 REPLAY_WEIGHT=0.2 \
  bash map_nav_src/scripts/run_r2r_weighted_replay.sh train
bash map_nav_src/scripts/run_r2r_weighted_replay.sh eval
```

可通过 `BATCH_SIZE`、`SEED`、`K`、`TRAJECTORIES`、`BETA`、`REPLAY_WEIGHT`、
`ITERS`、`LOG_EVERY`、`STUDENT_CKPT` 覆盖默认值。
默认轨迹文件名是 `train-k4.jsonl`；改变 K 时建议同时设置 `TRAJECTORIES`，真实 K 以文件元数据为准。
重复运行采集不会覆盖已存在的文件。中断会保留 `.partial` 文件；本版不支持续采，
请检查中断原因后选择新的输出文件。正式 JSONL 仅在采集完成后生成。

自定义奖励参数可直接运行 Python 入口，在相同架构参数上追加：

```bash
# 在 map_nav_src 下；完整架构参数参考 scripts/run_r2r_weighted_replay.sh。
python -m r2r.collect_trajectories --help
# --reward_success 1.0 --reward_ndtw 1.0 --reward_spl 1.0
```

## STOP 和路径语义

原版 `feedback=sample` 是训练模式，会按真值目标位置判停。
新 `sample_policy` 使用策略自己选择的 STOP，从不调用 teacher 选择动作或停止。
训练真值仅在一条采样轨迹结束后计算奖励，不参与采样决策。
原版模仿学习本身仍按其设计使用训练真值监督。

采集和回放均关闭结束后的历史 STOP 节点回退，奖励只评价实际记录的路径。
宏动作到全局节点所经过的中间节点仍计入路径长度；这里沿用 DUET 的全局动作语义。

- 普通 STOP：是策略动作，记录 log-prob，并参加回放监督。
- 达到动作上限、全局 frontier 耗尽：确定性的强制 STOP，log-prob=0，不参加回放监督。
- 与原版预算一致，`max_action_len=15` 最多包含 14 次移动和一次终止步骤。
- 已停止的 batch 行不再记录动作，不再贡献 loss。

`--policy_stop` 将贪心评估也切换到上述终止规则；它不改变原版 teacher/DAgger 训练行为。
为公平比较，脚本对 baseline 与 student **都**使用此开关。
不带开关的原版评估仍保持历史 STOP 回退行为，不能与自主 STOP 指标直接混为一组对照。

## 文件格式与一致性

JSONL 首行记录版本、数据 split、策略约定、采样数、模型/地图配置、checkpoint SHA256、
随机种子和奖励系数；之后每行是一条指令的完整 K 个候选。
候选保存宏动作目标 viewpoint ID、当时位置/视角、行为 log-prob、STOP 原因、实际路径和奖励。
采用 viewpoint ID 而非候选数组下标，避免候选排序变化造成标签错位。

回放从该指令的原始起点和朝向启动模拟器，重新编码观测并逐步构造 `GraphMap`，
使用当前学生参数求动作概率。加载器核对指令签名和配置；rollout 核对每步位置、视角、
动作合法性、STOP 和最终物理路径。不一致时立即报错。
轨迹文件不会把已观测到的地图直接注入未来步骤，也不会跨候选共享地图。

仅 `split=train` 且来自当前训练 annotations 的轨迹允许训练。
采集器可以显式采集 val_seen/val_unseen 进行诊断，但这些文件会被训练加载器拒绝。
SHA256 是来源记录，不是完整的实验恢复状态；原版 checkpoint 不保存 RNG 或 replay 抽样器状态，
因此从中断恢复不会保证与不中断训练逐步一致。

## 建议的第一组实验

1. 原版 DUET，单条 greedy，自主 STOP。
2. 相同训练步数、相同训练集的额外 IL/DAgger 微调，单条 greedy，自主 STOP。
3. 同样配置加 replay，单条 greedy，自主 STOP。

第二组用于排除“只是多训练了若干步”的解释；它与第三组每步计算量仍不同，需同时报告训练时间。
记录 SR、SPL、nDTW、训练时长和离线采集的环境交互数；多种子验证后再判断收益。

诊断候选覆盖能力：

```bash
LIMIT=100 K=4 bash map_nav_src/scripts/run_r2r_weighted_replay.sh diagnose
```

伴随 `.summary.json` 给出 mean success、oracle success@K、mean reward、best reward、mean SPL/nDTW。
**oracle success@K 使用真值选候选，只是诊断上界，不是可部署策略的成功率。**
评估后的验证候选不能进入 replay 训练。采样数 K 对应额外环境交互，不能作为免费测试算力。

## 验证范围

在仓库根目录运行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q map_nav_src/r2r tests
bash -n map_nav_src/scripts/run_r2r_weighted_replay.sh
```

单元测试依赖 NumPy 与 CPU PyTorch，替换模型构建和模拟器，执行真实 rollout、图更新、
加权损失、反向传播和优化器更新；无需 MatterSim 数据或 GPU。
它们不能替代完整 MatterSim/CUDA 冒烟测试与训练实验。

本版同时修复了恢复训练时迭代区间和 checkpoint epoch 记录的问题，避免载入一个大于
本次训练步数的 checkpoint 后不执行预期训练；并对评估禁用梯度以减少内存开销。
