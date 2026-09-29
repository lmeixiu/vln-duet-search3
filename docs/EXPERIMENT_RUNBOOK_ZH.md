# GPU 恢复后执行的 DUET 对照实验手册

本手册对应新版入口 `map_nav_src/scripts/run_r2r_experiment.sh`。
旧入口 `run_r2r_weighted_replay.sh` 保留用于复现第一版；它仍会跑 test，不要与新入口混用。
当前阶段先判断收益来自哪里，不扩大 K、不引入 TSMC，也不直接延长到更多训练步数。

## 1. 分支、实验问题与唯一的默认差异

| Git 分支 | variant.env | 训练内容 | 要回答的问题 |
|---|---|---|---|
| `main` | reward | 共享代码与手册，默认奖励回放 | 统一实现入口 |
| `exp/dagger-control` | dagger | 原版 DAgger，没有 replay | 单纯额外微调能提高多少？ |
| `exp/uniform-replay` | uniform | DAgger + 均匀回放 | 更多轨迹监督本身有没有用？ |
| `exp/reward-replay` | reward | DAgger + 奖励加权回放 | 奖励加权是否优于均匀回放？ |

三个实验分支使用同一套实现，仅 `experiments/variant.env` 的默认模式不同。
这样不会因复制三份训练代码而引入额外差异。脚本会读取该文件，启动前打印 variant。
`main` 不是第四个实验组。原始 checkpoint 作为单独的零更新 baseline 评估。

统一设置：同一 BASE_CKPT、训练 annotations、图像特征、模型架构、batch_size=8、
学习率 1e-5、seed=0、额外 5000 步、每 1000 步验证、policy_stop。
两个 replay 分支使用**字节一致的同一份训练轨迹文件**、同样的 replay_weight=0.2。
uniform 权重严格为 1/K，不是用很大的 beta 近似。
reward 使用 beta=1：w_i=softmax(R_i/beta)，同一指令内部归一化。

相同更新次数不等于相同计算量：replay 组每次多做一次回放及反向传播。
记录每组耗时，不能把结果称为等算力比较。uniform 和 reward 的计算结构相同。

## 2. 已有实验记录与这次的目标

用户提供的第一版训练日志：seed=0、K=4、beta=1、replay_weight=0.2、额外 10000 步。
总训练耗时约 249 分钟。下面是 val_unseen 的观测结果，不是多种子均值：

| 状态 | 总 iter 标签 | 额外更新 | SR | SPL | nDTW | 路径长度 |
|---|---:|---:|---:|---:|---:|---:|
| 训练前 | 14001 | 0 | 71.22 | 60.36 | 67.05 | 13.74 |
| 最佳 SPL | 19001 | 5000 | 71.99 | 61.21 | 67.63 | 13.04 |
| 最后 | 24001 | 10000 | 70.80 | 60.56 | 67.42 | 13.06 |

最佳提升为 SR +0.77、SPL +0.85、nDTW +0.58 个百分点。
这只能说明一次运行出现小幅正向信号，不能证明 replay 有效，更不能证明权重有效。
接下来依次完成 dagger 对照、uniform 对照，再决定是否重跑 reward 和追加随机种子。
旧 reward 结果可作初步参考；正式比较建议三个分支都按同一新版流程运行并保存逐指令预测。

## 3. 获取代码与切换分支

如果服务器已经有仓库：

```bash
cd /openbayes/home/vln-duet-search3
git status --short
git fetch origin
git switch exp/dagger-control
git pull --ff-only
cat experiments/variant.env
git log -1 --oneline
```

首次 `git switch exp/dagger-control` 会在同名远端分支唯一时自动建立本地跟踪分支。
如果没有自动建立，用 `git switch --track origin/exp/dagger-control`。
其他分支同理。`git status --short` 非空时先保存自己的改动，切勿为了切分支盲目 reset。

如果还没有仓库：

```bash
git clone https://github.com/lmeixiu/vln-duet-search3.git
cd vln-duet-search3
git switch exp/dagger-control
```

不要在该目录仍有训练进程运行时切换分支。依次跑完一组再切下一组。
如未来需多卡并行，给每个分支使用独立 clone 或 git worktree，不要共用正在变动的代码目录。
数据和输出可以通过下面的绝对路径共用。实验代码只支持单进程单 GPU，不是 DDP。

## 4. 设置环境：每个新终端都要执行

先激活你已经成功运行过 DUET 的环境，并确保 MatterSim 的 PYTHONPATH 设置仍然有效。
以下路径根据你之前日志给出，先确认文件存在：

```bash
cd /openbayes/home/vln-duet-search3
export BASE_CKPT=/openbayes/home/vln-duet-search3/datasets/R2R/trained_models/best_val_unseen
export DATA_ROOT=/openbayes/home/vln-duet-search3/datasets
export OUT_ROOT=/openbayes/home/duet-ablation-v2
export TRAJECTORIES=/openbayes/home/vln-duet-search3/duet-replay-full/train-k4.jsonl
export CUDA_VISIBLE_DEVICES=0
export SEED=0
export RUN_ID=pilot-5k

ls -lh "$BASE_CKPT" "$TRAJECTORIES"
```

这里直接复用第一次全量采集的 train-k4.jsonl，不必重新消耗 GPU 采集。
BASE_CKPT 是**最初的原版微调 checkpoint**，三个训练分支都从它出发，
不要将一个分支的训练结果作为下一个分支的 BASE_CKPT。
默认不恢复 optimizer，三个分支都使用新 optimizer；不是中断训练的精确续跑。

输出放在代码仓库外，切分支不影响数据或实验成果。
环境变量中的路径必须是实际路径，推荐绝对路径；新脚本不使用旧的 OUT_DIR。

## 5. 没有 GPU 时现在就能做的事情

### 查看实际将执行的命令

```bash
DRY_RUN=1 bash map_nav_src/scripts/run_r2r_experiment.sh train
DRY_RUN=1 bash map_nav_src/scripts/run_r2r_experiment.sh eval-best
```

DRY_RUN 不加载模型、不检查数据是否存在、不建立输出目录、不使用 GPU。
先看第一行 variant 是否与当前分支一致，再看 Python 参数和输出路径。
它验证命令组装，不代表真实训练一定能运行。

### 检查已有训练候选的质量

```bash
mkdir -p "$OUT_ROOT/diagnostics"
PYTHONPATH="$PWD/map_nav_src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m r2r.inspect_replay "$TRAJECTORIES" --beta 1.0 \
  --output "$OUT_ROOT/diagnostics/train-k4-beta1.json"
```

这个工具只读 JSONL，使用 Python 标准库，不需要 Torch、MatterSim 或 GPU。
输出含义：

| 字段 | 含义与解读 |
|---|---|
| groups | 指令组数，每组有 K 条候选 |
| unique_action_fraction | 每组不同宏动作序列数/K，再取平均；K=4 时全重复是 0.25，全不同是 1 |
| reward_range | 同一指令最大与最小奖励之差的平均值；接近 0 意味着权重缺少区分 |
| max_weight | 每组最大 softmax 权重的平均值；K=4 时接近 0.25 表示接近均匀 |
| effective_sample_fraction | ESS/K；接近 1 表示均匀，接近 1/K 表示权重集中于一个候选 |
| equal_reward_group_fraction | 奖励基本相等的指令组比例 |
| mean_success | 候选平均成功率，0–1 尺度 |
| oracle_success_at_k | 至少一个候选成功的指令比例；用真值判断，只能作为诊断上界 |
| termination_counts / fractions | 策略 STOP、预算强制 STOP、frontier 耗尽的数量/比例 |

无 GPU 的 Windows 本地环境也可以运行检查工具，但需要先将 JSONL 下载到本地。
本工具不验证所有模拟器状态；完整的加载和回放检查仍由训练入口执行。

## 6. 有 GPU 后的推荐执行顺序

### A. 原模型 baseline，只需评估一次

可以在任意实验分支执行：

```bash
bash map_nav_src/scripts/run_r2r_experiment.sh eval-base
```

结果保存到 `$OUT_ROOT/$RUN_ID/baseline/seed-0/validation/`。
此命令只评估 val_train_seen、val_seen、val_unseen，保存预测，不会进入 test。
三个分支共享该 baseline 输出目录，不要每切一次分支就重新跑一遍。

### B. 纯 DAgger 对照：第一优先级

```bash
git switch exp/dagger-control
cat experiments/variant.env
DRY_RUN=1 bash map_nav_src/scripts/run_r2r_experiment.sh train
ITERS=5000 LOG_EVERY=1000 bash map_nav_src/scripts/run_r2r_experiment.sh train
bash map_nav_src/scripts/run_r2r_experiment.sh eval-best
bash map_nav_src/scripts/run_r2r_experiment.sh eval-last
```

实际不传 `--replay_file`，因此没有 replay 数据加载、回放和额外梯度。
原版 DAgger 含 teacher rollout 与学生采样/示范者监督，其配置与两个 replay 组相同。
输出根目录：`$OUT_ROOT/$RUN_ID/dagger/seed-0/`。

### C. 均匀回放对照

确认上一组已退出，再执行：

```bash
git switch exp/uniform-replay
cat experiments/variant.env
ITERS=5000 LOG_EVERY=1000 bash map_nav_src/scripts/run_r2r_experiment.sh train
bash map_nav_src/scripts/run_r2r_experiment.sh eval-best
bash map_nav_src/scripts/run_r2r_experiment.sh eval-last
```

输出根目录：`$OUT_ROOT/$RUN_ID/uniform/seed-0/`。
每条候选权重为 1/K，均匀抽取候选后的 loss_weight=K*(1/K)=1。
仍读取奖励字段并检查文件合法性，但奖励不影响损失权重。
同样使用 replay_weight=0.2，不能为了此组方便而改成 1。

### D. 奖励加权回放

```bash
git switch exp/reward-replay
cat experiments/variant.env
ITERS=5000 LOG_EVERY=1000 BETA=1.0 REPLAY_WEIGHT=0.2 \
  bash map_nav_src/scripts/run_r2r_experiment.sh train
bash map_nav_src/scripts/run_r2r_experiment.sh eval-best
bash map_nav_src/scripts/run_r2r_experiment.sh eval-last
```

输出根目录：`$OUT_ROOT/$RUN_ID/reward/seed-0/`。
训练目标为 L_DAgger + 0.2*L_replay，其中候选权重是 exp(R/beta) 的归一化值。
不需要重新采集，不在验证时选取多个候选。

### 算力不足时的停止条件

先跑 B。若 B 与旧 reward 结果相当或更好，先分析候选诊断再决定是否跑 C/D。
若 D 优于 B，C 才能进一步确认收益是否来自奖励权重。
不要凭单个 seed 或最佳 checkpoint 的 0.x 个百分点差异宣称显著改进。
第一轮优先看相同固定更新数的 eval-last，同时记录同样验证频率下的 eval-best。

## 7. 若原来的轨迹文件不在：重新采集

```bash
export TRAJECTORIES="$OUT_ROOT/shared/train-k4-seed0.jsonl"
COLLECT_SEED=0 K=4 bash map_nav_src/scripts/run_r2r_experiment.sh collect
```

三个分支都可以采集，采集规则完全一样；只需做一次，再共享文件。
collect 不训练，使用冻结的 BASE_CKPT，temperature=1、自主 STOP、无结束回退。
不要复用 val_unseen 诊断文件训练，加载器会拒绝。
采集的 .partial 是未完成文件，本版不支持续采。

如果是第一次在新机器运行，可单独建立 smoke 输出：

```bash
TRAJECTORIES="$OUT_ROOT/shared/smoke-32.jsonl" LIMIT=32 \
  bash map_nav_src/scripts/run_r2r_experiment.sh collect
RUN_ID=smoke TRAJECTORIES="$OUT_ROOT/shared/smoke-32.jsonl" ITERS=20 LOG_EVERY=10 \
  bash map_nav_src/scripts/run_r2r_experiment.sh train
```

只有均匀/奖励分支的 smoke train 会实际检查回放链路。
这些行首变量只影响这一条命令，不会改变之前 export 的全量 TRAJECTORIES 或 RUN_ID。
smoke 结果只能说明可以运行，不能用于研究结论。

## 8. 输出目录与每个文件代表什么

```text
OUT_ROOT/
├── shared/                             # 共享的固定策略候选
├── diagnostics/                        # 不需要 GPU 的候选质量报告
└── pilot-5k/
    ├── baseline/seed-0/validation/
    ├── dagger/seed-0/
    ├── uniform/seed-0/
    └── reward/seed-0/
        ├── student/
        │   ├── console.log             # 完整标准输出和异常
        │   ├── command.txt             # 实际执行的命令
        │   ├── commit.txt              # 代码提交 SHA
        │   ├── worktree_status.txt     # 空文件表示运行时工作区干净
        │   ├── checkpoint.sha256       # 本次加载的权重校验和
        │   ├── trajectories.sha256     # replay 组的候选文件校验和
        │   ├── run_wall_seconds.txt    # 正常完成后的总耗时（含验证）
        │   ├── logs/training_args.json # 完整训练配置
        │   ├── logs/train.txt          # 各次验证结果和最佳结果
        │   └── ckpts/
        │       ├── best_val_unseen     # 按验证 SPL 选择
        │       └── latest_dict         # 最后一次保存
        ├── eval-best/validation/
        │   ├── logs/valid.txt
        │   ├── logs/validation_args.json
        │   └── preds/
        │       ├── metrics_val_unseen.json   # 指标汇总，可直接对比
        │       └── submit_val_unseen.json    # 每条指令的预测路径
        └── eval-last/validation/        # 最终 checkpoint 的相同输出
```

三个 variant 的目录结构相同。baseline 也保存指标和预测。
文件名 `submit_*.json` 沿用原版命名，不代表上传服务器，也不代表跑了 test。
没有 run_wall_seconds.txt 通常表示运行未正常结束，应先看 console.log。
checkpoint.sha256 可用于确认三个组确实从同一原始权重开始。

## 9. 指标与日志怎么读

| 指标 | 单位/含义 | 方向 |
|---|---|---|
| sr | 成功率，0–100 | 高好 |
| spl | 成功与路径效率，0–100 | 高好 |
| nDTW | 与参考路径匹配程度，0–100 | 高好 |
| oracle_sr | 路径曾到过目标附近的比例 | 仅诊断；高但 SR 低时检查停止行为 |
| nav_error | 最终位置到目标距离，米 | 低好 |
| lengths | 实际物理路径平均长度，米 | 需结合 SR；提前乱停也可能很短 |
| action_steps | 全局宏动作数量 | 不等于物理路径经过的节点数 |
| IL_loss | 原训练代码记录的监督损失均值 | 下降不保证泛化提升 |
| weighted replay loss | 已乘 REPLAY_WEIGHT 的回放损失；uniform 组也沿用此标签 | 不宜直接跨权重设置比较 |
| mean_action_entropy | DAgger 采样中有效动作的平均分布熵，nats | 仅诊断，不是越大或越小越好 |
| sampled_active_actions | 此日志区间统计熵的有效采样决策数 | 不包含 teacher 或 replay 步数 |

新代码去掉了旧日志里误导性的 total_actions=1、max_length=1 和无实际训练意义的 RL 零项。
旧 entropy 上万的数值是统计归一化不适配，不能与新版平均动作熵比较。
iter 标签包含原 checkpoint 已有迭代数，ITERS=5000 指额外更新 5000 次，而非跑到标签 5000。

## 10. 比较模板与判断顺序

先填固定 5000 步 eval-last 表，再另列 eval-best；不要把一组 best 与另一组 last 混比。

| 组别 | seed | checkpoint 规则 | SR | SPL | nDTW | 路径长度 | 总耗时 |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 原权重 | 待填 | 待填 | 待填 | 待填 | 待填 |
| dagger | 0 | last-5k | 待填 | 待填 | 待填 | 待填 | 待填 |
| uniform | 0 | last-5k | 待填 | 待填 | 待填 | 待填 | 待填 |
| reward | 0 | last-5k | 待填 | 待填 | 待填 | 待填 | 待填 |

- dagger-baseline：额外训练的收益。
- uniform-dagger：加入离线轨迹监督的收益，同时计算量也增加。
- reward-uniform：奖励加权相对于均匀监督的收益，这是最关键对照。

若 reward 稳定优于 uniform，再追加 SEED=1 和 SEED=2，其他设置保持不变。
各分支重复 `SEED=1 ... train/eval-best/eval-last` 即可，目录会变成 seed-1。
仍使用同一 TRAJECTORIES，先隔离训练随机性；它不涵盖重新采集候选的随机性。
同种子不同分支也不保证训练过程中随机状态逐步一致，因为回放会额外消耗 dropout 随机数。
正式结论报告多种子均值和波动，并保留逐指令预测用于配对分析。

## 11. 重跑、错误与保护规则

- 输出目录已存在：脚本拒绝运行，避免覆盖或混用结果。
  训练重跑用新 RUN_ID，例如 `RUN_ID=pilot-5k-rerun ... train`，评估也使用相同 RUN_ID。
- 仅重评现有模型：用 `EVAL_TAG=recheck ... eval-best`，不必改 RUN_ID。
- 评估外部 checkpoint：`STUDENT_CKPT=/绝对路径 EVAL_TAG=external ... eval-best`。
- 同一组失败后重启：使用新的 RUN_ID；本版不提供保存 RNG/replay 状态的精确断点恢复。
- replay/model configuration mismatch：首先核对 max_action_len、fusion、特征维度等，不能强行跳过检查。
- 日志停在 val_unseen 后：新入口只跑三个验证集，会显示 Starting/进度；先看进程和 console.log。
- 想跑 test：本轮不需要。新入口不会自动添加 test；不要为了保存预测使用旧 --submit。
- REPLAY_WEIGHT=0：replay 参数检查会拒绝。纯 DAgger 请切 dagger 分支。
- 改学习率、beta、回放强度：均视为新实验，使用新 RUN_ID；先完成默认对照，再调参。

## 12. 代码调用关系与改动定位

```text
experiments/variant.env
  └─ scripts/run_r2r_experiment.sh：组装参数、记录来源、保护输出目录
       ├─ collect / diagnose → r2r/collect_trajectories.py
       │    ├─ GMapNavAgent.rollout(feedback=sample_policy)
       │    └─ 完整轨迹评分 → 分组 JSONL + summary.json
       ├─ train → r2r/main_nav.py: train()
       │    ├─ dagger：仅原版训练
       │    └─ uniform/reward：ReplayBuffer 加载固定候选
       │         └─ Seq2SeqAgent.train()
       │              ├─ 原版 DAgger loss.backward()
       │              ├─ backward_replay()：重建地图、逐动作 CE、加权 backward
       │              └─ 共同梯度裁剪 → 一次 optimizer.step()
       └─ eval-* → r2r/main_nav.py: valid()
            └─ 单轨迹 argmax + policy_stop → 指标 JSON / 逐指令预测
```

`r2r/replay.py`：文件校验、uniform/reward 权重、按指令均匀抽样。
`r2r/agent.py`：自主停止、回放合法性检查、强制终止不计学习损失。
`r2r/env.py`：显式 batch 重置，保证回放从对应指令的起点开始。
`r2r/inspect_replay.py`：CPU 候选诊断，不改变文件或模型。
`r2r/parser.py`：新增 replay_mode、save_predictions、eval_splits。

GPU 不可用时可以运行 CPU 测试与 Shell 语法检查：

```bash
python -m unittest discover -s tests -v
python -m compileall -q map_nav_src/r2r tests
bash -n map_nav_src/scripts/run_r2r_experiment.sh
```

测试包含真实 CPU PyTorch 回放梯度，以及三种 variant 的命令组装检查，
模型和模拟器由小型 fixture 替代。通过这些检查不等于已经完成真实 GPU 三组实验。
