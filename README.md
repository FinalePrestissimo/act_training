# ACT Training (Real Robot)

这是一个从 `RoboTwin/policy/ACT` 抽离出来的独立训练仓库，专门用于**真机离线数据**训练 ACT。

## 特性

- 与 RoboTwin 的任务系统、仿真环境解耦
- 直接读取 ACT 格式数据目录：`episode_0.hdf5 ... episode_N.hdf5`
- 自动保存：
  - `dataset_stats.pkl`
  - `policy_last.ckpt`
  - `policy_best.ckpt`
  - 中间 checkpoint 与训练曲线图

## 目录结构

- `train_real.py`：训练入口
- `scripts/train_real.sh`：快速启动脚本
- `src/act_training/`：核心训练代码
  - `act_policy.py`
  - `data_utils.py`
  - `detr/`

## 数据格式要求

每个 `episode_i.hdf5` 至少包含：

- `/action`：形状 `[T, action_dim]`
- `/observations/qpos`：形状 `[T, qpos_dim]`
- `/observations/images/<camera_name>`：形状 `[T, H, W, C]`

默认相机名：

- `cam_high`
- `cam_right_wrist`
- `cam_left_wrist`

可通过 `--camera_names` 覆盖。

## 环境与依赖（纯 uv）

```bash
cd act_training
uv lock
uv sync
```


## 训练

### 方式1：一键脚本

```bash
bash scripts/train_real.sh /path/to/your_act_dataset 0
```

### 方式2：直接运行

```bash
uv run python train_real.py \
  --dataset_dir /path/to/your_act_dataset \
  --ckpt_dir ./checkpoints/run1 \
  --policy_class ACT \
  --camera_names cam_high,cam_right_wrist,cam_left_wrist \
  --batch_size 8 \
  --num_epochs 6000 \
  --lr 1e-5 \
  --save_freq 2000 \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --state_dim 14
```

## 说明

- 默认会自动统计 `episode_*.hdf5` 数量（`--num_episodes -1`）。
- 当前数据加载器要求 episode 文件编号连续（`0..N-1`）。
- `state_dim` 参数目前保持与原 ACT 代码一致（默认 14）。
