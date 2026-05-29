# 模型分段与层 Profiling

这个目录现在拆成三段流程：

1. 导出 stage checkpoint
2. 构造模型专属的 260-token 样本缓存
3. 用缓存样本做 stage profiling

## 1. 导出 stage checkpoint

从完整 HF checkpoint 读取目标层区间权重，生成小的 `stage_model.safetensors`，并默认复制 `config.json`、tokenizer 相关文件到输出目录。

```bash
python export_stage_checkpoint.py \
  --model-family qwen2 \
  --model-path /data/models/Qwen2.5-3B \
  --start-layer 0 \
  --end-layer 25 \
  --output ./stage_checkpoints/qwen2_0_25 \
  --torch-dtype source \
  --overwrite
```

```bash
python export_stage_checkpoint.py \
  --model-family llama \
  --model-path /data/models/llama-2-7b \
  --start-layer 0 \
  --end-layer 7 \
  --output ./stage_checkpoints/llama2_0_7 \
  --torch-dtype source \
  --overwrite
```

导出 4bit 存储、16bit compute 的量化 stage：

```bash
python export_stage_checkpoint.py \
  --model-family qwen2 \
  --model-path /data/models/Qwen2.5-3B \
  --start-layer 0 \
  --end-layer 7 \
  --output ./stage_checkpoints/qwen2_0_7_nf4 \
  --torch-dtype source \
  --quantization bnb_4bit_nf4 \
  --bnb-compute-dtype float16 \
  --overwrite
```

```bash
python export_stage_checkpoint.py \
  --model-family llama \
  --model-path /data/models/llama-2-7b \
  --start-layer 0 \
  --end-layer 10 \
  --output ./stage_checkpoints/llama2_7b_0_10_nf4 \
  --torch-dtype source \
  --quantization bnb_4bit_nf4 \
  --bnb-compute-dtype float16 \
  --overwrite
```

导出包含完整 Qwen decoder、final norm 和 `lm_head` 的全局 profiling checkpoint：

```bash
python export_stage_checkpoint.py \
  --model-family qwen2 \
  --model-path /data/models/Qwen2.5-3B \
  --start-layer 0 \
  --end-layer 35 \
  --include-lm-head \
  --output ./stage_checkpoints/qwen2_0_35_full \
  --torch-dtype source \
  --overwrite
```

Qwen2.5-3B 的 `num_hidden_layers=36`，所以 decoder layer 索引是 `0-35`，不是 `0-36`。

## 2. 构造模型专属 260-token 样本缓存

`build_profile_samples.py` 只支持本地已物化的数据集。它会：

- 从本地 LiMA 数据集中提取首轮用户 prompt
- 用目标模型 tokenizer 编码，并 `add_special_tokens=True`
- 跳过 token 长度 `<260` 的样本
- 截断 token 长度 `>=260` 的样本到前 `260` 个 token
- 保存成模型专属的 `.pt` 样本缓存

当前本地可用的 LiMA 路径是：

```bash
./dataset/lima_llama2/data
```

这个目录里已经包含 `parquet` 样本文件，可以直接作为 `--dataset-path`。

为 Qwen2.5-3B 构造 LiMA 样本：

```bash
python build_profile_samples.py \
  --dataset-path ./dataset/lima_llama2/data \
  --model-family qwen2 \
  --tokenizer-path /data/models/Qwen2.5-3B \
  --output ./profile_samples/qwen2_lima_len260.pt \
  --target-length 260 \
  --seed 42
```

为 Llama2-7B 构造 LiMA 样本：

```bash
python build_profile_samples.py \
  --dataset-path ./dataset/lima_llama2/data \
  --model-family llama \
  --tokenizer-path /data/models/llama-2-7b \
  --output ./profile_samples/llama2_7b_lima_len260.pt \
  --target-length 260 \
  --seed 42
```

如果 `dataset-path` 目录里只有 `README.md` 或缓存目录，没有真正的 dataset 文件，脚本会直接报错。

## 3. 用缓存样本做 stage profiling

`profile_segment.py` 现在支持两种入口：

- `--sample-file`：正式实验推荐，用预处理后的 260-token 样本缓存
- `--prompt`：单 prompt quick test，保留向后兼容

### 3.1 用缓存样本做 Qwen stage profiling

```bash
CUDA_VISIBLE_DEVICES=0 python profile_segment.py \
  --model-path ./stage_checkpoints/qwen2_0_4 \
  --sample-file ./profile_samples/qwen2_lima_len260.pt \
  --num-profile-samples 10 \
  --sample-seed 42 \
  --device cuda \
  --torch-dtype auto \
  --print-sample-summary
```

### 3.2 用缓存样本做 Llama stage profiling

```bash
CUDA_VISIBLE_DEVICES=0 python profile_segment.py \
  --model-path ./stage_checkpoints/llama2_7b_0_7 \
  --sample-file ./profile_samples/llama2_7b_lima_len260.pt \
  --num-profile-samples 10 \
  --sample-seed 42 \
  --device cuda \
  --torch-dtype auto \
  --print-sample-summary
```

### 3.3 用缓存样本做 量化 stage profiling

```bash
CUDA_VISIBLE_DEVICES=0 python profile_segment.py \
  --model-path ./stage_checkpoints/qwen2_0_7_nf4 \
  --sample-file ./profile_samples/qwen2_lima_len260.pt \
  --num-profile-samples 10 \
  --sample-seed 42 \
  --device cuda \
  --torch-dtype float16
```

```bash
CUDA_VISIBLE_DEVICES=0 python profile_segment.py \
  --model-path ./stage_checkpoints/llama2_7b_0_7_nf4 \
  --sample-file ./profile_samples/llama2_7b_lima_len260.pt \
  --num-profile-samples 10 \
  --sample-seed 42 \
  --device cuda \
  --torch-dtype float16
```

### 3.4 单 prompt quick test

```bash
CUDA_VISIBLE_DEVICES=0 python profile_segment.py \
  --model-path ./stage_checkpoints/qwen2_0_7 \
  --prompt "请继续写下去：人工智能正在改变" \
  --device cuda \
  --torch-dtype auto
```

### 3.5 全局 Qwen profiling

```bash
CUDA_VISIBLE_DEVICES=0 python profile_segment.py \
  --model-path ./stage_checkpoints/qwen2_0_35_full \
  --sample-file ./profile_samples/qwen2_lima_len260.pt \
  --num-profile-samples 10 \
  --sample-seed 42 \
  --device cuda \
  --torch-dtype auto \
  --print-sample-summary
```

这个模式会输出：

- `embed_tokens`
- `local_layer_00 / global_layer_00` 到 `local_layer_35 / global_layer_35`
- `norm`
- `lm_head`

如果需要同时观察 PyTorch CUDA allocator 内存，并启用主动 OOM 阈值检查，可以使用 `profile_segment_active_oom.py`：

```bash
CUDA_VISIBLE_DEVICES=0 python profile_segment_active_oom.py \
  --model-path ./stage_checkpoints/qwen2_0_35_full \
  --sample-file ./profile_samples/qwen2_lima_len260.pt \
  --num-profile-samples 10 \
  --sample-seed 42 \
  --device cuda \
  --torch-dtype auto \
  --print-sample-summary \
  --print-cuda-memory \
  --active-cuda-oom-limit-mib 4096 \
  --active-cuda-oom-metric max_reserved
```

## 结果口径

使用 `--sample-file` 时，profiling 口径固定为：

- token 长度固定 `260`
- `batch_size=1`
- 每次只处理一条样本
- 对每条样本单独执行完整 warmup 和 measure
- `warmup_runs` 只用于预热，不进入最终平均值
- `per-sample average` 和 `overall average` 只统计 `measure_runs`
- 每个 profiling target 输出 `avg ms`、`param MiB` 和 `forward_peak MiB`，其中 `MiB` 使用二进制单位（`1 MiB = 1,048,576 bytes`）
  - `avg ms`：该 target 的平均执行时间
  - `param MiB`：该 target 自己拥有的参数权重常驻内存
  - `forward_peak MiB`：该 target forward 期间 PyTorch CUDA `max_memory_allocated - memory_allocated_before_target` 的平均峰值增量；这是运行时临时/输出张量的峰值增量
  - 如果 `lm_head` 与 `embed_tokens` 共享权重，`lm_head` 的 `param MiB` 只统计额外自有参数，因此通常为 `0.00 MiB`
- 输出：
  - `embed_tokens` average
  - 每个 decoder layer average
  - 如果 checkpoint 使用 `--include-lm-head` 导出，还会输出 `norm` 和 `lm_head` average
  - `per-sample average`
  - `overall average across sampled inputs`
