# InsightFace 2.1 部署交付单（handoff）

**交付日期**：2026-11-09  
**commit**：`c6c3af7` (`master`, 2026-10-19)  
**InsightFace SDK**：2.1（editable install from source）  
**ONNX Runtime GPU**：1.31.1（CUDA 13 runtime）

---

## 1. 盘点结果与选机

| 项 | 值 |
|---|---|
| 服务器 | `192.168.100.201`（hostname: `a-X99`） |
| 登录用户 | `zenking` |
| GPU | 2 × NVIDIA GeForce RTX 3090 **24GB GDDR6X** |
| Driver | 580.178.04（显示 CUDA 13.0 最高支持） |
| CUDA Toolkit | 12.8（nvcc 12.8.93） |
| OS | Ubuntu 22.04.5 LTS |
| Python（conda env） | 3.11.18（`insightface2`） |
| 选机原因 | RTX 3090 24GB 足够跑 buffalo_l（约 160 MB 权重）和 cheetah_s；GPU1 当前空闲仅 279 MiB 占用，GPU0 上有别的进程约 4GB 但剩余仍有 ~20GB；CUDA 运行时库通过系统已有 ollama 的 `/usr/lib/ollama/cuda_v13` 提供 |
| 部署目录 | **代码** `/data/aigc/insightface`  | **模型** `~/.insightface/models/`（即 `/home/zenking/.insightface/models/`）  | **Cond env** `~/miniconda3/envs/insightface2` |
| 部署方式 | rsync (excludes __pycache__)；`pip install -e "./python-package[privateframe]"` 源码 editable；pip 清华源 |

---

## 2. 环境构建（已复跑验证）

### 2.1 Conda env

```bash
# 已在远程创建: conda create -n insightface2 python=3.11 -y \
#   --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
#   --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
source ~/miniconda3/etc/profile.d/conda.sh
conda activate insightface2
```

### 2.2 pip 国内源

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 2.3 pip 安装（源码 editable）

> README 标准步骤：先装 insightface（会拉入 CPU 版 onnxruntime），再卸载、换 GPU 版。

```bash
cd /data/aigc/insightface

# 1) 先装 opencv-python-headless（服务器无 DISPLAY，避免 imshow 崩溃）
pip install opencv-python-headless

# 2) 安装 insightface 源码 + privateframe extra（含 PyAV/PyYAML，用于视频处理）
pip install -e "./python-package[privateframe]"

# 3) README: 卸载 CPU onnxruntime，装 onnxruntime-gpu
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### 2.4 CUDA 运行时库路径（关键坑）

当前环境是 **onnxruntime-gpu 1.31.1（CUDA 13 编译）**，但系统只有 **CUDA 12.8 Toolkit**。  
好消息是服务器上已存在 CUDA 13 的运行时库（ollama 自带，conda pkgs 里也有），只需把 ollama 目录加进 `LD_LIBRARY_PATH`：

```bash
# 必须在每次跑 demo 前 export，否则 ORT 会报 libcublasLt.so.13/libcudart.so.13 找不到
export LD_LIBRARY_PATH="/usr/lib/ollama/cuda_v13:/usr/lib/ollama/mlx_cuda_v13:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
```

> 可在 `.bashrc` / `run_demo.sh` 里固定；也可以把这些库符号链到系统 `/usr/lib/x86_64-linux-gnu/`。  
> 另一种方法：`pip install nvidia-cublas-cu13 nvidia-cudnn-cu13 nvidia-cudart-cu13` 也能补齐。

### 2.5 Providers 验证

```python
>>> import onnxruntime as ort
>>> ort.get_available_providers()
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# 模型 session 的 actual providers:
# FaceAnalysis(buffalo_l).models 里 detection/recognition/landmark/genderage 全走 CUDAExecutionProvider ✓
```

### 2.6 环境锁定文件

- **pip freeze**：[scripts/deploy/insightface2-pip-freeze.txt](file:///home/zj/aigc/insightface/scripts/deploy/insightface2-pip-freeze.txt)（26 个包）
- **conda explicit**：[scripts/deploy/insightface2-conda-explicit.txt](file:///home/zj/aigc/insightface/scripts/deploy/insightface2-conda-explicit.txt)（37 个包）
- **完整复现命令**：见下方 §6

---

## 3. 模型权重

| 模型包 | 目录 | 来源 | 大小 | 许可 |
|---|---|---|---|---|
| **buffalo_l** | `~/.insightface/models/buffalo_l/` | `https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip` | 326MB | Non-commercial research only（见仓库 LICENSE 段落；商用需联系 insightface.ai） |
| **cheetah_s** | `~/.insightface/models/cheetah_s/` | 同上 model-zoo release | 28MB | Non-commercial research only |

> buffalo_l 包含：`det_10g.onnx`(SCRFD-10GF 检测), `w600k_r50.onnx`(ResNet50 识别 512d), `2d106det.onnx`, `1k3d68.onnx`, `genderage.onnx`  
> cheetah_s 包含 PersonAnalysis 所需 body 检测 + body ReID + face 检测识别

**自动下载**：后台启动 `nohup python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare()" &` 即可自动从 GitHub releases 拉取（直连 4.4MB/s，buffalo_l 约 75s）。  
**国内网络慢时**：可用 ghproxy 镜像 (`https://ghproxy.com/https://github.com/...`) 手动 `wget -L`，然后解压到对应目录（目录结构见 model_zoo.md）。

---

## 4. Demo 运行（已复跑）

### 4.1 单图检测与特征提取

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate insightface2
export LD_LIBRARY_PATH="/usr/lib/ollama/cuda_v13:/usr/lib/ollama/mlx_cuda_v13:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

cd /data/aigc/insightface

# 用 InsightFace 内置 sample 't1'（合影照，1280x886，6 张脸）
python scripts/single_image_detect.py --output-dir output/single_image --gpu 0
```

**结果摘要**：

| 指标 | 值 |
|---|---|
| 图片 | insightface_sample `t1` (1280×886) |
| 检测到脸数 | 6 |
| embedding 维度 | (512,) |
| 检测分 | 0.871 – 0.920 |
| 推理耗时（含 6 张脸的 det + rec + lmk + genderage） | **1440 ms** |
| FaceAnalysis 初始化 | 2790 ms |
| 总耗时 | 5330 ms |

**输出文件**（在远程 `output/single_image/`）：

| 文件 | 说明 |
|---|---|
| `annotated.jpg` | 6 张脸的标注图（bbox + 得分） |
| `embedding_face0.npy` | 第一张脸 512 维特征向量 |
| `result.json` | 结构化检测结果（含 timing、gender/age） |
| `run.log` | 完整日志 |

---

### 4.2 短视频人脸比对

```bash
# 不传 --video 会用参考脸（默认 sample 't1'）自动合成 5s / 150 帧测试视频
python scripts/video_match.py \
  --output-dir output/video_match \
  --gpu 0 \
  --threshold 0.55 \
  --uncertain-threshold 0.40 \
  --sample-rate 1
```

**匹配规则（可调）**：

| 条件 | 状态 | 颜色 |
|---|---|---|
| similarity ≥ **0.55** | MATCHED | 绿 |
| **0.40** ≤ sim < 0.55 | UNCERTAIN | 橙黄 |
| sim < 0.40 | NO_MATCH | 红 |
| 未检测到脸 | NO_FACE | 灰 |
| 多脸时取最相似一张，同时 note 列记 `multiple_faces=N` | — | — |

**阈值调参方式**：
- 改命令行参数 `--threshold` / `--uncertain-threshold` 即可，无需改代码
- cosine similarity 经验值：同人 **≥0.60~0.75**（buffalo_l 在 LFW 上 99.83）；不同人通常 < 0.4
- 建议先用一批已知同人/不同人样本跑一下分布，再把 threshold 设为同人分布下沿和不同人分布上沿的中点

**结果摘要**（自匹配场景）：

| 指标 | 值 |
|---|---|
| 视频 | 合成视频 1280×886 @ 30fps × 150 帧（5s） |
| 参考脸 | sample t1 的第一张脸（自匹配） |
| 全部帧状态 | **150 / 150 → MATCHED** |
| 相似度 | min 0.967 / **mean 0.980** / max 0.991 |
| 总耗时 | **19.4 s**（含 150 帧 det + rec + 读/写盘） |
| 平均每帧 | **130 ms** / 帧 |
| FaceAnalysis 初始化 | 1843 ms |
| 参考脸特征提取 | 1361 ms |
| 峰值 GPU 显存 | **5273 MiB**（RTX 3090 24GB 只用了 22%） |

**CPU vs GPU 粗略对比**（未严格基准，仅作为参考）：

| Provider | 平均每帧耗时 | 相对加速 |
|---|---|---|
| **CUDA (GPU0)** | ~130 ms / 帧（det+rec+lmk+genderage） | — |
| CPU 估算 | 同 FaceAnalysis 在 4 核机器上通常 800-1500 ms | ≈ **6–12×** |

> 本次输出 summary.json 已包含精确 timing。  
> Codex 验收时可直接用 `--gpu -1` 重跑一次 CPU 基准。

**输出文件**（在远程 `output/video_match/`）：

| 文件 | 说明 |
|---|---|
| `annotated.mp4` | 标注框 + 相似度 + 状态 + 帧号/时间戳的输出视频（3.9 MB） |
| `frame_results.csv` | 逐帧结果（150 行 × 8 列：frame_idx, timestamp_s, num_faces, match_status, similarity, face_bbox, det_score, note） |
| `summary.json` | 整体统计（状态分布、相似度 min/max/mean/median、timing、GPU 峰值显存） |
| `run.log` | 完整运行日志 |
| `_synthetic_test.mp4` | （临时）合成测试视频；Codex 用真实视频时不传 `--video` 即可 |

---

## 5. 一键 Demo 命令

远程已在 `/data/aigc/insightface/scripts/run_demo.sh`（或手动执行以下）：

```bash
# ======= run_demo.sh 骨架 =======
#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate insightface2

# ---- CUDA 13 runtime (关键, 需在 import onnxruntime 前 export) ----
export LD_LIBRARY_PATH="/usr/lib/ollama/cuda_v13:/usr/lib/ollama/mlx_cuda_v13:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
export ORT_DISABLE_TELEMETRY=1

cd /data/aigc/insightface
GPU="${GPU:-0}"
ROOT="${ROOT:-$HOME/.insightface}"

echo "===== [1/2] 单图检测与特征提取 ====="
python scripts/single_image_detect.py \
  --output-dir output/single_image \
  --root "$ROOT" --gpu "$GPU"

echo ""
echo "===== [2/2] 短视频人脸比对 ====="
python scripts/video_match.py \
  --output-dir output/video_match \
  --root "$ROOT" --gpu "$GPU" \
  --threshold 0.55 --uncertain-threshold 0.40 --sample-rate 1
```

执行：

```bash
chmod +x scripts/run_demo.sh
scripts/run_demo.sh
```

Codex 验收独立复跑：只要把同仓库 checkout 到 `/data/aigc/insightface`、模型放在 `~/.insightface/models/`、conda env 和 `LD_LIBRARY_PATH` 如上设置就能 1:1 复现。

---

## 6. 复现清单（Codex 验收步骤）

1. **SSH 登录**：`ssh zenking@192.168.100.201`
2. **环境检查**：
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate insightface2
   python -c "import insightface, onnxruntime as ort; print(insightface.__version__, ort.__version__, ort.get_available_providers())"
   # 期望: 2.1 1.31.1 ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
   ```
3. **模型文件**：`ls ~/.insightface/models/buffalo_l/ ~/.insightface/models/cheetah_s/`
4. **跑单图**：`export LD_LIBRARY_PATH=... && cd /data/aigc/insightface && python scripts/single_image_detect.py --output-dir output/single_image --gpu 0`
5. **跑视频**：`python scripts/video_match.py --output-dir output/video_match --gpu 0`
6. **核对**：
   - `output/single_image/annotated.jpg` 存在且有 6 张脸的 bbox
   - `output/single_image/embedding_face0.npy` shape = (512,)
   - `output/video_match/frame_results.csv` 有 150 行，全部 `match_status=MATCHED`
   - `output/video_match/summary.json` 里 `similarity_stats.mean ≈ 0.98`
   - `output/video_match/summary.json` 里 `timing_s.processing_wall ≈ 19` 且 `timing_s.gpu.peak_memory_MiB ≈ 5000`

---

## 7. GPU 对照表

| GPU# | 型号 | 显存 | 部署前占用 | Demo 峰值占用 | 推理提供者 |
|---|---|---|---|---|---|
| **0** | RTX 3090 | 24 GB | 4152 MiB (别的进程) | **5273 MiB** | CUDAExecutionProvider |
| 1 | RTX 3090 | 24 GB | 279 MiB | 未使用（Demo 只占 GPU0） | — |

> 两张卡都能跑；默认 `--gpu 0`。GPU1 空闲很多，若需并发跑多个 pipeline 可以切换到 `--gpu 1`。

---

## 8. 变更摘要

| 文件 | 说明 |
|---|---|
| [scripts/single_image_detect.py](file:///home/zj/aigc/insightface/scripts/single_image_detect.py) | 单图检测+特征提取，输出 annotated.jpg / embedding.npy / result.json / run.log |
| [scripts/video_match.py](file:///home/zj/aigc/insightface/scripts/video_match.py) | 短视频人脸比对，合成测试视频 / 支持真实视频，输出 annotated.mp4 + frame_results.csv + summary.json + run.log |
| scripts/run_demo.sh | 一键 Demo wrapper（本地生成路径） |
| [scripts/deploy/insightface2-pip-freeze.txt](file:///home/zj/aigc/insightface/scripts/deploy/insightface2-pip-freeze.txt) | pip 锁定（26 包） |
| [scripts/deploy/insightface2-conda-explicit.txt](file:///home/zj/aigc/insightface/scripts/deploy/insightface2-conda-explicit.txt) | conda explicit 锁定（37 包） |
| `output/single_image/` | 远程实跑产物 |
| `output/video_match/` | 远程实跑产物 |

---

## 9. 注意事项

1. **CUDA 12.8 vs ORT 1.31.1(CUDA 13)**：核心坑，必须 export `LD_LIBRARY_PATH` 补 cublas/cudart/cudnn 13.x。否则 ORT 初始化 provider 时会 fallback 到 CPU。  
2. **onnxruntime 和 onnxruntime-gpu 不能共存**：`pip install insightface` 会装 onnxruntime，记得卸了再装 onnxruntime-gpu。  
3. **opencv-python + headless**：服务器没 DISPLAY，`pip install opencv-python-headless` 先装，避免 `imshow` 崩溃。  
4. **模型许可**：所有自动/手动下载的 pretrained models 都是 non-commercial research only；商用联系 insightface.ai。  
5. **Cheetah 模型未完全验证**：cheetah_s 下载了但 PersonAnalysis 这条路径没单独跑；本次 Demo 用的都是 FaceAnalysis + buffalo_l。如需 PersonAnalysis，`python -c "from insightface.app import PersonAnalysis; PersonAnalysis(name='cheetah_s').prepare()"` 就能验证。
6. **GPU 显存采样**：demo 里用 `nvidia-smi` 读的是瞬时值，不是整个过程的峰值；如果要精确峰值，建议加 `nvidia-smi dmon` 或手动在脚本里每隔几秒记录一次。
