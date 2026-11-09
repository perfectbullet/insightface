#!/usr/bin/env bash
# =============================================================================
# InsightFace 2.1 一键 Demo (远程: 192.168.100.201)
# =============================================================================
# 用法:
#   bash scripts/run_demo.sh                 # 用默认 GPU0 跑两个 Demo
#   GPU=1 bash scripts/run_demo.sh           # 换 GPU1
#   ROOT=/custom/model/root bash scripts/run_demo.sh
#   DRY=1 bash scripts/run_demo.sh           # 只打印不执行
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 变量 ----
GPU="${GPU:-0}"
ROOT="${ROOT:-$HOME/.insightface}"
DRY="${DRY:-0}"

# ---- Conda ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate insightface2

# ---- 关键: CUDA 13 runtime (onnxruntime-gpu 1.31.1 需要 libcublasLt.so.13 / libcudart.so.13) ----
# 远程系统已通过 ollama 预装; 如果你换了机器, 可以 pip install nvidia-cublas-cu13 nvidia-cudnn-cu13 nvidia-cudart-cu13
export LD_LIBRARY_PATH="/usr/lib/ollama/cuda_v13:/usr/lib/ollama/mlx_cuda_v13:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export ORT_DISABLE_TELEMETRY=1

echo "============================================================"
echo " InsightFace 2.1 Demo Runner"
echo "============================================================"
echo " Project : $PROJECT_ROOT"
echo " Conda   : $(which python)  ($(python -V 2>&1))"
echo " GPU     : $GPU"
echo " Model   : $ROOT"
echo " ORT     : $(python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())')"
echo "============================================================"

run() {
    echo ""
    echo "\$ $*"
    if [ "$DRY" != "1" ]; then
        eval "$@"
    fi
}

cd "$PROJECT_ROOT"

echo ""
echo "===== [1/2] 单图检测 + 特征提取 ====="
run "python scripts/single_image_detect.py \
      --output-dir output/single_image \
      --root \"$ROOT\" --gpu $GPU"

echo ""
echo "===== [2/2] 短视频人脸比对 ====="
run "python scripts/video_match.py \
      --output-dir output/video_match \
      --root \"$ROOT\" --gpu $GPU \
      --threshold 0.55 --uncertain-threshold 0.40 --sample-rate 1"

echo ""
echo "============================================================"
echo " Demo 完成 ✓"
echo "============================================================"
echo " 单图结果: $PROJECT_ROOT/output/single_image/"
echo " 视频结果: $PROJECT_ROOT/output/video_match/"
ls -lh output/single_image/ 2>/dev/null | sed 's/^/  /'
echo "---"
ls -lh output/video_match/ 2>/dev/null | sed 's/^/  /'
echo ""
echo " 快速校验:"
echo "  head -3  output/video_match/frame_results.csv"
echo "  cat      output/video_match/summary.json"
echo "============================================================"
