#!/usr/bin/env python3
"""
短视频人脸比对 Demo
=====================
逐帧 (或按 sample-rate 降频) 检测视频中的人脸，与参考脸做相似度匹配。

输出:
  - annotated.mp4          标注框 + 相似度 + 匹配状态的输出视频
  - frame_results.csv      逐帧结果: frame_idx, timestamp, num_faces,
                           match_status, similarity, face_bbox, note
  - run.log                运行日志 + 各阶段耗时
  - summary.json           整体统计

匹配规则 (可调):
  similarity >= threshold  => MATCHED
  0 < similarity < threshold => UNCERTAIN
  未检测到脸 / 无法提取特征   => NO_FACE / MULTI_FACE / NO_EMBEDDING

用法:
  # 用 InsightFace 内置 sample image 自动生成测试视频 (自匹配)
  python scripts/video_match.py --output-dir output/video_match --gpu 0

  # 指定参考脸和视频
  python scripts/video_match.py \
    --reference /data/photos/alice.jpg \
    --video /data/clips/meeting.mp4 \
    --output-dir output/video_match \
    --threshold 0.55 \
    --sample-rate 5 \
    --gpu 0
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ============================================================
# 数据类
# ============================================================
@dataclass
class FrameRecord:
    frame_idx: int
    timestamp_s: float                  # 秒
    num_faces: int                      # 本帧检测到的脸数
    match_status: str                   # MATCHED / UNCERTAIN / NO_FACE / MULTI_FACE / NO_EMBEDDING
    similarity: Optional[float] = None  # 与参考脸的最高相似度 (cosine), 无脸时为 None
    face_bbox: Optional[list] = None    # 最佳匹配脸的 bbox [x1,y1,x2,y2]
    det_score: Optional[float] = None   # 最佳匹配脸的检测分
    note: str = ""


# ============================================================
# Logger
# ============================================================
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("video_match")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================
# 合成测试视频
# ============================================================
def generate_synthetic_video(ref_img: np.ndarray,
                             out_path: Path,
                             n_frames: int = 150,   # 5 秒 @ 30fps
                             fps: float = 30.0) -> Path:
    """
    用参考图合成一段短视频，模拟一个人脸在画面中做微小的平移/缩放/亮度变化。
    便于无测试素材时自匹配验证。
    """
    h, w = ref_img.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for i in range(n_frames):
        # 周期平移: sin 波在 ±15 px 内左右摆
        dx = int(round(15 * math.sin(2 * math.pi * i / n_frames * 2)))
        dy = int(round(10 * math.cos(2 * math.pi * i / n_frames * 3)))
        # 周期亮度: 0.85 ~ 1.15
        bright = 0.85 + 0.30 * (0.5 + 0.5 * math.sin(2 * math.pi * i / n_frames * 1.5))

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(ref_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        frame = np.clip(frame.astype(np.float32) * bright, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return out_path


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=str, default=None,
                        help="输入视频；不传则用 --reference (或 InsightFace sample 't1') 合成 5 秒测试视频")
    parser.add_argument("--reference", type=str, default=None,
                        help="参考脸图片；不传则默认用 video 第 0 帧或 sample 't1'")
    parser.add_argument("--output-dir", type=str, default="output/video_match")
    parser.add_argument("--root", type=str, default=os.path.expanduser("~/.insightface"))
    parser.add_argument("--gpu", type=int, default=0, help="GPU 卡号；-1 纯 CPU")
    parser.add_argument("--model", type=str, default="buffalo_l",
                        help="FaceAnalysis 模型包 (默认 buffalo_l)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="相似度阈值 (cosine), >= 阈值视为 MATCHED  (默认 0.55)")
    parser.add_argument("--uncertain-threshold", type=float, default=0.40,
                        help="低于此值视为 NO_MATCH；介于两者为 UNCERTAIN (默认 0.40)")
    parser.add_argument("--sample-rate", type=int, default=1,
                        help="降频采样: 每 N 帧分析一次 (默认 1 = 全帧)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理多少帧; 0 = 全部")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "run.log")

    THRESH = args.threshold
    UNCERTAIN_LOW = args.uncertain_threshold
    assert 0 < UNCERTAIN_LOW < THRESH < 1.0, "阈值应满足 0 < uncertain < threshold < 1"

    logger.info("=" * 60)
    logger.info("短视频人脸比对 Demo")
    logger.info("=" * 60)
    logger.info(f"输出目录         : {out_dir}")
    logger.info(f"InsightFace root : {args.root}")
    logger.info(f"GPU / ctx_id     : {args.gpu}")
    logger.info(f"模型包           : {args.model}")
    logger.info(f"匹配阈值         : MATCHED >= {THRESH:.2f}")
    logger.info(f"                  : UNCERTAIN [{UNCERTAIN_LOW:.2f}, {THRESH:.2f})")
    logger.info(f"                  : NO_MATCH  < {UNCERTAIN_LOW:.2f}")
    logger.info(f"帧采样率         : 每 {args.sample_rate} 帧分析一次")

    # ---------- 加载 FaceAnalysis ----------
    t0 = time.perf_counter()
    from insightface.app import FaceAnalysis

    if args.gpu >= 0:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = args.gpu
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    app = FaceAnalysis(name=args.model, root=args.root, providers=providers)
    app.prepare(ctx_id=ctx_id)
    t_init = time.perf_counter() - t0
    logger.info(f"FaceAnalysis 初始化耗时 : {t_init*1000:.1f} ms  providers={providers}")

    # ---------- 确定参考脸 ----------
    def _imread_p(path: Path) -> np.ndarray:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    t1 = time.perf_counter()
    synthetic_ref_img = None

    if args.reference:
        ref_path = Path(args.reference).expanduser().resolve()
        ref_img = _imread_p(ref_path)
        source_ref = str(ref_path)
    else:
        from insightface.data import get_image
        ref_img = get_image("t1")
        source_ref = "insightface_sample:t1"
        synthetic_ref_img = ref_img  # 用于后续合成视频

    faces_ref = app.get(ref_img)
    if len(faces_ref) < 1 or faces_ref[0].embedding is None:
        logger.error(f"参考脸未检测到或无 embedding: {source_ref}")
        sys.exit(1)
    ref_emb = faces_ref[0].embedding
    ref_bbox = faces_ref[0].bbox.astype(int).tolist()
    logger.info(f"参考脸来源  : {source_ref}")
    logger.info(f"参考脸 bbox : {ref_bbox}  emb_shape={ref_emb.shape}")
    logger.info(f"参考脸提取耗时 : {(time.perf_counter()-t1)*1000:.1f} ms")

    # ---------- 打开 / 合成视频 ----------
    t2 = time.perf_counter()
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.exists():
            logger.error(f"视频不存在: {video_path}")
            sys.exit(1)
        source_video = str(video_path)
        cap = cv2.VideoCapture(str(video_path))
    else:
        # 合成测试视频
        synthetic_path = out_dir / "_synthetic_test.mp4"
        logger.info(f"未指定视频，用参考脸合成测试视频 -> {synthetic_path}")
        video_path = generate_synthetic_video(
            synthetic_ref_img if synthetic_ref_img is not None else ref_img,
            synthetic_path, n_frames=150, fps=30.0,
        )
        source_video = f"synthetic_from_ref: {source_ref}"
        cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logger.info(f"视频来源    : {source_video}")
    logger.info(f"视频信息    : {src_w}x{src_h} @ {src_fps:.2f} fps  total={src_total}")

    # ---------- 准备输出 ----------
    annotated_path = out_dir / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, src_fps, (src_w, src_h))

    csv_path = out_dir / "frame_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    fieldnames = ["frame_idx", "timestamp_s", "num_faces",
                  "match_status", "similarity", "face_bbox", "det_score", "note"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # ---------- 逐帧处理 ----------
    # 辅助: cosine similarity
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    status_colors = {
        "MATCHED":      (0, 220, 0),     # 绿
        "UNCERTAIN":    (0, 200, 240),   # 橙黄
        "NO_MATCH":     (0, 0, 220),     # 红
        "NO_FACE":      (180, 180, 180), # 灰
        "NO_EMBEDDING": (180, 180, 180),
    }

    frame_idx = -1
    analyzed = 0
    face_count_hist = {0: 0, 1: 0, "2+": 0}
    status_hist = {}
    sim_values = []
    wall_start = time.perf_counter()
    # GPU 显存采样
    max_gpu_mem = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            if frame_idx < 0:
                logger.error("视频打开但读不到任何帧")
                sys.exit(1)
            break
        frame_idx += 1
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        timestamp_s = frame_idx / src_fps

        # 降频采样: 只在每 sample_rate 的倍数帧做检测
        do_analyze = (frame_idx % args.sample_rate == 0)
        record = FrameRecord(frame_idx=frame_idx, timestamp_s=round(timestamp_s, 3),
                             num_faces=0, match_status="NO_FACE", similarity=None,
                             face_bbox=None, det_score=None, note="skipped")

        if do_analyze:
            analyzed += 1
            faces = app.get(frame)
            nf = len(faces)
            record.num_faces = nf
            record.note = ""

            # 统计 GPU 显存 (nvidia-smi, 后台 1 次/秒)
            try:
                import subprocess as sp
                mem_out = sp.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True
                ).strip().splitlines()
                my = int(mem_out[ctx_id]) if ctx_id >= 0 and ctx_id < len(mem_out) else 0
                max_gpu_mem = max(max_gpu_mem, my)
            except Exception:
                pass

            if nf == 0:
                record.match_status = "NO_FACE"
                record.note = "no face detected"
            else:
                # 取所有有 embedding 的脸，算相似度，取最高
                best_sim, best_face = -2.0, None
                have_any_emb = False
                for f in faces:
                    if f.embedding is not None:
                        s = cosine(f.embedding, ref_emb)
                        have_any_emb = True
                        if s > best_sim:
                            best_sim = s
                            best_face = f

                if not have_any_emb:
                    record.match_status = "NO_EMBEDDING"
                    record.note = "no face has embedding"
                elif best_face is None:
                    record.match_status = "NO_EMBEDDING"
                else:
                    record.similarity = round(best_sim, 4)
                    record.face_bbox = best_face.bbox.astype(int).tolist()
                    record.det_score = round(float(best_face.det_score), 4)
                    if best_sim >= THRESH:
                        record.match_status = "MATCHED"
                    elif best_sim >= UNCERTAIN_LOW:
                        record.match_status = "UNCERTAIN"
                    else:
                        record.match_status = "NO_MATCH"

                if nf >= 2:
                    face_count_hist["2+"] = face_count_hist.get("2+", 0) + 1
                    record.note = (record.note + f" multiple_faces={nf}").strip()
                else:
                    face_count_hist[1] = face_count_hist.get(1, 0) + 1
            face_count_hist[0] = face_count_hist.get(0, 0) + (1 if nf == 0 else 0)
        else:
            # 非采样帧: 沿用状态用于可视化 (可选: 也可以不画框)
            # 为保持输出 MP4 全帧有内容，我们仍然画一个灰框状态
            pass

        # CSV 写入
        csv_writer.writerow(asdict(record))
        status_hist[record.match_status] = status_hist.get(record.match_status, 0) + 1
        if record.similarity is not None:
            sim_values.append(record.similarity)

        # 可视化标注 (只对采样帧画，其他帧原样输出)
        if do_analyze and record.face_bbox is not None:
            vis = frame.copy()
            x1, y1, x2, y2 = record.face_bbox
            color = status_colors.get(record.match_status, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = (f"{record.match_status} sim={record.similarity:.3f}"
                     f" det={record.det_score:.3f}")
            cv2.putText(vis, label.encode("ascii", errors="replace").decode(),
                        (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            # 帧号 + 时间戳
            info = f"frame={frame_idx} t={timestamp_s:.2f}s nfaces={record.num_faces}"
            cv2.putText(vis, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            writer.write(vis)
        else:
            # 仍然画个小角落 info
            vis = frame.copy()
            tag = f"frame={frame_idx} t={timestamp_s:.2f}s status={record.match_status}"
            cv2.putText(vis, tag.encode("ascii", errors="replace").decode(),
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            writer.write(vis)

        if (frame_idx + 1) % 30 == 0:
            logger.info(f"  frame {frame_idx:6d} ({timestamp_s:6.2f}s) "
                        f"status={record.match_status:14s} "
                        f"sim={record.similarity if record.similarity is not None else '-':>7} "
                        f"nfaces={record.num_faces}")

    cap.release()
    writer.release()
    csv_file.close()

    wall_total = time.perf_counter() - wall_start
    analyzed_frames = (frame_idx + 1) if frame_idx >= 0 else 0

    # ---------- 汇总 ----------
    summary = {
        "video_source": source_video,
        "reference_source": source_ref,
        "video_info": {
            "width": src_w, "height": src_h, "fps": src_fps,
            "total_frames": src_total, "duration_s": round(src_total / src_fps, 2) if src_fps else None,
        },
        "model": args.model,
        "ctx_id": ctx_id,
        "providers": providers,
        "thresholds": {
            "matched": THRESH, "uncertain_low": UNCERTAIN_LOW,
        },
        "sample_rate": args.sample_rate,
        "total_frames_read": analyzed_frames,
        "frames_analyzed": analyzed,
        "face_count_histogram": face_count_hist,
        "status_histogram": status_hist,
        "similarity_stats": {
            "count": len(sim_values),
            "min": round(min(sim_values), 4) if sim_values else None,
            "max": round(max(sim_values), 4) if sim_values else None,
            "mean": round(float(np.mean(sim_values)), 4) if sim_values else None,
            "median": round(float(np.median(sim_values)), 4) if sim_values else None,
        },
        "timing_s": {
            "faceanalysis_init": round(t_init, 3),
            "ref_embed_extract": round(t1, 3),
            "processing_wall": round(wall_total, 3),
            "per_frame_ms_avg": round(wall_total / analyzed_frames * 1000, 2) if analyzed_frames else None,
            "analyzed_frame_ms_avg": round(wall_total / analyzed * 1000, 2) if analyzed else None,
        },
        "gpu": {
            "ctx_id": ctx_id,
            "peak_memory_MiB": max_gpu_mem,
        },
        "outputs": {
            "annotated_mp4": str(annotated_path),
            "frame_results_csv": str(csv_path),
            "run_log": str(out_dir / "run.log"),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("完成 ✓")
    logger.info(f"总处理帧数   : {analyzed_frames}  (采样 {analyzed} 帧做检测)")
    logger.info(f"总耗时       : {wall_total:.1f} s  (含读帧/写盘)")
    logger.info(f"平均每帧耗时 : {summary['timing_s']['per_frame_ms_avg']} ms / 全帧")
    logger.info(f"             : {summary['timing_s']['analyzed_frame_ms_avg']} ms / 检测帧")
    logger.info(f"状态分布     : {status_hist}")
    logger.info(f"相似度统计   : {summary['similarity_stats']}")
    logger.info(f"峰值 GPU 显存: {max_gpu_mem} MiB  (ctx_id={ctx_id})")
    logger.info("-" * 60)
    logger.info(f"输出文件:")
    logger.info(f"  MP4 : {annotated_path}")
    logger.info(f"  CSV : {csv_path}")
    logger.info(f"  JSON: {summary_path}")
    logger.info(f"  LOG : {out_dir / 'run.log'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
