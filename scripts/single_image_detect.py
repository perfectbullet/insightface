#!/usr/bin/env python3
"""
单图检测与特征提取 Demo
========================
使用 InsightFace 的 FaceAnalysis 对单张图片进行:
  1. 人脸检测 (detection: SCRFD-10GF, buffalo_l)
  2. 人脸对齐 (landmark: 2d106)
  3. 身份特征提取 (recognition: ResNet50@WebFace600K, 512-dim embedding)
  4. 可选: 性别年龄属性

输出:
  - annotated.jpg         标注框 + 置信度 + 关键点
  - embedding.npy         第一个脸的 512 维特征向量
  - result.json           结构化检测结果
  - run.log               运行日志 + 耗时

用法:
  python scripts/single_image_detect.py \
    --image /path/to/image.jpg \
    --output-dir /path/to/output \
    --root ~/.insightface \
    --gpu 0

若不传 --image，默认用 InsightFace 内置 sample 't1'。
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("single_image")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # file + console
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=str, default=None,
                        help="输入图片路径；默认用 InsightFace 内置 sample 't1'")
    parser.add_argument("--output-dir", type=str, default="output/single_image",
                        help="输出目录 (默认 output/single_image)")
    parser.add_argument("--root", type=str, default=os.path.expanduser("~/.insightface"),
                        help="InsightFace 模型根目录")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU 卡号；-1 表示纯 CPU (默认 0)")
    parser.add_argument("--model", type=str, default="buffalo_l",
                        help="FaceAnalysis 模型包名 (默认 buffalo_l)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "run.log")

    logger.info("=" * 60)
    logger.info("单图检测与特征提取 Demo")
    logger.info("=" * 60)
    logger.info(f"输出目录    : {out_dir}")
    logger.info(f"InsightFace root : {args.root}")
    logger.info(f"GPU / ctx_id : {args.gpu}")
    logger.info(f"模型包       : {args.model}")

    # ---- 加载图片 ----
    t0 = time.perf_counter()
    if args.image is None:
        from insightface.data import get_image
        logger.info("使用 InsightFace 内置 sample image 't1'")
        img = get_image("t1")
        image_source = "insightface_sample:t1"
    else:
        img_path = Path(args.image).expanduser().resolve()
        if not img_path.exists():
            logger.error(f"图片不存在: {img_path}")
            sys.exit(1)
        # 支持中文路径
        data = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"无法解码图片: {img_path}")
            sys.exit(1)
        image_source = str(img_path)

    h, w = img.shape[:2]
    t_load = time.perf_counter() - t0
    logger.info(f"图片来源    : {image_source}")
    logger.info(f"图片尺寸    : {w}x{h}  shape={img.shape}  dtype={img.dtype}")
    logger.info(f"图片加载耗时 : {t_load*1000:.1f} ms")

    # ---- 初始化 FaceAnalysis ----
    t1 = time.perf_counter()
    from insightface.app import FaceAnalysis

    if args.gpu >= 0:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = args.gpu
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    logger.info(f"providers   : {providers}")
    app = FaceAnalysis(name=args.model, root=args.root, providers=providers)
    app.prepare(ctx_id=ctx_id)
    t_init = time.perf_counter() - t1
    logger.info(f"FaceAnalysis 初始化耗时 : {t_init*1000:.1f} ms")

    # ---- 检测 + 特征提取 ----
    t2 = time.perf_counter()
    faces = app.get(img)
    t_infer = time.perf_counter() - t2
    logger.info(f"检测到人脸数 : {len(faces)}")
    logger.info(f"推理耗时    : {t_infer*1000:.1f} ms")

    # ---- 汇总结果 ----
    results = []
    for i, face in enumerate(faces):
        item = {
            "face_index": i,
            "bbox": face.bbox.astype(int).tolist(),
            "det_score": round(float(face.det_score), 4),
            "kps": face.kps.astype(int).tolist() if face.kps is not None else None,
            "embedding_shape": list(face.embedding.shape) if face.embedding is not None else None,
            "has_embedding": face.embedding is not None,
        }
        # 属性 (如果模型包带 gender_age)
        if getattr(face, "gender", None) is not None:
            item["gender"] = int(face.gender)
        if getattr(face, "age", None) is not None:
            item["age"] = int(face.age)
        results.append(item)
        logger.info(f"  face[{i}]: bbox={item['bbox']} det_score={item['det_score']} "
                    f"embedding={'OK '+str(item['embedding_shape']) if item['has_embedding'] else 'N/A'} "
                    f"gender={item.get('gender','-')} age={item.get('age','-')}")

    # ---- 保存结果 ----
    # 1) 标注图
    annotated = app.draw_on(img, faces)
    anno_path = out_dir / "annotated.jpg"
    cv2.imwrite(str(anno_path), annotated)
    logger.info(f"标注图片已保存 : {anno_path}")

    # 2) 第一个脸的 embedding
    if len(faces) >= 1 and faces[0].embedding is not None:
        emb_path = out_dir / "embedding_face0.npy"
        np.save(str(emb_path), faces[0].embedding)
        logger.info(f"特征向量已保存 : {emb_path}  shape={faces[0].embedding.shape}")

    # 3) JSON 结果
    summary = {
        "image_source": image_source,
        "image_size": [w, h],
        "model": args.model,
        "ctx_id": ctx_id,
        "providers": providers,
        "num_faces_detected": len(faces),
        "timing_ms": {
            "image_load": round(t_load * 1000, 1),
            "faceanalysis_init": round(t_init * 1000, 1),
            "inference": round(t_infer * 1000, 1),
            "total": round((t_load + t_init + t_infer) * 1000, 1),
        },
        "faces": results,
    }
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"结构化结果已保存 : {json_path}")

    logger.info("=" * 60)
    logger.info("完成 ✓")
    logger.info(f"总耗时 : {summary['timing_ms']['total']:.1f} ms")
    logger.info(f"输出目录 : {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
