"""
SAM3 ZMQ 서버 — (rgb, text) 또는 (rgb, box) 받아 mask 반환.

실행 환경: WSL2 native, conda env 'sam3' (Python 3.12 + PyTorch 2.10).
바인드 주소: tcp://0.0.0.0:5556
FP 컨테이너에서 host.docker.internal:5556 으로 접속.

프로토콜:
  요청 (multipart):
    [0] JSON: {
            "w": int, "h": int,
            "text": str (옵션, 예: "bowl"),
            "box": [cx, cy, w, h] (옵션, normalized 0~1)
        }
        text와 box 중 적어도 하나 필요. 둘 다 주면 text 먼저 set 후 box 추가.
    [1] rgb HxWx3 uint8 (RGB 순서)
  응답:
    [0] JSON: {"status": "ok", "score": float, "n_masks": int}
        또는    {"status": "fail", "reason": str}
    [1] mask HxW uint8 (0/255), status == "ok" 일 때만

여러 mask 후보가 있으면 최고 score 선택.
"""

import json
import logging
import os
import sys

import numpy as np
import torch
import zmq
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sam3_server] %(message)s",
    datefmt="%H:%M:%S",
)

# 로컬 SAM3.1 체크포인트 경로
CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT",
    "/mnt/d/ProjectsTracking/sam3/checkpoints/sam3.1_multiplex.pt",
)
BIND_ADDR = os.environ.get("SAM3_BIND_ADDR", "tcp://0.0.0.0:5556")


def build():
    logging.info(f"loading SAM3.1 from {CHECKPOINT} ...")
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT}")
    model = build_sam3_image_model(checkpoint_path=CHECKPOINT, load_from_HF=False)
    processor = Sam3Processor(model)
    logging.info(f"SAM3.1 ready on device={next(model.parameters()).device}")
    return processor


def to_numpy(x):
    """torch tensor or ndarray → ndarray. BFloat16 등 numpy 미지원 dtype은 float32로 캐스팅."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "dtype") and x.dtype == torch.bfloat16:
        x = x.float()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def handle(processor, parts):
    if len(parts) != 2:
        raise ValueError(f"expected 2 frames, got {len(parts)}")
    header = json.loads(parts[0].decode())
    w, h = int(header["w"]), int(header["h"])
    text = header.get("text")
    box = header.get("box")  # [cx, cy, w, h] normalized
    if text is None and box is None:
        raise ValueError("either 'text' or 'box' (or both) required in header")

    rgb = np.frombuffer(parts[1], dtype=np.uint8).reshape(h, w, 3)
    image = Image.fromarray(rgb)  # SAM3 processor는 PIL Image 기대

    # SAM3는 bfloat16 autocast 컨텍스트에서 돌도록 설계됨 (examples 노트북과 동일)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(image)
        if text is not None:
            output = processor.set_text_prompt(state=inference_state, prompt=text)
        if box is not None:
            if len(box) != 4:
                raise ValueError(f"box must be [cx, cy, w, h], got {box}")
            # add_geometric_prompt: text 없으면 내부에서 dummy "visual" prompt 설정.
            # label=True (positive box).
            output = processor.add_geometric_prompt(
                box=[float(x) for x in box], label=True, state=inference_state
            )

    prompt_desc = []
    if text is not None: prompt_desc.append(f"text='{text}'")
    if box is not None: prompt_desc.append(f"box={box}")
    prompt_str = ", ".join(prompt_desc)

    masks = to_numpy(output["masks"])     # (N, H, W) — bool 또는 float
    scores = to_numpy(output["scores"])   # (N,)
    n_masks = int(len(masks))
    logging.info(f"prompt={prompt_str}: n_masks={n_masks}, scores={scores.tolist() if n_masks>0 else []}")
    if n_masks == 0:
        raise RuntimeError(f"no masks found for prompt {prompt_str}")

    # [DIAG] 후보별 형태 통계. 박스 프롬프트가 '객체' 대신 '윤곽 링'을 돌려주는 사례가 있어
    # (2026-07-22), 어느 후보가 실제 객체인지 로그로 남긴다.
    #   채움률(=mask px / bbox 넓이) 이 낮으면 링/얼룩, 높으면 꽉 찬 객체.
    for i in range(n_masks):
        mi = masks[i]
        if mi.ndim == 3 and mi.shape[0] == 1:
            mi = mi[0]
        mb = mi if mi.dtype == np.bool_ else (mi > 0.5)
        npx = int(mb.sum())
        if npx == 0:
            logging.info(f"  cand[{i}] score={float(scores[i]):.4f} px=0")
            continue
        ys, xs = np.nonzero(mb)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        logging.info(
            f"  cand[{i}] score={float(scores[i]):.4f} px={npx} "
            f"bbox={bw}x{bh}@({x0},{y0}) fill={npx / (bw * bh) * 100:.1f}%")

    # 최고 score 선택. SAM3_PICK 로 오버라이드 가능(진단용): "score"(기본) | "fill" | "0".."N"
    pick = os.environ.get("SAM3_PICK", "score")
    if pick == "score":
        best = int(np.argmax(scores))
    elif pick == "fill":
        # 채움률이 가장 높은 후보 = 링이 아니라 꽉 찬 덩어리
        fills = []
        for i in range(n_masks):
            mi = masks[i]
            if mi.ndim == 3 and mi.shape[0] == 1:
                mi = mi[0]
            mb = mi if mi.dtype == np.bool_ else (mi > 0.5)
            if mb.sum() == 0:
                fills.append(0.0); continue
            ys, xs = np.nonzero(mb)
            bw = int(xs.max() - xs.min() + 1); bh = int(ys.max() - ys.min() + 1)
            fills.append(float(mb.sum()) / (bw * bh))
        best = int(np.argmax(fills))
    else:
        best = max(0, min(n_masks - 1, int(pick)))
    logging.info(f"  -> picked cand[{best}] (SAM3_PICK={pick})")
    mask = masks[best]
    # SAM3 출력이 (N, 1, H, W) 형태일 때 singleton 차원 제거
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.dtype != np.bool_:
        mask = mask > 0.5

    if mask.shape != (h, w):
        raise RuntimeError(f"mask shape mismatch: got {mask.shape}, expected {(h, w)}")

    mask_uint8 = mask.astype(np.uint8) * 255

    return mask_uint8, float(scores[best]), n_masks


def main():
    processor = build()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(BIND_ADDR)
    logging.info(f"listening on {BIND_ADDR}")

    while True:
        parts = sock.recv_multipart()
        try:
            mask, score, n_masks = handle(processor, parts)
            sock.send_multipart([
                json.dumps({"status": "ok", "score": score, "n_masks": n_masks}).encode(),
                mask.tobytes(),
            ])
            logging.info(
                f"sent mask: px={int((mask > 0).sum())} score={score:.3f} n={n_masks}"
            )
        except Exception as e:
            logging.exception("handle failed")
            try:
                sock.send_multipart([
                    json.dumps({"status": "fail", "reason": str(e)}).encode(),
                ])
            except Exception:
                pass


if __name__ == "__main__":
    main()
