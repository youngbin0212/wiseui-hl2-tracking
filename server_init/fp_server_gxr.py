"""
Galaxy XR용 FoundationPose 서버.

API:
    POST /register
        body: {"rgb": <b64 png>, "depth": <b64 png 16bit mm>, "mask": <b64 png 8bit>,
               "K": [9 floats], "mesh_id": "Bowl_White" (optional, 현재 무시)}
        response: {"pose": [[16 floats 4x4]], "server_proc_ms": ..., "session_id": "..."}

    POST /track
        body: {"rgb": <b64 png>, "depth": <b64 png 16bit mm>, "K": [9 floats]}
        response: {"pose": [[16 floats]], "server_proc_ms": ...}

    POST /estimate    (편의: init=true면 register, 아니면 track)
        body: register/track body + "init": bool
        response: register/track response

    POST /register_with_box
        body: {"rgb": <b64 png>, "depth_raw": <b64 raw depth16>,
               "depth_raw_width": int, "depth_raw_height": int,
               "K": [9 floats], "box": [cx, cy, w, h] (normalized 0~1)}
        내부에서 sam3_server에 (rgb, box)를 ZMQ로 보내 mask 받고,
        그 mask로 register() 호출.
        response: {"pose": [[..]], "session_id": "...", "mask_png_b64": "...",
                   "sam_proc_ms": ..., "server_proc_ms": ...}

    GET /health
        response: {"ok": true, "initialized": bool}

CLI:
    python fp_server_gxr.py                    # 서버 실행 (default mesh: Bowl_White)
    python fp_server_gxr.py --mesh PATH        # 다른 mesh 사용
    python fp_server_gxr.py --port 8000        # 포트 변경
    python fp_server_gxr.py --self-test        # HTTP 없이 mustard0로 1프레임 register 검증

기기 측 사용 흐름:
    1. /health 호출 → 서버 살아있는지 확인
    2. /register 호출 (init mask 포함) → 첫 프레임 pose 받기
    3. /track 반복 호출 → 이후 프레임 pose 받기
    4. tracking 실패 의심되면 /register로 다시 초기화

주의:
    - depth는 16-bit PNG (mm 단위) 가정. 변환: depth_m = depth_raw_uint16 / 1000.0
    - mask는 8-bit single-channel PNG. >0이면 객체, 0이면 배경
    - K는 row-major 9개 float ([fx,0,cx, 0,fy,cy, 0,0,1])
"""

# Python 3.9 conda env에서도 `X | None` 같은 PEP 604 syntax를 쓰기 위해.
# 모든 annotation을 lazy string으로 평가시킴.
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

# ZMQ — sam3_server (host) 호출용. pyzmq가 FP 컨테이너에 설치돼 있어야 함.
#   pip install pyzmq
import zmq

# FoundationPose deps — 이 파일을 FoundationPose 루트에서 실행해야 import 됨.
from estimater import (
    FoundationPose, PoseRefinePredictor, ScorePredictor,
    dr, trimesh, set_logging_format, set_seed,
)
from datareader import YcbineoatReader

# pose 시각화용. nvdiffrast로 실제 mesh를 렌더해서 RGB 위에 합성.
import cv2
import torch
from Utils import draw_posed_3d_box, draw_xyz_axis, nvdiffrast_render


# ─────────────────── Global state ───────────────────
_est: FoundationPose | None = None
_mesh = None
_lock = threading.Lock()
_session_id: str | None = None
_default_mesh_path = None
# pose overlay용 — mesh의 oriented bbox + 원점 보정(to_origin). init 때 계산.
_bbox = None
_to_origin = None

# SAM3 ZMQ connection (lazy). FP 컨테이너 내부에서 host의 sam3_server에 접속.
# 기본 주소는 docker host gateway. SAM3_SERVER_ADDR 환경변수로 오버라이드 가능.
_zmq_ctx: zmq.Context | None = None
_zmq_sock: zmq.Socket | None = None
_zmq_addr: str = os.environ.get("SAM3_SERVER_ADDR", "tcp://host.docker.internal:5556")


# ─────────────────── Init / inference ───────────────────
def init_estimator(mesh_file: str) -> None:
    """FoundationPose estimator를 메모리에 올림. 시작 시 한 번."""
    global _est, _mesh, _bbox, _to_origin
    logging.info(f"loading mesh: {mesh_file}")
    _mesh = trimesh.load(mesh_file)
    if _mesh.vertices is None or len(_mesh.vertices) == 0:
        raise RuntimeError(f"mesh has no vertices: {mesh_file}")

    # ★ trimesh 가 map_Kd 텍스처를 2x2 더미로 로드하는 버그(2026-07-23 확인). 그러면 FP render-compare
    #   가 균일색으로 렌더 → 납작한 책의 회전 판별 불가 → score 평평(82.5~80.5), 회전 틀림.
    #   실제 PNG 를 직접 붙여 고친다. FoundationPose() 생성 전에 해야 mesh_tensors 에 반영됨.
    try:
        import trimesh.visual.texture as _TV
        from PIL import Image as _PImg
        img = getattr(getattr(_mesh.visual, "material", None), "image", None)
        tiny = (img is None) or (max(img.size) <= 4)
        if isinstance(_mesh.visual, _TV.TextureVisuals) and tiny:
            texp = None
            mtl = os.path.splitext(mesh_file)[0] + ".mtl"
            if os.path.exists(mtl):
                for line in open(mtl, "r", errors="ignore"):
                    if line.strip().lower().startswith("map_kd"):
                        texp = os.path.join(os.path.dirname(mesh_file), line.split(None, 1)[1].strip())
                        break
            if texp is None or not os.path.exists(texp):
                cand = os.path.splitext(mesh_file)[0] + ".png"   # obj 이름.png 폴백
                texp = cand if os.path.exists(cand) else None
            if texp:
                _mesh.visual.material.image = _PImg.open(texp).convert("RGB")
                logging.info(f"★ texture fixed: {texp} -> {_mesh.visual.material.image.size}")
            else:
                logging.warning("★ texture 2x2 dummy 인데 실제 PNG 를 못 찾음 → 회전 판별 불가 위험")
        else:
            logging.info(f"texture OK: {None if img is None else img.size}")

        # A/B: 텍스처를 강제로 끄고 회색으로 (FP_NO_TEX=1). scorer 가 텍스처를 신호로 쓰는지
        #   노이즈로 쓰는지 판별. 회색이 상위 25개를 더 벌리면 텍스처가 appearance 불일치로 해가 됨.
        if os.environ.get("FP_NO_TEX", "0") == "1":
            import trimesh.visual as _TVis
            _mesh.visual = _TVis.ColorVisuals(
                mesh=_mesh,
                vertex_colors=np.tile(np.array([128, 128, 128], np.uint8), (len(_mesh.vertices), 1)))
            logging.info("★ FP_NO_TEX=1 → 텍스처 끄고 회색. (A/B 대조군)")
    except Exception as e:
        logging.warning(f"texture fix skipped: {e}")

    # pose overlay용 oriented bounding box (run_demo.py와 동일 방식).
    _to_origin, extents = trimesh.bounds.oriented_bounds(_mesh)
    _bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    logging.info(f"mesh bbox extents: {extents}")

    # amp(fp16) 옵션. scorer forward 가 fp16 autocast 로 돌면 score_logit 이 fp16 정밀도로
    #   양자화돼(54 근처 간격 0.03125) 252개 회전 후보 순위가 뭉개진다(HL2 납작 책에서 관측).
    #   ⚠️ 기본은 on(원래 동작) — 이 서버는 Galaxy XR(gxr)도 쓰므로 기본을 바꾸면 Galaxy 에 영향.
    #   HL2 세션은 FP_AMP=0 으로 명시적으로 꺼서 fp32 로 돌린다.
    _amp = os.environ.get("FP_AMP", "1") == "1"
    logging.info(f"initializing FoundationPose (scorer amp={_amp} + refiner + glctx)…")
    scorer = ScorePredictor(amp=_amp)
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # debug_dir은 debug=0이어도 FoundationPose 내부에서 makedirs를 호출하므로 None이면 죽음.
    # 실제 경로 필요. /tmp 아래에 만들어둠.
    debug_dir = "/tmp/fp_server_gxr"
    os.makedirs(debug_dir, exist_ok=True)
    _est = FoundationPose(
        model_pts=_mesh.vertices,
        model_normals=_mesh.vertex_normals,
        mesh=_mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=0,
        glctx=glctx,
    )
    logging.info("FoundationPose ready")


def _decode_image(b64_str: str) -> np.ndarray:
    """Base64 PNG → numpy array. raw decode만, 채널 처리는 별도 함수에서."""
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw))
    return np.array(img)


def _decode_rgb(b64_str: str) -> np.ndarray:
    """RGB 3-channel uint8로 정규화. RGBA → RGB, 그레이스케일 → 3채널 복제."""
    arr = _decode_image(b64_str)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def _decode_depth_meters(b64_str: str) -> np.ndarray:
    """16-bit mm PNG → float32 meters."""
    arr = _decode_image(b64_str)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32) / 1000.0


def _decode_depth_raw16_meters(b64_str: str, width: int, height: int) -> np.ndarray:
    """
    Android Camera2 ImageFormat.DEPTH16 raw bytes → float32 meters.
    DEPTH16 packing: little-endian 16-bit per pixel.
      bits 0-12: depth in mm
      bits 13-15: confidence (0=invalid, 7=best)
    """
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    depth_mm = (arr & 0x1FFF).astype(np.float32)
    # confidence가 0이면 depth invalid → 0으로 두면 FP가 zfar로 처리
    return depth_mm / 1000.0


def _resize_depth_to(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """nearest-neighbor로 depth resize. RGB 해상도에 맞추기 위함."""
    if depth.shape == (height, width):
        return depth
    img = Image.fromarray(depth, mode="F")
    img = img.resize((width, height), Image.NEAREST)
    return np.array(img, dtype=np.float32)


def _decode_mask_bool(b64_str: str) -> np.ndarray:
    """단일 채널 mask → bool. >0이면 객체."""
    arr = _decode_image(b64_str)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def _decode_K(k_list) -> np.ndarray:
    """row-major 9개 float → 3x3 numpy."""
    return np.array(k_list, dtype=np.float64).reshape(3, 3)


def _render_overlay_b64(rgb: np.ndarray, pose: np.ndarray, K: np.ndarray) -> str:
    """pose에 맞춰 실제 mesh를 nvdiffrast로 렌더해서 RGB 위에 반투명 합성, PNG b64 반환.
    텍스처 mesh가 그대로 올라가므로 추적이 맞는지 가장 확실하게 보임."""
    H, W = rgb.shape[:2]
    ob_in_cams = torch.as_tensor(pose.reshape(1, 4, 4), device="cuda", dtype=torch.float)
    color, depth, _ = nvdiffrast_render(
        K=K, H=H, W=W, ob_in_cams=ob_in_cams, glctx=_est.glctx,
        mesh_tensors=_est.mesh_tensors, mesh=_est.mesh,
        use_light=True, w_ambient=0.9, w_diffuse=0.4,
    )
    rendered = (color[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mask = depth[0].detach().cpu().numpy() > 0   # mesh가 그려진 픽셀
    blend = 0.65   # 0=원본만, 1=렌더만. 반투명으로 원본 그릇과 정렬 확인.
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * (1.0 - blend) + rendered[mask].astype(np.float32) * blend
    out = out.clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _diag_pose_depth(tag, pose, depth, mask, K):
    """등록 pose 의 물체 거리(tz)와 마스크 영역 depth 를 대조. 밀림/크기 원인 확정용.
      - mesh 가 크게 렌더되고 밀리면(2026-07-23 관측 1.20x) pose tz 가 실제보다 작다는 뜻.
      - tz 와 median(depth in mask) 이 크게 다르면 depth 정합/값 문제, 비슷하면 FP refine 문제.
    """
    K = np.asarray(K).reshape(3, 3)
    tx, ty, tz = float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])
    dv = depth[mask]; dv = dv[dv > 0]
    med = float(np.median(dv)) if dv.size else 0.0
    p10, p50, p90 = (np.percentile(dv, [10, 50, 90]).tolist() if dv.size else [0, 0, 0])
    u = v = -1.0
    if tz > 1e-6:
        u = K[0, 0] * tx / tz + K[0, 2]; v = K[1, 1] * ty / tz + K[1, 2]
    logging.info(
        f"[{tag}] pose t=({tx:.3f},{ty:.3f},{tz:.3f})m  "
        f"depth in mask valid={dv.size}/{int(mask.sum())} "
        f"({100*dv.size/max(int(mask.sum()),1):.0f}%) median={med:.3f}m p10/50/90={p10:.3f}/{p50:.3f}/{p90:.3f}  "
        f"obj_origin_proj=({u:.0f},{v:.0f})  "
        f"tz/median={tz/med if med>0 else 0:.2f} (1 이면 일치; <1 이면 pose 가 depth 보다 가까움)")
    try:
        np.save("debug_depth.npy", depth.astype(np.float32))
    except Exception:
        pass


def _decode_depth_any(payload: dict) -> np.ndarray:
    """payload에 depth_raw가 있으면 raw16, 아니면 기존 PNG 디코더 사용."""
    if "depth_raw" in payload:
        return _decode_depth_raw16_meters(
            payload["depth_raw"],
            int(payload["depth_raw_width"]),
            int(payload["depth_raw_height"]),
        )
    return _decode_depth_meters(payload["depth"])


def _ensure_zmq_sock() -> zmq.Socket:
    """SAM3 서버에 연결된 REQ 소켓 보장. lazy init, 첫 호출 시 connect."""
    global _zmq_ctx, _zmq_sock
    if _zmq_sock is None:
        _zmq_ctx = zmq.Context.instance()
        _zmq_sock = _zmq_ctx.socket(zmq.REQ)
        # SAM3 inference + 대형 mask 전송 → 충분히 길게.
        _zmq_sock.setsockopt(zmq.RCVTIMEO, 30000)   # 30s
        _zmq_sock.setsockopt(zmq.SNDTIMEO, 5000)
        _zmq_sock.connect(_zmq_addr)
        logging.info(f"connected to SAM3 server at {_zmq_addr}")
    return _zmq_sock


def _call_sam3_mask(rgb: np.ndarray, box_norm, text=None) -> np.ndarray:
    """RGB(HxWx3 uint8) + box (normalized cxcywh) [+ text] → SAM3에 ZMQ로 요청, bool mask 반환.
    text(옵션): semantic 프롬프트. 주면 SAM3가 'box 안의 그 객체'를 더 정확히 segment."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got shape={rgb.shape}")
    h, w = rgb.shape[:2]
    if len(box_norm) != 4:
        raise ValueError(f"box must be [cx,cy,w,h], got {box_norm}")
    sock = _ensure_zmq_sock()
    header_dict = {"w": int(w), "h": int(h), "box": [float(x) for x in box_norm]}
    if text:
        header_dict["text"] = str(text)
    header = json.dumps(header_dict).encode()
    rgb_bytes = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    sock.send_multipart([header, rgb_bytes])
    reply = sock.recv_multipart()
    resp = json.loads(reply[0].decode())
    if resp.get("status") != "ok":
        raise RuntimeError(f"SAM3 returned fail: {resp.get('reason', 'unknown')}")
    if len(reply) < 2:
        raise RuntimeError("SAM3 ok response missing mask payload")
    mask = np.frombuffer(reply[1], dtype=np.uint8).reshape(h, w)
    score = float(resp.get("score", 0.0))
    logging.info(f"SAM3 mask received: px={int((mask > 0).sum())} score={score:.3f}")
    return mask > 0


def do_register_with_box(payload: dict) -> dict:
    """RGB+depth+K+box를 받아 sam3_server로 mask 만들고 FP register 실행."""
    global _session_id
    rgb = _decode_rgb(payload["rgb"])
    depth = _decode_depth_any(payload)
    depth = _resize_depth_to(depth, rgb.shape[0], rgb.shape[1])
    K = _decode_K(payload["K"])
    if "box" not in payload:
        raise ValueError("'box' field required for /register_with_box")
    box = payload["box"]
    text = payload.get("text")   # 옵션: semantic 프롬프트 (예: "glue bottle")
    iteration = int(payload.get("iteration", 5))

    t_sam_start = time.perf_counter()
    mask = _call_sam3_mask(rgb, box, text=text)
    sam_ms = (time.perf_counter() - t_sam_start) * 1000.0

    with _lock:
        if _est is None:
            raise RuntimeError("estimator not initialized")
        pose = _est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iteration)
        _session_id = uuid.uuid4().hex[:8]

    # 디버그: register 입력/결과를 PNG로 저장 (pose가 객체에 맞는지 눈으로 확인용).
    try:
        import cv2
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug_rgb.png", rgb_bgr)   # ★ FP 가 받은 raw 프레임 — offset 은 이것 기준으로만
        mvis = rgb_bgr.copy(); mvis[mask] = (0, 0, 255)   # ① mask 오버레이
        cv2.imwrite("debug_sam_mask.png", mvis)
        vis_b64 = _render_overlay_b64(rgb, pose, K)         # ② FP pose 오버레이
        with open("debug_fp_pose.png", "wb") as f:
            f.write(base64.b64decode(vis_b64))
        _diag_pose_depth("box+text", pose, depth, mask, K)
    except Exception as e:
        logging.warning(f"debug img save failed: {e}")

    # mask를 device 시각화용으로 PNG b64로도 반환.
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result = {
        "pose": pose.reshape(4, 4).tolist(),
        "session_id": _session_id,
        "mask_png_b64": mask_b64,
        "sam_proc_ms": sam_ms,
    }
    if payload.get("overlay", False):
        result["vis_png_b64"] = _render_overlay_b64(rgb, pose, K)
    return result


def do_register(payload: dict) -> dict:
    """첫 프레임 — heavy pose estimation. mask 필수."""
    global _session_id
    rgb = _decode_rgb(payload["rgb"])
    depth = _decode_depth_any(payload)
    # depth 해상도가 RGB와 다르면 맞춰줌 (Galaxy XR은 RGB 640x480, depth 240x320)
    depth = _resize_depth_to(depth, rgb.shape[0], rgb.shape[1])
    K = _decode_K(payload["K"])
    mask = _decode_mask_bool(payload["mask"])
    iteration = int(payload.get("iteration", 5))

    with _lock:
        if _est is None:
            raise RuntimeError("estimator not initialized")
        pose = _est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iteration)
        _session_id = uuid.uuid4().hex[:8]

    # 디버그 저장 — do_register_with_box 와 동일하게 text-only(/register) 경로도 그림을 남긴다.
    #   debug_sam_mask.png : 입력 mask 오버레이
    #   debug_fp_pose.png  : FP pose 로 mesh 렌더 (책에 얹히는지 = 등록 정합 판정)
    #   debug_depth.png    : 등록에 쓴 depth (책 영역에 유효값이 있는지 = 밀림 원인 확인용)
    try:
        import cv2
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug_rgb.png", rgb_bgr)   # ★ FP 가 실제로 받은 raw 프레임 — offset 은 이것 기준으로만 재라
        mvis = rgb_bgr.copy(); mvis[mask] = (0, 0, 255)
        cv2.imwrite("debug_sam_mask.png", mvis)
        with open("debug_fp_pose.png", "wb") as f:
            f.write(base64.b64decode(_render_overlay_b64(rgb, pose, K)))
        d = depth.astype(np.float32)
        dn = (d / max(d.max(), 1e-6) * 255).astype(np.uint8)
        dvis = cv2.applyColorMap(dn, cv2.COLORMAP_JET); dvis[d <= 0] = (0, 0, 0)
        dvis[mask] = (dvis[mask] * 0.6 + np.array([255, 255, 255]) * 0.4).astype(np.uint8)
        cv2.imwrite("debug_depth.png", dvis)
        _diag_pose_depth("text-only", pose, depth, mask, K)
    except Exception as e:
        logging.warning(f"debug img save failed: {e}")

    result = {
        "pose": pose.reshape(4, 4).tolist(),
        "session_id": _session_id,
    }
    if payload.get("overlay", False):
        result["vis_png_b64"] = _render_overlay_b64(rgb, pose, K)
    return result


def do_track(payload: dict) -> dict:
    """이후 프레임 — light tracking. register 먼저 필요."""
    if _session_id is None:
        raise RuntimeError("not registered yet — call /register first")

    rgb = _decode_rgb(payload["rgb"])
    depth = _decode_depth_any(payload)
    depth = _resize_depth_to(depth, rgb.shape[0], rgb.shape[1])
    K = _decode_K(payload["K"])
    iteration = int(payload.get("iteration", 2))

    with _lock:
        pose = _est.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)

    result = {"pose": pose.reshape(4, 4).tolist()}
    if payload.get("overlay", False):
        result["vis_png_b64"] = _render_overlay_b64(rgb, pose, K)
    return result


# ─────────────────── Self-test (no HTTP) ───────────────────
def self_test(test_scene_dir: str, mesh_file: str) -> None:
    """mustard0 형식 데이터로 FP 동작 검증. HTTP 없이 1프레임만 register."""
    set_logging_format()
    set_seed(0)
    init_estimator(mesh_file)

    logging.info(f"self-test on: {test_scene_dir}")
    reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)
    color = reader.get_color(0)
    depth = reader.get_depth(0)
    mask = reader.get_mask(0).astype(bool)

    t0 = time.perf_counter()
    pose = _est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
    t1 = time.perf_counter()
    logging.info(f"register OK in {(t1 - t0) * 1000:.1f} ms")
    print("\n=== Pose (object in camera frame) ===")
    print(pose.reshape(4, 4))

    # 두 번째 프레임 track도 검증
    if len(reader.color_files) > 1:
        color2 = reader.get_color(1)
        depth2 = reader.get_depth(1)
        t0 = time.perf_counter()
        pose2 = _est.track_one(rgb=color2, depth=depth2, K=reader.K, iteration=2)
        t1 = time.perf_counter()
        logging.info(f"track_one OK in {(t1 - t0) * 1000:.1f} ms")
        print("\n=== Track pose (frame 1) ===")
        print(pose2.reshape(4, 4))


# ─────────────────── HTTP handler ───────────────────
class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        endpoint = self.path
        if endpoint not in ("/register", "/track", "/estimate", "/register_with_box"):
            self._respond(404, {"error": "not found"})
            return

        t_start = time.perf_counter()
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            self._respond(400, {"error": f"bad json: {e}"})
            return

        try:
            if endpoint == "/register":
                result = do_register(payload)
            elif endpoint == "/track":
                result = do_track(payload)
            elif endpoint == "/register_with_box":
                result = do_register_with_box(payload)
            else:  # /estimate
                if payload.get("init") or _session_id is None:
                    result = do_register(payload)
                else:
                    result = do_track(payload)
        except Exception as e:
            logging.exception("estimate failed")
            self._respond(500, {"error": f"{type(e).__name__}: {e}"})
            return

        t_end = time.perf_counter()
        result["server_proc_ms"] = (t_end - t_start) * 1000.0
        self._respond(200, result)

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self._respond(200, {
                "ok": True,
                "initialized": _est is not None,
                "registered": _session_id is not None,
                "endpoints": [
                    "POST /register", "POST /track", "POST /estimate",
                    "POST /register_with_box", "GET /health",
                ],
                "sam3_addr": _zmq_addr,
            })
            return
        self._respond(404, {"error": "not found"})

    def _respond(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        # 기본 액세스 로그 끔 — 매 요청마다 시끄러움.
        pass


# ─────────────────── Main ───────────────────
def main() -> None:
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh",
        default=f"{code_dir}/my_data/Bowl_White/model.obj",
        help="object mesh .obj path",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run FP locally on demo_data/mustard0 to verify install (no HTTP)",
    )
    parser.add_argument(
        "--self-test-scene",
        default=f"{code_dir}/demo_data/mustard0",
        help="scene dir for self-test (must have rgb/, depth/, masks/, cam_K.txt)",
    )
    parser.add_argument(
        "--self-test-mesh",
        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj",
        help="mesh for self-test (default: mustard mesh that pairs with mustard0 scene)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.self_test:
        self_test(args.self_test_scene, args.self_test_mesh)
        return

    set_logging_format()
    set_seed(0)
    init_estimator(args.mesh)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nfp_server_gxr listening on http://{args.host}:{args.port}")
    print(f"  mesh: {args.mesh}")
    print(f"  endpoints: POST /register | /track | /estimate | /register_with_box | GET /health")
    print(f"  SAM3 ZMQ: {_zmq_addr} (lazy connect)")
    print("Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
