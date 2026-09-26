"""
트랙B-A 시퀀싱: wiseui FP 초기 pose 엔드포인트 (PC). 트랙A 그대로 재사용.

wiseui(기기)가 GET /init 하면:
  1. wiseui hl2ss(192.168.0.16)에서 PV+depth 한 프레임 grab (hl2_capture)
     (이때 wiseui 는 hl2ss PV+depth ON, PhotoCapture OFF 상태)
  2. SAM3(text=book) → mask
  3. fp_server_gxr(8000) /register → ob_in_cam pose
  4. {"ok":true,"pose":[16 row-major]} 반환 → wiseui 가 hl2ss 끄고 PhotoCapture 로 추적
재투영/좌표변환 없음 — FP pose 가 hl2ss PV 카메라 기준 ob_in_cam, srt3d 도 같은 PV 카메라.

사전: sam3_server(WSL:5556) + fp_server_gxr(Docker:8000, joke_book) + 이 스크립트(my_base).
실행:  conda activate my_base; cd D:\\ProjectsTracking\\hl2_pipeline; python init_server.py
"""
import io
import os
import json
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2
import zmq
import requests
from PIL import Image

from hl2_capture import HL2Capture

HL2_HOST = os.environ.get("HL2_HOST", "192.168.0.16")
FP_URL = os.environ.get("FP_URL", "http://localhost:8000")
SAM_ADDR = os.environ.get("SAM3_ADDR", "tcp://localhost:5556")
OBJ_TEXT = os.environ.get("OBJ_TEXT", "book")
PORT = int(os.environ.get("INIT_PORT", "8002"))
# FP register refine 반복 수. 재빌드 없이 튜닝: 초기 자세 밀림 진단용(refiner vs 미수렴 판별).
#   INIT_ITER=1  → depth 초기값에 가까움 / INIT_ITER=15 → 더 수렴. 기본 5.
INIT_ITER = int(os.environ.get("INIT_ITER", "5"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def encode_png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode("ascii")


def sam3_mask(rgb, text):
    h, w = rgb.shape[:2]
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, 30000); s.setsockopt(zmq.SNDTIMEO, 5000); s.connect(SAM_ADDR)
    try:
        s.send_multipart([json.dumps({"w": int(w), "h": int(h), "text": text}).encode(),
                          np.ascontiguousarray(rgb, np.uint8).tobytes()])
        rep = s.recv_multipart()
    finally:
        s.close(0)
    resp = json.loads(rep[0].decode())
    if resp.get("status") != "ok" or len(rep) < 2:
        raise RuntimeError(f"SAM3 fail: {resp.get('reason')}")
    return (np.frombuffer(rep[1], np.uint8).reshape(h, w) > 0)


def _grab_frame():
    """hl2ss 에서 PV+depth 한 프레임 → (rgb, depth_mm, K_list)."""
    cap = HL2Capture(HL2_HOST); cap.open()
    try:
        f = cap.grab()
    finally:
        cap.close()
    rgb = cv2.cvtColor(f["rgb"], cv2.COLOR_BGR2RGB)
    depth_mm = np.clip(f["depth"] * 1000.0, 0, 65535).astype(np.uint16)
    Km = np.asarray(f["K"], float).reshape(3, 3)
    K = Km.flatten().tolist()
    h, w = rgb.shape[:2]
    fx, fy, cx, cy = Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2]
    # [DIAG] hl2ss PV K — 디바이스(srt3d)가 찍는 K 와 정규화값(fx/W 등)으로 비교. 여기가 정답 기준.
    print(f"[init_server] grab OK: rgb{f['rgb'].shape} depth유효{int((depth_mm>0).sum())}px")
    print(f"[init_server][DIAG] hl2ss PV K {w}x{h} fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  "
          f"norm fx/W={fx/w:.4f} fy/H={fy/h:.4f} cx/W={cx/w:.4f} cy/H={cy/h:.4f}")
    return rgb, depth_mm, K


def compute_init_pose():
    """텍스트(book) 자동 검출 경로 — 기존 GET /init 용. (SAM3 text → mask → FP /register)"""
    rgb, depth_mm, K = _grab_frame()
    mask = sam3_mask(rgb, OBJ_TEXT)
    cv2.imwrite(os.path.join(OUT, "init_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT, "init_mask.png"), (mask.astype(np.uint8) * 255))
    print(f"[init_server] text='{OBJ_TEXT}' mask{int(mask.sum())}px")

    payload = {"rgb": encode_png(rgb), "depth": encode_png(depth_mm),
               "mask": encode_png((mask.astype(np.uint8) * 255)),
               "K": K, "iteration": INIT_ITER}
    r = requests.post(f"{FP_URL}/register", json=payload, timeout=120)
    r.raise_for_status()
    pose = np.array(r.json()["pose"], float).reshape(4, 4)
    print(f"[init_server] pose:\n{pose}")
    return pose.reshape(16).tolist()


def compute_init_pose_box(box, text=None):
    """크롭 등록 경로 — POST /init_box 용. device 가 준 정규화 box[cx,cy,w,h] (0~1) →
    hl2ss grab → FP /register_with_box (내부에서 SAM3 box 프롬프트로 mask 생성).
    text 는 옵션(box 안 객체를 더 정확히 segment). 좌표변환 없음 — box 는 해상도 무관 정규화."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise ValueError(f"box must be [cx,cy,w,h] normalized 0~1, got {box}")
    rgb, depth_mm, K = _grab_frame()
    cv2.imwrite(os.path.join(OUT, "init_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[init_server] box={['%.3f' % v for v in box]} text={text!r}")

    payload = {"rgb": encode_png(rgb), "depth": encode_png(depth_mm),
               "K": K, "box": [float(v) for v in box], "iteration": INIT_ITER}
    if text:
        payload["text"] = str(text)
    r = requests.post(f"{FP_URL}/register_with_box", json=payload, timeout=120)
    r.raise_for_status()
    resp = r.json()
    pose = np.array(resp["pose"], float).reshape(4, 4)
    print(f"[init_server] pose:\n{pose}")
    return pose.reshape(16).tolist()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            self._send(200, {"ok": True}); return
        if p != "/init":
            self._send(404, {"error": "use GET /init or POST /init_box"}); return
        try:
            pose = compute_init_pose()
            self._send(200, {"ok": True, "pose": pose})
        except Exception as e:
            print(f"[init_server] FAIL: {type(e).__name__}: {e}")
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        p = self.path.split("?")[0]
        if p != "/init_box":
            self._send(404, {"error": "use POST /init_box"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode()) if n else {}
            box = body.get("box")
            # text 기본값 = OBJ_TEXT("book"). 기기가 text 를 안 보내도 SAM3 에 text 를 준다.
            #   근거(2026-07-22 실측): 박스 단독 프롬프트는 어수선한 장면에서 카오스
            #   (20px 밀리면 마스크가 배경 링으로 붕괴, 채움률 27%). text 를 넣으면 견고(88%).
            #   박스의 역할은 정밀 조준이 아니라 '어느 물체인지' 판별 → 축방향 밀림엔 관대.
            #   INIT_TEXT_ONLY=1 이면 box 를 버리고 text 만으로 등록(가장 깨끗한 마스크, 단일 객체용).
            text = body.get("text") or OBJ_TEXT
            if os.environ.get("INIT_TEXT_ONLY", "0") == "1":
                print(f"[init_server] /init_box: TEXT-ONLY (box 무시) text={text!r}")
                pose = compute_init_pose()  # text 경로: /register (box 없음)
            else:
                print(f"[init_server] /init_box: box+text text={text!r}")
                pose = compute_init_pose_box(box, text)
            self._send(200, {"ok": True, "pose": pose})
        except Exception as e:
            print(f"[init_server] FAIL: {type(e).__name__}: {e}")
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"[init_server] HL2={HL2_HOST} FP={FP_URL} SAM={SAM_ADDR} text='{OBJ_TEXT}'")
    print(f"[init_server] GET /init (text='{OBJ_TEXT}')  |  POST /init_box (crop box)  "
          f"on 0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
