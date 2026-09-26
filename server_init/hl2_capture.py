"""
HL2SS capture wrapper — PC가 HoloLens 2에서 한 프레임을
(RGB, 정렬된 depth, K, cam-to-world pose) 형태로 가져온다.

설계 위치 (경로2 파이프라인 1단계):
    HL2 ── hl2ss ──> PC[이 모듈] ──> FP.register() / SRT3D
즉 FoundationPose.register(K, rgb, depth, ob_mask) 에 바로 먹일 수 있는
RGB + (PV 이미지에 정렬된) depth(미터) + K(3x3 OpenCV 컨벤션)를 만든다.

원칙: 원본 hl2ss 레포는 건드리지 않는다. 그 viewer/ 폴더를 sys.path에 추가해
      라이브러리로만 사용한다. (sample_pv_depth_lt.py 의 정렬 로직을 함수로 추출.)

핵심 변환 메모:
- hl2ss 의 PV intrinsics(pv_fix_calibration 후)는 row-vector(전치) 컨벤션 4x4:
      [0,0]=fx, [1,1]=fy, [2,0]=cx, [2,1]=cy
  → OpenCV/FP 표준 3x3 K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
- depth 는 rm_depth_normalize 가 미터로 준다. FP 도 미터를 기대 → 단위 일치.
- depth 는 long-throw 센서 해상도에서 PV 이미지 좌표로 reproject 되어
  PV 와 같은 (height, width) 로 정렬된다 (정렬 안 된 곳은 0).
"""

import os
import sys
import time

import numpy as np

# --- hl2ss viewer 라이브러리를 path 에 추가 (원본 레포 수정 없음) -----------
_HL2SS_VIEWER = os.environ.get(
    "HL2SS_VIEWER",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "hl2ss", "viewer")),
)
if _HL2SS_VIEWER not in sys.path:
    sys.path.insert(0, _HL2SS_VIEWER)

import hl2ss          # noqa: E402
import hl2ss_lnm      # noqa: E402
import hl2ss_mp       # noqa: E402
import hl2ss_3dcv     # noqa: E402


def intrinsics_to_K(color_intrinsics):
    """hl2ss PV intrinsics(전치 4x4) → OpenCV 표준 3x3 K(float32)."""
    fx = float(color_intrinsics[0, 0])
    fy = float(color_intrinsics[1, 1])
    cx = float(color_intrinsics[2, 0])
    cy = float(color_intrinsics[2, 1])
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


class HL2Capture:
    """HoloLens 2 에서 PV+depth 동기 프레임을 가져오는 캡처기.

    사용 예:
        cap = HL2Capture("192.168.1.7")
        cap.open()
        frame = cap.grab()          # 첫 유효 프레임까지 블로킹
        cap.close()
    """

    def __init__(self, host, pv_width=640, pv_height=360, pv_fps=30,
                 calibration_path=None, max_depth=3.0):
        self.host = host
        self.pv_width = pv_width
        self.pv_height = pv_height
        self.pv_fps = pv_fps
        self.max_depth = max_depth
        # 캘리브레이션은 없으면 hl2ss 가 디바이스에서 받아 여기에 캐시한다.
        self.calibration_path = calibration_path or os.path.join(
            os.path.dirname(__file__), "calibration")
        os.makedirs(self.calibration_path, exist_ok=True)

        self._sink_pv = None
        self._sink_lt = None
        self._calib_lt = None
        self._xy1_o = None
        self._xy1_d = None
        self._scale = None
        self._pv_intrinsics = None
        self._pv_extrinsics = None

    # -- lifecycle ----------------------------------------------------------
    def open(self):
        host = self.host

        # PV 서브시스템 시작
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # long-throw depth 캘리브레이션 (없으면 디바이스에서 다운로드)
        self._calib_lt = hl2ss_3dcv.get_calibration_rm(
            self.calibration_path, host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
        xy1, self._scale = hl2ss_3dcv.rm_depth_compute_rays(
            self._calib_lt.uv2xy, self._calib_lt.scale)
        # origin / diagonal 코너 광선 (zero-order hold 블록 채우기용)
        self._xy1_o = xy1[:-1, :-1, :]
        self._xy1_d = xy1[1:, 1:, :]

        # 스트림 sink 시작
        self._sink_pv = hl2ss_mp.stream(
            hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO,
                            width=self.pv_width, height=self.pv_height,
                            framerate=self.pv_fps))
        self._sink_lt = hl2ss_mp.stream(
            hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
        self._sink_pv.open()
        self._sink_lt.open()

        # PV intrinsics/extrinsics 자리표 (오토포커스로 매 프레임 갱신됨)
        self._pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
        self._pv_extrinsics = np.eye(4, 4, dtype=np.float32)
        return self

    def close(self):
        if self._sink_pv is not None:
            self._sink_pv.close()
            self._sink_pv = None
        if self._sink_lt is not None:
            self._sink_lt.close()
            self._sink_lt = None
        hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # -- capture ------------------------------------------------------------
    def grab(self, timeout_s=30.0, poll_s=0.03):
        """첫 유효 (PV pose & depth pose) 프레임을 가져와 dict 반환.

        long-throw depth 는 5 FPS 라 프레임 간격이 ~200ms → 시간 기반으로
        대기하며 폴링한다 (즉시 회전하면 프레임이 도착하기 전에 끝남).

        반환 dict:
            rgb          : (H, W, 3) uint8  — PV 컬러 (hl2ss 기본 BGR)
            depth        : (H, W)   float32 — PV 에 정렬된 depth(미터), 빈 곳 0
            K            : (3, 3)   float32 — OpenCV 컨벤션 intrinsics
            pv_pose      : (4, 4)   float32 — PV reference-to-world (row-vector)
            color_extrinsics : (4,4) float32 — rignode↔camera
            timestamp    : int              — depth 프레임 타임스탬프 (100ns)
        실패 시(timeout) RuntimeError (왜 실패했는지 카운터 포함).
        """
        deadline = time.time() + timeout_s
        # 진단용 카운터
        n_no_lt = n_lt_badpose = n_no_pv = n_pv_badpose = 0
        while time.time() < deadline:
            _, data_lt = self._sink_lt.get_most_recent_frame()
            if data_lt is None:
                n_no_lt += 1
                time.sleep(poll_s)
                continue
            if not hl2ss.is_valid_pose(data_lt.pose):
                n_lt_badpose += 1
                time.sleep(poll_s)
                continue

            _, data_pv = self._sink_pv.get_nearest(data_lt.timestamp)
            if data_pv is None:
                n_no_pv += 1
                time.sleep(poll_s)
                continue
            if not hl2ss.is_valid_pose(data_pv.pose):
                n_pv_badpose += 1
                time.sleep(poll_s)
                continue

            return self._build_frame(data_lt, data_pv)

        raise RuntimeError(
            f"[hl2_capture] {timeout_s:.0f}s 안에 유효 프레임 없음. 진단: "
            f"depth프레임없음={n_no_lt}, depth포즈무효={n_lt_badpose}, "
            f"PV매칭없음={n_no_pv}, PV포즈무효={n_pv_badpose}.\n"
            f"  - depth/PV프레임없음 → 스트림 미수신(서버 앱/Research Mode/방화벽).\n"
            f"  - 포즈무효 → 기기 공간추적 끊김(주변이 너무 단조롭거나 어두움; "
            f"움직이며 특징 많은 곳을 보게 한 뒤 재시도).")

    # -- internal -----------------------------------------------------------
    def _build_frame(self, data_lt, data_pv):
        calib_lt = self._calib_lt
        pv_w, pv_h = self.pv_width, self.pv_height

        depth = data_lt.payload.depth
        z = hl2ss_3dcv.rm_depth_normalize(depth, self._scale)
        color = data_pv.payload.image

        # PV intrinsics 갱신 (오토포커스 대응) + 부호 보정
        self._pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(
            self._pv_intrinsics,
            data_pv.payload.focal_length, data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(
            self._pv_intrinsics, self._pv_extrinsics)

        # long-throw depth → world → PV 카메라 → PV 이미지 (sample_pv_depth_lt 로직)
        lt_to_world = (hl2ss_3dcv.camera_to_rignode(calib_lt.extrinsics)
                       @ hl2ss_3dcv.reference_to_world(data_lt.pose))
        world_to_pv = (hl2ss_3dcv.world_to_reference(data_pv.pose)
                       @ hl2ss_3dcv.rignode_to_camera(color_extrinsics))
        pv_to_pv_image = hl2ss_3dcv.camera_to_image(color_intrinsics)

        # zero-order hold 블록 채우기: origin/diagonal 코너 광선으로 PV 픽셀 사각형을 채운다
        xy1_o, xy1_d = self._xy1_o, self._xy1_d
        zc = z[:-1, :-1, :]

        lt_pts_o = hl2ss_3dcv.rm_depth_to_points(xy1_o, zc)
        pv_pts_o = hl2ss_3dcv.transform(
            hl2ss_3dcv.transform(lt_pts_o, lt_to_world), world_to_pv)
        pv_depth = pv_pts_o[:, :, 2:]
        pv_uv_o = hl2ss_3dcv.project(pv_pts_o, pv_to_pv_image)

        lt_pts_d = hl2ss_3dcv.rm_depth_to_points(xy1_d, zc)
        pv_uv_d = hl2ss_3dcv.project(
            hl2ss_3dcv.transform(lt_pts_d, lt_to_world),
            world_to_pv @ pv_to_pv_image)

        list_o = hl2ss_3dcv.block_to_list(pv_uv_o)
        list_d = hl2ss_3dcv.block_to_list(pv_uv_d)
        list_depth = hl2ss_3dcv.block_to_list(pv_depth)
        mask = (depth[:-1, :-1].reshape((-1,)) > 0)

        pv_list = np.hstack((np.floor(list_o[mask, :]),
                             np.floor(list_d[mask, :]) + 1,
                             list_depth[mask]))
        pv_z = np.zeros((pv_h, pv_w), dtype=np.float32)
        for n in range(pv_list.shape[0]):
            u0, v0 = int(pv_list[n, 0]), int(pv_list[n, 1])
            u1, v1 = int(pv_list[n, 2]), int(pv_list[n, 3])
            if (u0 < 0) or (u0 >= pv_w) or (u1 < 0) or (u1 > pv_w):
                continue
            if (v0 < 0) or (v0 >= pv_h) or (v1 < 0) or (v1 > pv_h):
                continue
            pv_z[v0:v1, u0:u1] = pv_list[n, 4]

        return {
            "rgb": color,
            "depth": pv_z,
            "K": intrinsics_to_K(color_intrinsics),
            "pv_pose": np.asarray(data_pv.pose, dtype=np.float32),
            "color_extrinsics": np.asarray(color_extrinsics, dtype=np.float32),
            "timestamp": int(data_lt.timestamp),
        }
