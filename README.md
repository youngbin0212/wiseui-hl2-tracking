# WiseUI HL2 — Tracking 모듈 & 데이터셋 공유 (김영빈, 2026-09-26)

HoloLens 2 기반 WiseUI 객체 추적 데모에서 tracking에 실제로 쓰인 코드와 데이터셋입니다.

## 1. 추적 구조 (누가 무엇을 추적하나)

- **매 프레임 6DoF 추적 = SRT3D, HoloLens 온디바이스** (`srt3d_uwp.dll`, UWP ARM64 네이티브 플러그인).
  Unity 클라이언트가 PV 카메라 프레임을 넣고 pose를 받습니다.
- **초기 pose 등록(초기화/재초기화) = PC 서버**: SAM3가 마스크 생성 → FoundationPose가 초기 pose register.
  초기화 이후 프레임 추적은 서버를 거치지 않습니다.
- 파일명 `fp_server_gxr.py`의 "gxr"는 Galaxy XR 실험에서 시작한 레거시 이름이고, HL2 시스템이 그대로 사용 중입니다.

```
HoloLens 2 (Unity/UWP + hl2ss 스트리밍)
 ├─ 초기화: → init_server(Windows, :8002) → SAM3(WSL, ZMQ :5556) → FoundationPose(Docker, :8000) /register
 └─ 이후 매 프레임: srt3d_uwp.dll 온디바이스 추적 (srt3d_track_rgb)
```

## 2. 데이터셋 / 모델 (전부 로컬 PC 보관, 연구실 서버 아님 — 공개 데이터는 이름·링크로 충분)

| 용도 | 출처 | 비고 |
|---|---|---|
| 데모 객체 **tissue_box(각티슈)** | **자체 제작 메쉬** (실물 6면 촬영 → 텍스처 박스 메쉬) | 공개 데이터셋 아님 → **`mesh_tissue_box/`에 동봉**: `tissue_box_hl2c.obj`+`.mtl`+텍스처 `atlas.png`+srt3d용 `.obj.meta`, 제작 소스 `source_rectified/` 포함 |
| 그 외 실험 객체 메쉬 | **Berkeley Amazon Picking Challenge object scans** — <https://rll.berkeley.edu/amazon_picking_challenge/> | joke_book, tennis_ball, school_glue, dove_bar, eggs_puppies, duck_toy, pencil_12, plastic_stir (`textured_meshes/optimized_poisson_texture_mapped_mesh.obj`) |
| 단일-메쉬 기본값 | **Google Scanned Objects** `Threshold_Bead_Cereal_Bowl_White` | fp_server_gxr.py 기본 메쉬 |
| FoundationPose 가중치 | NVlabs 공식 체크포인트 `2023-10-28-18-33-37`(refiner), `2024-01-11-20-02-45`(scorer) | FoundationPose GitHub 릴리스 |
| SAM3 가중치 | `sam3.1_multiplex.pt` (SAM 3.1 공식) | |

별도 학습/평가용 데이터셋은 없습니다. (FoundationPose demo_data의 mustard0 등은 서버 self-test 전용.)

## 3. 코드

### client_ondevice_tracking/ — 매 프레임 추적 (SRT3D)

| 파일 | 역할 |
|---|---|
| `Srt3dTracker.cs` | Unity 측 추적 루프 — init_server에서 초기 pose 받아 srt3d에 주입, 이후 온디바이스 추적 + 좌표 변환 |
| `Srt3dNative.cs` | `srt3d_uwp.dll` P/Invoke 바인딩 (init / reset_pose / track_rgb, C ABI 6함수) |
| `Srt3dPvCapture.cs` | HoloLens PV 카메라 프레임 캡처 |
| `srt3d_uwp_native/` | **DLL 소스 전체** — `srt3d_uwp.cpp`(C ABI 래퍼) + `srt3d/`(코어) + CMake + `build_uwp_arm64.ps1` |

SRT3D 코어 계보: SRT3D(Stoiber et al., 3DObjectTracking) → pysrt3d(<https://github.com/Jianxff/pysrt3d>)의
OpenGL 제거 버전(source_nogl) → UWP ARM64 래핑(`srt3d_uwp`). 빌드에는 `third_party/`로 eigen3,
tiny_obj_loader가 추가로 필요합니다(용량 문제로 zip에서 제외 — 둘 다 공개 라이브러리).
각 객체 메쉬는 `.obj` 옆에 srt3d용 `.obj.meta`가 필요합니다(pysrt3d의 gen_meta.py로 생성).

### server_init/ — 초기화 경로 (실행에 실제 쓴 파일 그대로)

| 파일 | 역할 |
|---|---|
| `init_server.py` | Windows 초기화 서버(:8002) — hl2ss로 PV+depth 캡처, SAM3→FP 오케스트레이션, `/init`(텍스트 자동검출)·`/init_box`(박스 지정). `OBJ_TEXT` 환경변수로 대상 텍스트 지정 |
| `hl2_capture.py` | hl2ss 캡처 모듈 (init_server 의존) |
| `sam3_server.py` | SAM3 세그멘테이션 ZMQ 서비스(:5556, WSL conda) |
| `fp_server_gxr.py` | FoundationPose HTTP 서비스(:8000, Docker) — `/register`, `/track`, `/register_with_box`, `/health` |
| `patches/foundationpose.patch` | upstream FoundationPose 로컬 수정 (estimater.py, predict_score.py) |
| `config/dependencies.lock.yaml` | upstream 고정 커밋 기록 |

(다중 오브젝트 멀티-메쉬 변형, 타이밍 측정 변형, 테스트 클라이언트도 있습니다 — 필요하시면 말씀 주세요.)

## 4. 실제 실행 커맨드 (객체: 각티슈)

```powershell
# ① Windows — init 서버 (conda my_base)
$env:HL2SS_VIEWER = "D:\...\wiseui-hl2\dependencies\hl2ss\viewer"
$env:OBJ_TEXT = "tissue box"
python init_server.py
```

```bash
# ② WSL — SAM3 (conda sam3)
cd .../wiseui-hl2/dependencies/sam3
python sam3_server.py

# ③ WSL → Docker — FoundationPose
docker exec -it <container> bash
cd /mnt/d/.../wiseui-hl2/dependencies/FoundationPose
python fp_server_gxr.py --mesh my_data/tissue_box/textured_meshes/tissue_box_hl2c.obj
```

이후 HoloLens 앱에서 초기화(에어탭/텍스트 자동검출) → 온디바이스 SRT3D 추적.

## 5. 환경 / 설치

| 구성요소 | 환경 | 설치 |
|---|---|---|
| init 서버 (Windows) | conda env (예: my_base) | `pip install -r server_init/requirements.txt` (numpy, Pillow, opencv-python, requests, pyzmq, av). hl2ss는 pip 패키지가 아니라 hl2ss 체크아웃의 `viewer/` 폴더를 `HL2SS_VIEWER` 환경변수로 잡아서 사용 |
| SAM3 (WSL2) | conda env `sam3` | SAM3 공식 저장소 설치 절차 그대로 + 체크포인트 `sam3.1_multiplex.pt` 경로 지정 |
| FoundationPose (Docker) | upstream 공식 docker 이미지 (CUDA 필요) | FoundationPose 저장소의 `docker/` 셋업 그대로 + 컨테이너 안에 `pip install pyzmq` 추가 + `patches/foundationpose.patch` 적용 + 공식 weights 2종을 `weights/`에 배치 |
| srt3d_uwp.dll 빌드 | Visual Studio ARM64 UWP 툴체인 + CMake | `srt3d_uwp_native/build_uwp_arm64.ps1`. third_party로 eigen3, tiny_obj_loader 필요(공개 라이브러리, zip에서 제외) |
| HL2 클라이언트 | Unity/UWP | 클라이언트 저장소(GitHub public) 참고 — 빌드된 앱에는 hl2ss 스트리밍 플러그인 포함 |

## 6. Upstream 의존성 (고정 커밋)

| 저장소 | 커밋 | 비고 |
|---|---|---|
| <https://github.com/NVlabs/FoundationPose> | `e3d597b` | `patches/foundationpose.patch` 적용 |
| <https://github.com/facebookresearch/sam3> | `f66a251` | |
| <https://github.com/jdibenes/hl2ss> | `fcc4e84` | HL2 스트리밍 |
| <https://github.com/Jianxff/pysrt3d> | (checkout) | SRT3D 코어의 출발점 |

## 7. GitHub 저장소

- 클라이언트 (public): <https://github.com/youngbin0212/wiseui-hl2-client> — Unity/UWP HL2 앱 전체
- 서버 (private): <https://github.com/youngbin0212/wiseui-hl2-server> — 정리된 서버 저장소. 전체가 필요하시면 collaborator로 초대해 드릴게요.
