# wiseui-hl2 tracking 공유용

홀로렌즈2 객체 추적 데모에서 tracking 관련 코드/데이터만 뽑아놓은 repo.
유니티 클라이언트 전체는 https://github.com/youngbin0212/wiseui-hl2-client

## 구조

추적 자체는 홀로렌즈 안에서 SRT3D(`srt3d_uwp.dll`)가 매 프레임 돌고,
PC 서버는 처음에 초기 pose 잡을 때만 사용 (SAM3 마스크 → FoundationPose register).
초기화 끝나면 서버 안 거침.

`fp_server_gxr.py`의 gxr는 예전 갤럭시XR 실험 때 이름이 남은 것. 홀로렌즈에서 그대로 쓰는 코드.

## 폴더

- `client_ondevice_tracking/` — 유니티쪽 추적 스크립트 3개 + srt3d_uwp.dll 소스 (CMake, ARM64 UWP 빌드 스크립트 포함)
  - 빌드하려면 eigen3, tiny_obj_loader 필요 (용량 때문에 뺌, 공개 라이브러리)
  - 코어는 pysrt3d(https://github.com/Jianxff/pysrt3d)의 nogl 버전 기반. 메쉬는 .obj 옆에 srt3d용 `.obj.meta` 필요 (pysrt3d의 gen_meta.py로 생성)
- `server_init/` — 초기화 서버 코드, 실제 돌리던 파일 그대로
  - `init_server.py`, `hl2_capture.py` : 윈도우 conda. 패키지는 requirements.txt. hl2ss는 pip이 아니라 hl2ss 체크아웃의 viewer 폴더를 `HL2SS_VIEWER` 환경변수로 지정
  - `sam3_server.py` : WSL conda (sam3 공식 설치 + sam3.1_multiplex.pt)
  - `fp_server_gxr.py` : FoundationPose 도커 컨테이너 안에서 실행. 컨테이너에 `pip install pyzmq` 필요, `patches/foundationpose.patch` 적용해야 함
  - `config/dependencies.lock.yaml` : upstream 커밋 버전 기록
- `mesh_tissue_box/` — 데모에 쓴 각티슈 메쉬 (직접 제작, 6면 촬영해서 만든 것. obj/mtl/atlas.png/meta + 제작 소스)

## 데이터셋

전부 로컬 PC에 받아서 사용 (연구실 서버 아님).

- 각티슈: 직접 제작 → 이 repo에 포함
- 나머지 실험 객체 메쉬: Amazon Picking Challenge object scans — https://rll.berkeley.edu/amazon_picking_challenge/
  (joke_book, tennis_ball, school_glue, dove_bar, eggs_puppies, duck_toy, pencil_12, plastic_stir / 각 객체의 textured_meshes/optimized_poisson_texture_mapped_mesh.obj 사용)
- Bowl_White (fp_server_gxr.py 기본 메쉬): Google Scanned Objects의 Threshold_Bead_Cereal_Bowl_White
- FoundationPose weights: 공식 릴리스 2개 (2023-10-28-18-33-37 refiner / 2024-01-11-20-02-45 scorer)
- SAM3 weights: sam3.1_multiplex.pt

별도 학습/평가용 데이터셋 없음.

## 실행코드 (객체: 각티슈)

```powershell
# 서버 (VS Code로 wiseui-hl2 열고)
conda activate my_base
$env:HL2SS_VIEWER = "D:\ProjectsTracking\wiseui-hl2\dependencies\hl2ss\viewer"
$env:OBJ_TEXT = "tissue box"
cd server\wiseui-hl2-server-local-archive
python init_server.py
```

```bash
# wsl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3
cd /mnt/d/ProjectsTracking/wiseui-hl2/dependencies/sam3
python sam3_server.py
```

```bash
# wsl -> docker
docker ps -a
docker start d4d62b7aeefa   # FoundationPose 컨테이너 (ID는 PC마다 다름)
docker exec -it d4d62b7aeefa bash

cd /mnt/d/ProjectsTracking/wiseui-hl2/dependencies/FoundationPose
python fp_server_gxr.py --mesh my_data/tissue_box/textured_meshes/tissue_box_hl2c.obj
```

이후 홀로렌즈 앱에서 초기화(에어탭 or 텍스트 자동검출)하면 온디바이스 추적 시작.
포트: SAM3 5556(ZMQ) / FoundationPose 8000 / init 서버 8002

## upstream 버전

- FoundationPose https://github.com/NVlabs/FoundationPose — e3d597b (+ patches/foundationpose.patch)
- sam3 https://github.com/facebookresearch/sam3 — f66a251
- hl2ss https://github.com/jdibenes/hl2ss — fcc4e84
