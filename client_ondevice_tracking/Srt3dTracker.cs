// 트랙B-A 시퀀싱: FP init(hl2ss) → hl2ss stop → MediaFrameReader(PV) 로 srt3d 추적.
//
// 흐름:
//   1. (앱 시작 시 Hl2ssBootstrap 가 hl2ss PV+depth ON)
//   2. PC init_server(GET /init)에 초기 pose 요청 — PC가 wiseui hl2ss 에서 PV+depth 받아
//      SAM+FP 로 ob_in_cam pose 계산해 회신 (트랙A 그대로, 좌표문제 0 — 같은 카메라)
//   3. Hl2ssBootstrap.Stop() → 카메라 반납
//   4. MFR(PV 896x504@30) ON → 첫 프레임에 srt3d_init + reset_pose(FP pose) → 매 프레임 track
//      → 박스를 추적 pose 에 렌더 (srt3d 는 RGB-only 라 depth 불필요)
//
// 캡처 경로는 Srt3dPvCapture(MediaFrameReader) 하나뿐. PhotoCapture 경로는 제거됨 —
// 4K(3904x2196)만 줘서 fps≈0 이었고, cam2world 광학축 규약(−Z fwd)이 MFR(+Z fwd)과 달라
// 좌표 디버깅 때 "어느 경로가 실제로 도는가"를 매번 재확인해야 했음.
//
// 사전(PC): sam3_server(5556) + fp_server_gxr(8000,tissue_box) + init_server.py(8002)
// 빌드 전: Capabilities WebCam + InternetClient + PrivateNetworkClientServer.
// [TUNE] _serverBaseUrl (PC init_server 주소, Inspector 노출) / 좌표 후보는 C_CV2U·S_LEGACY 주석 참조

using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using Microsoft.MixedReality.Toolkit;             // CoreServices, MixedRealityPose
using Microsoft.MixedReality.Toolkit.Input;       // IMixedRealityPointerHandler, HandJointUtils
using Microsoft.MixedReality.Toolkit.Utilities;   // TrackedHandJoint, Handedness

public class Srt3dTracker : MonoBehaviour, IMixedRealityPointerHandler
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        var go = new GameObject("Srt3dTracker");
        DontDestroyOnLoad(go);
        go.AddComponent<Srt3dTracker>();
    }

    // ── OpenCV cam ↔ Unity cam 변환 후보 (한 빌드 판별용) ──────────────────
    // C = diag(1,-1,1) : OpenCV(X right, Y down, Z fwd, RH) → Unity(X right, Y up, Z fwd, LH).
    //     Z 는 안 뒤집힘 — 양쪽 다 +Z 전방.
    // S = diag(1,-1,-1): 구 PhotoCapture 용. TryGetCameraToWorldMatrix 는 −Z forward 규약이라
    //     (MS 샘플이 pos = col3 − col2, LookRotation(−col2, col1) 을 쓰는 게 근거) Z 반전이 필요했음.
    //     MS ToUnity() 로 만든 MFR cam2world 는 F=diag(1,1,-1) conjugation 을 거쳐 +Z forward
    //     (Unity 표준) → 여기선 S 의 Z 반전이 잉여. cam2world 출처가 바뀌며 전제가 깨진 것.
    // translation 비교: S·M → (tx,−ty,−tz) / C·M·C → (tx,−ty,+tz). Z 만 다름.
    static readonly Matrix4x4 C_CV2U = Matrix4x4.Scale(new Vector3(1f, -1f, 1f));
    static readonly Matrix4x4 S_LEGACY = Matrix4x4.Scale(new Vector3(1f, -1f, -1f));
    // 후보 world 행렬 (렌더는 B 하나만, A/D 는 HUD 진단값 병기용 — auto-pick 아님).
    Matrix4x4 _candA, _candB, _candD;

    // ── [gaze 1단계] 고정 깊이 가정으로 시선 픽셀 계산 (등록 로직은 아직 안 건드림) ────
    // spatial mesh 레이캐스트는 안 쓴다: 정적 환경만 담고 있어 손에 든 책을 못 맞추고
    // 뒤 벽/모니터를 맞는다 → NOHIT 보다 나쁜 "그럴듯하게 틀린" 깊이가 나온다.
    //
    // 대신 카메라 공간의 z = GazeZ 평면과 교차시킨다. 눈-PV 베이스라인은 ray 원점이
    // 카메라 원점이 아니라는 점으로 자동 반영됨(= 고정 깊이 하의 parallax 보정).
    // 깊이 가정이 틀린 만큼의 픽셀 오차는 baseline·fx 규모: b=5cm, fx=686, 40cm 가정이면
    // 25~60cm 구간에서 ~50px 이내. 40cm 에서 책 너비 10cm ≈ 171px 이므로 박스를 1.5배로
    // 잡으면 감싼다 → SAM3 박스 프롬프트엔 충분. 이게 성립하면 point cloud / 네이티브
    // 플러그인 / PV 점유 타이밍이 전부 불필요해진다.
    //
    // GazeZ 는 추적 중 핀치로 순환한다. 점이 책 위에 떨어지는 z 를 찾으면 그게 곧
    // 눈-PV 베이스라인 실측이 된다(현재 5cm 는 추정값).
    static readonly float[] GAZE_Z_OPTIONS = { 0.25f, 0.30f, 0.40f, 0.50f, 0.60f };
    int _gazeZIdx = 2;                                  // 기본 0.40m
    float GazeZ { get { return GAZE_Z_OPTIONS[_gazeZIdx]; } }
    bool _gazeEnableTried = false;
    string _gazeHud = "gaze: (init)";
    Vector2 _gazePx = new Vector2(-1f, -1f);
    bool _gazePxValid = false;
    // [gaze 2단계] MRTK 우회 — OpenXR(UnityEngine.XR) 에서 직접 눈 데이터를 읽는다.
    // MRTK 경로는 cal=? 에서 멈춰 있었다(= 활성 프로필이 read-only 라 eye gaze data provider
    // 자체가 등록 안 됨 → 보정 상태조차 못 읽음). 아래가 주 경로, MRTK 는 대조용으로 병기.
    readonly List<UnityEngine.XR.InputDevice> _xrEyeDevs = new List<UnityEngine.XR.InputDevice>();
    string _gazeMrtkHud = "mrtk: (init)";
    // 최신 PV 프레임의 K/pose — world 점을 PV 픽셀로 투영하는 데 필요(검증된 값 그대로 재사용).
    Matrix4x4 _lastC2W = Matrix4x4.identity; bool _hasLastC2W = false;
    float _lastFx, _lastFy, _lastCx, _lastCy;

    // [TUNE]
    int _pvWidth = 760, _pvHeight = 428;
    // PC init_server 주소. Inspector 에서 실험 환경에 맞게 바꿀 것 (기본값은 로컬 루프백).
    // 두 엔드포인트가 호스트·포트를 공유하므로 base 만 노출하고 경로는 파생시킨다.
    [SerializeField]
    [Tooltip("PC init_server 의 base URL. 형식: http://<PC-IP>:<PORT> (끝의 / 는 있어도 됨)")]
    string _serverBaseUrl = "http://127.0.0.1:8002";
    string ServerBase { get { return (_serverBaseUrl ?? "").TrimEnd('/'); } }
    string InitUrl { get { return ServerBase + "/init"; } }          // 텍스트 자동검출(구 방식, fallback)
    string InitBoxUrl { get { return ServerBase + "/init_box"; } }   // 크롭 등록(두 모서리 pinch)
    // 초기 등록 방식 [TUNE _initMode]:
    //   TextCenter = 객체를 중앙에 두고 SAM text('book') 자동검출 (구 방식, GET /init)
    //   CenterBox  = 화면 중앙 고정 박스에 객체 맞추고 air-tap 1회 (POST /init_box)
    //   CropPinch  = 두 모서리 air-tap 으로 박스 직접 지정 (POST /init_box)
    //   DragDraw   = 핀치로 한 모서리→반대 모서리 드래그해 박스 직접 그림 (POST /init_box)
    enum InitMode { TextCenter, CenterBox, CropPinch, DragDraw }
    InitMode _initMode = InitMode.CenterBox;
    // 세로 박스 프리셋 (정규화 w,h). 정사각 폐기 — 실측(2026-07-22): 세로로 긴 물체에 정사각
    // 박스를 씌우면 좌우 테이블이 절반이라 SAM 이 배경을 잡음(채움 24%). 세로 박스는 74%+.
    // 박스는 정밀 조준이 아니라 '대충 여기 + 어느 물체'(text 가 판별) 용도라 크기 여유는 관대.
    //   pinch = 크기 순환 (gaze 확정 모드일 때). fallback(머리조준) 모드에선 pinch = 확정.
    static readonly Vector2[] BOX_PRESETS = {
        new Vector2(0.25f, 0.70f),   // ★ 실측 검증값 (프로브 vert, 채움 74%). 픽셀종횡비 1.575
        new Vector2(0.30f, 0.80f),   // 조금 크게 (같은 길쭉함 유지, 1.51)
        new Vector2(0.20f, 0.60f),   // 작게 (근접/작은 물체, 1.66)
    };
    int _boxPresetIdx = 0;           // 기본 = 검증값
    float _boxBaseW = 0.50f;         // [TUNE] 박스 폭(정규화). 높이는 cam.aspect 로 시야 정사각 계산.
                                      //   0.40 -> 0.50: 각티슈(23.5x12x11.5cm)가 박스보다 커 보인다는
                                      //   실측 피드백(2026-09-18)으로 키움. 박스는 SAM3 text 프롬프트의
                                      //   "대충 여기" 힌트라 정밀 조준 불필요 — 여유 있게 크게 잡아도 무방.
    float _centerBoxW = 0.40f, _centerBoxH = 0.55f;  // 매 프레임 재계산됨
    bool _startReq = false;          // 등록 시작 요청 (핀치 또는 음성 "start")
    Vector2 _boxCenterNorm = new Vector2(0.5f, 0.5f); // 박스 중심(이미지 정규화). gaze 있으면 gaze, 없으면 0.5
    bool _centerBoxMode = false;                   // (내부) 가운데박스 air-tap 분기용
    bool _drawMode = false;                         // (내부) 드래그-드로우 진행중
    Vector2 _dragA, _dragB;                         // 박스 미리보기 두 모서리(이미지 정규화)
    LineRenderer _boxLr;                             // 박스 시각화(가운데/드로우 공용)
    // [DIAG/입력] 핀치 감지 진단·폴백. MRTK 전역 클릭이 안 오면 손관절 거리로 직접 핀치 검출.
    int _clickCount = 0, _pinchCount = 0; bool _wasPinch = false; float _pinchDist = -1f;
    float _pinchOn = 0.03f, _pinchOff = 0.05f;     // [TUNE] 핀치 on/off 임계(히스테리시스, m)
    // ── UI 상태 ─────────────────────────────────────────────────────────
    bool _showDebug = false;           // 디버그 HUD 토글 (음성 "toggle debug"). 기본 꺼짐.
    string _regStatus = "";            // 등록 단계 문구 (한 줄). 운용 중엔 미사용.
    float _dwellSec = 3.0f;            // [TUNE] gaze dwell 확정 시간(초)
    float _dwellRadiusNorm = 0.06f;    // [TUNE] 이 반경(정규화) 안에 gaze 가 머물면 dwell 누적
    float _dwellT = 0f;                // 현재 dwell 누적
    Vector2 _dwellAnchor;              // dwell 시작 지점(정규화)
    bool _gazeForBox = false;          // 이번 프레임 박스 위치를 gaze 로 잡았나(진단/분기)
    MeshRenderer _boxFill;             // 박스 반투명 채움(quad)
#if !UNITY_EDITOR
    UnityEngine.Windows.Speech.KeywordRecognizer _kw;   // 음성 "toggle debug"
#endif

    // 두 모서리 air-tap 수집 상태 (인터페이스 콜백에서 채움). 이미지 정규화(원점 top-left) 좌표.
    readonly List<Vector2> _corners = new List<Vector2>();
    bool _selecting = false, _boxReady = false, _handlerRegistered = false;
    float[] _boxCxcywh;

    TextMesh _hud; Transform _target;
    bool _inited = false, _tracking = false;
    string _meshPath; int _frame = 0; float _conf = 0f;
    Srt3dPvCapture _pvCap;
    bool _mfrRunning = false; byte[] _mfrRgb; bool _lastHasPose = false;
    float _obZ = 0f;   // [DIAG-1] ob_in_cam Z (양수=카메라 앞=srt3d pose 정상, 음수=srt3d pose 이상)
    bool _dumpedFrame = false;      // [DIAG-B] 첫 MFR 프레임 PNG 덤프 (conf=0 이미지 확인)
    float _projU = -1f, _projV = -1f; int _mfrW, _mfrH;  // [DIAG] 객체중심 투영 위치 (이미지 안이면 K/pose OK)
    // [DIAG] 처리 fps (버퍼링 정량화). Update 에서 _frame 증가율로 계산.
    float _fps = 0f, _fpsAccum = 0f; int _fpsLastFrame = 0;
    // [DIAG] 객체/헤드 world 좌표. 머리만 움직일 때 objW 가 (거의) 그대로면 world 좌표계 정상 + 트래킹 OK.
    Vector3 _objWorld, _headWorld;
    float _trackElapsed = 0f;    // [DIAG] 누적 평균 fps 용 (총 프레임/경과)
    string _kInfo = "";           // [DIAG] srt3d 에 넘긴 K (init 시 계산)
    float[] _prevPose;            // [DIAG] 직전 프레임 ob_in_cam (Δ 계산용)
    float _dT = 0f, _dR = 0f;     // [DIAG] 프레임간 translation(m) / rotation(도) 변화량
    float[] _fpPose;   // FP 초기 pose (row-major 16, ob_in_cam OpenCV)
    Matrix4x4 _renderCorrection = Matrix4x4.identity; // depth plane 기반 object-local 보정
    Matrix4x4 _pendWorld; bool _hasPend = false;
    // per-vertex 렌더 (gxr 방식): object 정점을 매 프레임 full 4x4 로 world 변환 → Matrix4x4.rotation/
    // Transform 분해(반사행렬에서 깨짐) 회피. GO 는 identity 유지, mesh 정점 자체가 world.
    // ★ 후보 B(C·M·C, det +1)든 A(C·M, det −1, 거울상)든 정점 변환은 정확 — 판별에 유리.
    Vector3[] _objVerts; int[] _tris; Mesh _mesh; Vector3[] _worldVerts;
    LineRenderer[] _axes; Shader _stdShader;

    void Start()
    {
        var cube = GameObject.Find("Cube");           // SampleScene 튜토리얼 큐브 — 불필요, 끔
        if (cube != null) cube.SetActive(false);
        _hud = CreateHud(); _target = CreateTarget();
        StartCoroutine(HideVisualProfiler());
        SetupVoice();
        Hud("srt3d: copying model...");
        StartCoroutine(Boot());
    }

    IEnumerator HideVisualProfiler()
    {
        // MRTK's default profile initializes asynchronously and can re-enable
        // diagnostics after this component's Start(). Keep the service off so
        // initialization order cannot bring the CPU/GPU window back.
        while (true)
        {
            if (CoreServices.DiagnosticsSystem != null)
            {
                CoreServices.DiagnosticsSystem.ShowProfiler = false;
                CoreServices.DiagnosticsSystem.ShowDiagnostics = false;
            }
            yield return null;
        }
    }

    // 음성 "toggle debug" → 디버그 HUD on/off. MRTK Speech 프로필(read-only)에 의존하지 않도록
    // Windows 음성 API(KeywordRecognizer)를 직접 쓴다. 손 안 써도 등록 중에 켜고 끌 수 있음.
    void SetupVoice()
    {
#if !UNITY_EDITOR
        try
        {
            _kw = new UnityEngine.Windows.Speech.KeywordRecognizer(
                new[] { "toggle debug", "debug" });
            _kw.OnPhraseRecognized += (args) => { _showDebug = !_showDebug; };
            _kw.Start();
        }
        catch (System.Exception e) { Debug.LogWarning("[Srt3dTracker] voice init fail: " + e.Message); }
#endif
    }

    IEnumerator Boot()
    {
        string configSource = "fallback";
        yield return WiseUiRuntimeConfig.Load((url, source) =>
        {
            _serverBaseUrl = url;
            configSource = source;
        });
        Debug.Log($"[Srt3dTracker] init server={ServerBase} source={configSource}");

        string d = Path.Combine(Application.persistentDataPath, "srt3d");
        Directory.CreateDirectory(d);
        _meshPath = Path.Combine(d, "tissue_box_hl2c.obj");
        string meta = Path.Combine(d, "tissue_box_hl2c.obj.meta");
        yield return Copy("srt3d/tissue_box_hl2c.obj", _meshPath);
        yield return Copy("srt3d/tissue_box_hl2c.obj.meta.bytes", meta);
        if (!File.Exists(_meshPath) || !File.Exists(meta)) { Hud("FAIL: model copy"); yield break; }

        // 렌더용 full solid mesh 를 .obj 좌표 그대로 파싱해 Unity Mesh 로 주입 (Unity 임포터 축변환 회피).
        // (tissue_box_hl2c.obj = 각티슈 큐브(23.5x12x11.5cm), 원점=bbox 중심, 단위=미터. 예전엔 joke_book_hl2c
        //  wireframe 을 오버레이했으나 "흩뿌려짐" 으로 보여서 full mesh 를 solid 로 그리는 쪽으로 바꿨다.)
        yield return BuildSolidMesh("srt3d/tissue_box_hl2c.obj");
#if UNITY_EDITOR
        Hud("Editor: native skipped");
#else
        yield return InitFlow();
#endif
    }

#if !UNITY_EDITOR
    // 1~4단계 시퀀스
    IEnumerator InitFlow()
    {
        // 1~2: FP 초기 pose 획득 (성공까지 재시도). 이때 hl2ss(PV+depth) 가 PC에 스트림 중.
        switch (_initMode)
        {
            case InitMode.DragDraw:  yield return DragDrawRegister(); break;      // 핀치 드래그로 박스 그림 → /init_box
            case InitMode.CenterBox: yield return CenterBoxRegister(); break;    // 가운데 박스 → /init_box
            case InitMode.CropPinch: yield return SelectBoxThenRegister(); break; // 두 모서리 pinch → /init_box
            default:                 yield return TextInitLoop(); break;          // text('book') → /init
        }

        // 3: PV 캡처 시작 → 추적.
        //    ★ hl2ss.Initialize(false) 로 끄면 native crash → Stop() 호출 안 함.
        //    대신 PC가 grab 끝낼 때 stop_subsystem_pv 로 PV 를 이미 반납했으니 그대로 시도.
        Hud("FP pose 받음 — 카메라 전환 중...");
        yield return new WaitForSeconds(2.0f);
        yield return StartMfrCapture();
    }

    // [2단계] MediaFrameReader PV 캡처 open (ExclusiveControl — 검증됨). Unity world 좌표계 주입.
    // 이후 프레임 소비는 Update 에서 (메인 스레드 srt3d 호출).
    IEnumerator StartMfrCapture()
    {
        _pvCap = new Srt3dPvCapture();
#if ENABLE_WINMD_SUPPORT
        // Unity world 좌표계 (hl2ss 가 쓰는 것과 동일 API) → cam2world 계산용.
        var scs = Microsoft.MixedReality.OpenXR.PerceptionInterop.GetSceneCoordinateSystem(Pose.identity)
                  as Windows.Perception.Spatial.SpatialCoordinateSystem;
        if (scs != null) _pvCap.SetWorldCoordinateSystem(scs);
        else Debug.LogWarning("[Srt3dTracker] scs null — cam2world 불가");

        Hud("PV open (ExclusiveControl)...");
        var t = _pvCap.TryOpenAsync(Windows.Media.Capture.MediaCaptureSharingMode.ExclusiveControl);
        while (!t.IsCompleted) { Hud("PV open\n" + _pvCap.Status); yield return null; }
        if (!t.Result) { Hud("PV open FAIL\n" + _pvCap.Status); yield break; }
        _mfrRunning = true;
        Hud($"PV OPEN {_pvCap.Width}x{_pvCap.Height}\n{_pvCap.Status}\nintr {_pvCap.DiagIntr}\ntracking...");
#else
        Hud("MFR: WinRT 아님(에디터)");
        yield return null;
#endif
    }

    // [DIAG-B] srt3d 에 넘기는 rgb 를 PNG 로 저장(persistentDataPath). Device Portal 파일탐색기로 확인.
    // 채널 순서(RGB/BGR)·상하 flip·내용을 눈으로 → conf=0 원인 규명. (Unity 텍스처라 상하 반전 가능)
    void DumpFrame(byte[] rgb, int w, int h)
    {
        _dumpedFrame = true;
        try
        {
            // ⚠️ Unity Texture2D 의 raw 데이터는 bottom-up(행 0 = 아래)인데 우리 rgb 버퍼는
            //    top-down(행 0 = 위)이다. 그냥 넣으면 PNG 가 상하 반전돼 저장된다.
            //    이 아티팩트 때문에 예전에 "좌우 미러"로 오독해 FlipH 를 켰다가 추적이 반대로
            //    가는 버그를 만들었다(Srt3dPvCapture.cs 상단 주석 참조). 행을 뒤집어 넣는다.
            byte[] flipped = new byte[rgb.Length];
            int stride = w * 3;
            for (int y = 0; y < h; y++)
                System.Array.Copy(rgb, y * stride, flipped, (h - 1 - y) * stride, stride);

            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            tex.LoadRawTextureData(flipped); tex.Apply();
            byte[] png = tex.EncodeToPNG(); Destroy(tex);
            string p = Path.Combine(Application.persistentDataPath, "mfr_frame0.png");
            File.WriteAllBytes(p, png);
            Debug.Log($"[Srt3dTracker][DIAG] dumped {p}  {w}x{h}  bytes={png.Length}");
            Hud($"frame0 PNG 저장\n{w}x{h}\n{p}");
        }
        catch (System.Exception e) { Debug.Log("[Srt3dTracker] dump fail: " + e.Message); }
    }

    // MFR 프레임 1장 → srt3d (K=실제 intrinsics, cam2world=frame pose). 메인 스레드에서 호출.
    void ProcessMfrFrame(byte[] rgb, int w, int h, float fx, float fy, float cx, float cy,
                         Matrix4x4 c2w, bool hasK, bool hasPose)
    {
        _lastHasPose = hasPose;
        if (!hasK) { if (_frame % 30 == 0) Hud("intrinsics NULL — K 없음\n(다시 추측 경로 필요)"); return; }
        // gaze 투영용으로 최신 K/pose 보관 (등록 전에도 gaze HUD 가 떠야 해서 여기서 채움).
        _lastFx = fx; _lastFy = fy; _lastCx = cx; _lastCy = cy;
        if (hasPose) { _lastC2W = c2w; _hasLastC2W = true; }
        _mfrW = w; _mfrH = h;
        if (!_dumpedFrame && rgb != null && w > 0) DumpFrame(rgb, w, h);   // [DIAG-B] 첫 프레임 PNG
        if (!_inited)
        {
            float[] K = { fx, 0, cx, 0, fy, cy, 0, 0, 1 };
            _kInfo = $"K {w}x{h} fx={fx:F0} fy={fy:F0} cx={cx:F0} cy={cy:F0}";
            Debug.Log("[Srt3dTracker][DIAG] MFR intrinsics " + _kInfo + $"  hasPose={hasPose}");
            if (Srt3dNative.srt3d_init(_meshPath, K, w, h) == 0) { Hud("srt3d_init FAIL\n" + Srt3dNative.LastError()); return; }
            Srt3dNative.srt3d_reset_pose(_fpPose);
            _inited = true; _tracking = true; _target.gameObject.SetActive(true);
            Vector3 t0 = new Vector3(_fpPose[3], _fpPose[7], _fpPose[11]);
            // [DIAG-1] 초기 ob_in_cam Z (등록에서 온 값). PhotoCapture 에서 잘 됐으니 양수여야 정상.
            Debug.Log($"[Srt3dTracker][DIAG] MFR reset ob_in_cam t=({t0.x:F3},{t0.y:F3},{t0.z:F3})");
            _obZ = t0.z;
            Hud($"init OK (MFR) {w}x{h}\nobj t=({t0.x:F2},{t0.y:F2},{t0.z:F2})");
            return;
        }
        float[] outv = new float[17];
        if (Srt3dNative.srt3d_track_rgb(rgb, w, h, outv) == 0) { if (_frame % 30 == 0) Hud("track FAIL\n" + Srt3dNative.LastError()); return; }
        _frame++; _conf = outv[16]; _obZ = outv[11]; ComputePoseDelta(outv);
        // [DIAG] 객체중심(ob_in_cam translation)을 K로 이미지에 투영. 이미지 안(0~w,0~h)의 책 근처면
        // K/pose 정합 → conf 문제는 다른 것. 벗어나면 투영이 어긋난 것 = conf=0 원인.
        _projU = outv[11] != 0f ? fx * (outv[3] / outv[11]) + cx : -1f;
        _projV = outv[11] != 0f ? fy * (outv[7] / outv[11]) + cy : -1f;
        // 렌더는 B 하나로 고정. A/D 는 HUD 병기용 진단값 — 자동 선택(auto-pick) 아님.
        // cam2world 없으면(비 locatable) 렌더 스킵 — identity 로 그리면 원점에 뜸(금지).
        if (hasPose)
        {
            // Track with FoundationPose's raw contour solution, then preserve
            // the depth support-plane correction in the visible pose.
            Matrix4x4 M = PoseToMatrix(outv) * _renderCorrection;
            _candA = c2w * (C_CV2U * M);                // 앞쪽 C 만
            _candB = c2w * (C_CV2U * M * C_CV2U);       // 양쪽 conjugation
            _candD = c2w * (S_LEGACY * M);              // 현재(구 PhotoCapture 전제) — 대조군
            _pendWorld = _candB;
            _hasPend = true;
        }
    }

    // 구 방식: SAM3 text='book' 자동검출 (fallback). 성공까지 재시도.
    IEnumerator TextInitLoop()
    {
        while (_fpPose == null)
        {
            Hud("FP 초기 pose 요청 중...\n(책을 시야 중앙에 두세요)");
            using (var req = UnityWebRequest.Get(InitUrl))
            {
                req.timeout = 130;
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                {
                    var j = JsonUtility.FromJson<InitResp>(req.downloadHandler.text);
                    if (AcceptInitResponse(j)) { }
                    else
                        Hud("FP 응답 오류\n" + req.downloadHandler.text);
                }
                else
                    Hud("FP 요청 실패\n" + req.error + "\n(서버 3개 확인) 5s 재시도");
            }
            if (_fpPose == null) yield return new WaitForSeconds(5f);
        }
    }

    // 크롭 등록: 두 모서리 air-tap 으로 box 지정 → POST /init_box. 실패 시 재선택.
    IEnumerator SelectBoxThenRegister()
    {
        while (_fpPose == null)
        {
            yield return CollectTwoCorners();               // _boxCxcywh 채워짐
            Hud("등록 중...\n(머리를 움직이지 마세요)");
            yield return PostBox(_boxCxcywh);
            if (_fpPose == null)
            {
                Hud("등록 실패 — 다시 두 모서리를 pinch 하세요\n(서버 확인)");
                yield return new WaitForSeconds(2f);
            }
        }
    }

    // gaze ray → 헤드 카메라 뷰포트 → 이미지 정규화(top-left). 등록 중엔 MFR/PV K 가 없어
    // PV 픽셀 투영을 못 하므로, 헤드 카메라 뷰포트 근사를 쓴다(box+text 라 이 오차는 관대).
    bool TryGazeToImageNorm(out Vector2 img)
    {
        img = new Vector2(0.5f, 0.5f);
        Camera cam = _cam != null ? _cam : Camera.main;
        if (cam == null) return false;
        if (!TryGetXrGazeRay(out Vector3 oW, out Vector3 dW, out _)) return false;
        Vector3 vp = cam.WorldToViewportPoint(oW + dW * 1.0f);
        if (vp.z <= 0f) return false;
        img = new Vector2(Mathf.Clamp01(vp.x), Mathf.Clamp01(1f - vp.y));  // viewport(y↑) → 이미지(y↓)
        return true;
    }

    // 세로 박스 등록: gaze 로 박스를 물체에 얹고 dwell(1s)로 확정. gaze 없으면 머리 중앙 + 핀치.
    //   박스 크기는 프리셋(세로) 순환 — gaze 모드에선 핀치가 크기 순환, 확정은 dwell.
    //   fallback(머리 조준)에선 핀치가 확정. 어느 쪽이든 POST /init_box (box+text='book').
    IEnumerator CenterBoxRegister()
    {
        // ── 시작 게이트: 앱 시작 후 5초 자동 시작(핀치 감지 불안정 → 시간 기반). 핀치로 조기 시작 가능.
        //    한 번만. 등록 실패 재시도는 곧바로 조준으로 돌아간다.
        _selecting = false; _centerBoxMode = false; _wasPinch = false;
        float t = 5f;
        while (t > 0f)
        {
            if (PinchRising()) break;   // 핀치 되면 즉시 시작
            Hud($"등록 {Mathf.CeilToInt(t)}초 후 자동 시작\n(물체를 시야 중앙에 두세요)");
            t -= Time.unscaledDeltaTime;
            yield return null;
        }

        while (_fpPose == null)
        {
            _boxReady = false; _wasPinch = false; _dwellT = 0f;
            _selecting = true; _centerBoxMode = true;   // LateUpdate 가 박스 그림

            while (!_boxReady)
            {
                // 박스: 시야 기준 정사각형. 정규화 h = w × cam.aspect 라야 화면상 정사각(프레임 비율 보정).
                Camera cam = _cam != null ? _cam : Camera.main;
                _centerBoxW = _boxBaseW;
                _centerBoxH = Mathf.Clamp01(_boxBaseW * (cam != null ? cam.aspect : 1f));

                // 박스 위치: gaze 있으면 gaze, 없으면 화면 중앙.
                _gazeForBox = TryGazeToImageNorm(out Vector2 gz);
                _boxCenterNorm = _gazeForBox ? gz : new Vector2(0.5f, 0.5f);

                bool pinchRise = PinchRising();

                if (_gazeForBox)
                {
                    // dwell 만으로 확정 (핀치 안 씀 → 크기 안 바뀜).
                    if (_dwellT <= 0f) _dwellAnchor = _boxCenterNorm;
                    if ((_boxCenterNorm - _dwellAnchor).magnitude <= _dwellRadiusNorm)
                        _dwellT += Time.unscaledDeltaTime;
                    else { _dwellT = 0f; _dwellAnchor = _boxCenterNorm; }

                    if (_dwellT >= _dwellSec) _boxReady = true;   // dwell 확정
                    int pct = Mathf.RoundToInt(Mathf.Clamp01(_dwellT / _dwellSec) * 100f);
                    _regStatus = $"물체를 박스에 두고 응시  {pct}%";
                }
                else
                {
                    // 폴백: 머리로 조준, 핀치로 확정.
                    if (pinchRise) _boxReady = true;
                    _regStatus = "물체를 박스에 맞추고 핀치\n(시선 추적 대기 중)";
                }
                Hud(_regStatus);
                yield return null;
            }
            _selecting = false; _centerBoxMode = false;
            Hud("등록 중… 머리를 움직이지 마세요");
            yield return PostBox(new float[] { _boxCenterNorm.x, _boxCenterNorm.y, _centerBoxW, _centerBoxH });
            if (_fpPose == null)
            {
                Hud("등록 실패 — 다시 시도");
                yield return new WaitForSeconds(1.5f);
            }
        }
        Hud("등록 완료");
    }

    // 핀치 상승엣지 (히스테리시스). 크기 순환/확정 공용.
    bool PinchRising()
    {
        bool p = IsPinching(out _pinchDist);
        bool rise = p && !_wasPinch;
        _wasPinch = p;
        return rise;
    }

    // 두 번 탭(핀치) 등록: 한 모서리에서 핀치(탭) → 반대 모서리에서 핀치(탭) → 확정 → POST /init_box.
    // 잡고 있을 필요 없음(hold/release 감지 불안정 회피). 두 탭 사이엔 손끝을 따라 박스가 미리보기됨.
    IEnumerator DragDrawRegister()
    {
        while (_fpPose == null)
        {
            _boxReady = false; _wasPinch = false; _corners.Clear();
            _selecting = true; _drawMode = true;             // LateUpdate 가 박스 미리보기 그림
            while (!_boxReady)
            {
                UpdateTwoTap();   // 핀치 상승엣지로 모서리 2개 수집 → 확정 시 _boxReady
                Hud("박스 그리기 (핀치=탭)\n" +
                    (_corners.Count == 0 ? "① 한 모서리에서 핀치" : "② 반대 모서리에서 핀치") +
                    $"\n[d={(_pinchDist < 0f ? "n/a(손을 시야에)" : _pinchDist.ToString("F3"))}]");
                yield return null;
            }
            _selecting = false; _drawMode = false;
            Hud($"box cx={_boxCxcywh[0]:F2} cy={_boxCxcywh[1]:F2} w={_boxCxcywh[2]:F2} h={_boxCxcywh[3]:F2}\n등록 중...(머리 고정)");
            yield return PostBox(_boxCxcywh);
            if (_fpPose == null)
            {
                Hud("등록 실패 — 다시 그리세요\n(서버 확인)");
                yield return new WaitForSeconds(2f);
            }
        }
    }

    // 핀치 상승엣지(탭)로 모서리 수집. 0개→탭=모서리1, 1개→탭=모서리2(확정). 미리보기용 _dragA/_dragB 갱신.
    void UpdateTwoTap()
    {
        bool pinch = IsPinching(out _pinchDist);
        bool rising = pinch && !_wasPinch;
        bool haveTip = TryIndexTipImageNorm(out Vector2 cur);
        if (!haveTip) cur = new Vector2(0.5f, 0.5f);

        if (rising && haveTip)
        {
            _pinchCount++;
            if (_corners.Count == 0) _corners.Add(cur);                 // 모서리1
            else if (_corners.Count == 1)                              // 모서리2 → 확정(최소 크기 통과 시)
            {
                var box = CornersToBox(_corners[0], cur);
                if (box[2] >= 0.04f && box[3] >= 0.04f) { _corners.Add(cur); _boxCxcywh = box; _boxReady = true; }
                // 너무 작으면 무시(다시 ② 시도)
            }
        }
        // 미리보기 두 점: 모서리1 전엔 손끝 커서, 후엔 모서리1↔현재 손끝.
        if (_corners.Count == 0) { _dragA = cur - new Vector2(0.02f, 0.02f); _dragB = cur + new Vector2(0.02f, 0.02f); }
        else { _dragA = _corners[0]; _dragB = cur; }
        _wasPinch = pinch;
    }
#endif

    void Update()
    {
        // MFR: 백그라운드가 채운 최신 프레임 소비 → srt3d (메인 스레드).
#if ENABLE_WINMD_SUPPORT
        if (_mfrRunning && _pvCap != null &&
            _pvCap.TryGetLatest(ref _mfrRgb, out int mw, out int mh, out float mfx, out float mfy,
                                out float mcx, out float mcy, out Matrix4x4 mc2w, out bool mHasK, out bool mHasPose))
            ProcessMfrFrame(_mfrRgb, mw, mh, mfx, mfy, mcx, mcy, mc2w, mHasK, mHasPose);
#endif

        // [gaze] 추적 중엔 핀치가 비어 있으므로 가정 깊이 순환에 쓴다(등록 중엔 확정용이라 제외).
        if (_tracking && !_selecting)
        {
            bool gz = IsPinching(out _pinchDist);
            if (gz && !_wasPinch) _gazeZIdx = (_gazeZIdx + 1) % GAZE_Z_OPTIONS.Length;
            _wasPinch = gz;
        }
        UpdateGazeDiag();   // [gaze 1단계] 등록/추적 어느 단계든 항상 갱신

        _fpsAccum += Time.unscaledDeltaTime;
        if (_fpsAccum >= 0.5f) { _fps = (_frame - _fpsLastFrame) / _fpsAccum; _fpsLastFrame = _frame; _fpsAccum = 0f; }
        if (_tracking) _trackElapsed += Time.unscaledDeltaTime;

        if (_hasPend && _tracking && _target != null)
        {
            _hasPend = false;
            ApplyPose(_pendWorld);
            _objWorld = _pendWorld.MultiplyPoint3x4(Vector3.zero);
            if (_cam != null) _headWorld = _cam.transform.position;
            if (_frame % 5 == 0)
            {
                float avg = _trackElapsed > 0.5f ? _frame / _trackElapsed : 0f;
                if (!_showDebug)
                {
                    // 운용 중: 한 줄. 이게 전부.
                    Hud($"conf={_conf:F2}  fps={avg:F0}");
                }
                else
                {
                    // 디버그 (음성 "toggle debug"): 전체 진단.
                    string diag = "?";
#if ENABLE_WINMD_SUPPORT
                    if (_pvCap != null) diag = $"fmt={_pvCap.DiagFmt} nz={_pvCap.NonzeroPct:F0}% " +
                                               $"flip={(_pvCap.FlipH ? "H" : "-")}{(_pvCap.FlipV ? "V" : "-")}";
#endif
                    Hud($"conf={_conf:F2} obZ={_obZ:F2} fps={avg:F0} f={_frame}\n" +
                        $"proj({_projU:F0},{_projV:F0}) img {_mfrW}x{_mfrH}\n" +
                        $"A{Oih(_candA)}\nB{Oih(_candB)} <-render\nD{Oih(_candD)}\n" +
                        $"{_gazeHud}\n{diag}");
                }
            }
        }
    }

    // ── [gaze 2단계] OpenXR 직접 읽기 → _gazeHud / _gazePx (MRTK 는 대조용 병기) ─────
    // MRTK 경로가 cal=? 에서 멈춤 = 보정 상태조차 못 읽음 = eye gaze data provider 미등록.
    // 그래서 주 경로를 UnityEngine.XR.InputDevices 로 바꾼다. 투영(고정 깊이 평면) 로직은
    // 그대로 재사용하고 origin/direction 출처만 교체.
    //
    // HUD 표기:
    //   gaze(451,239) z=0.40 IN dev=1 fix=Y pos=Y   정상
    //   gaze INVALID dev=0                          OpenXR 레벨에 eye tracking 디바이스 없음
    //                                               → capability(GazeInput) / Eye Gaze Interaction
    //                                                 Profile / 기기 동의 팝업 거부 쪽 원인
    //   gaze INVALID dev=1 fix=N                    디바이스는 있는데 fixation point 무효
    //                                               → 눈 보정 미실행 또는 일시적 추적 손실
    //   gaze noK                                    아직 PV 프레임/K 가 없음 (캡처 시작 전)
    void UpdateGazeDiag()
    {
        _gazePxValid = false;
        UpdateMrtkGazeDiag();   // 대조용 — 실패해도 아래 OpenXR 경로엔 영향 없음

        Vector3 oW, dW; string tag;
        if (!TryGetXrGazeRay(out oW, out dW, out tag)) { _gazeHud = $"gaze INVALID {tag}"; return; }

        float z0 = GazeZ;
        if (!_hasLastC2W) { _gazeHud = $"gaze noK (PV 프레임 대기) z={z0:F2} {tag}"; return; }

        // 시선 ray(world) → PV 카메라(Unity 규약) → OpenCV 규약(C = diag(1,-1,1)).
        Matrix4x4 w2c = _lastC2W.inverse;
        Vector3 oU = w2c.MultiplyPoint3x4(oW);
        Vector3 dU = w2c.MultiplyVector(dW);
        Vector3 o = new Vector3(oU.x, -oU.y, oU.z);
        Vector3 d = new Vector3(dU.x, -dU.y, dU.z);

        // z = z0 평면과 교차. o.z != 0 이라 눈-PV 베이스라인이 자동 반영된다.
        if (d.z <= 1e-4f) { _gazeHud = $"gaze BACKWARD z={z0:F2} {tag}"; return; }
        float t = (z0 - o.z) / d.z;
        if (t <= 0f) { _gazeHud = $"gaze BEHIND z={z0:F2} {tag}"; return; }
        Vector3 P = o + t * d;
        _gazePx = new Vector2(_lastFx * P.x / P.z + _lastCx, _lastFy * P.y / P.z + _lastCy);
        _gazePxValid = true;

        bool inImg = _gazePx.x >= 0 && _gazePx.x < _mfrW && _gazePx.y >= 0 && _gazePx.y < _mfrH;
        _gazeHud = $"gaze({_gazePx.x:F0},{_gazePx.y:F0}) z={z0:F2}(핀치로변경) " +
                   $"{(inImg ? "IN" : "OUT")} {tag}";
    }

    // OpenXR(UnityEngine.XR) 에서 눈 ray 를 world 좌표로 뽑는다.
    //
    // [좌표계] eyesData 가 주는 값은 world 도 camera-local 도 아니라 XR tracking(playspace)
    // 공간이다 — head node pose 와 같은 공간. Unity 는 이걸 카메라의 부모 트랜스폼
    // (MRTK 에선 MixedRealityPlayspace) 에 걸어 world 로 올린다. 그래서 world 변환은
    // Camera.main.transform 이 아니라 그 '부모' 행렬로 해야 한다. camera-local 로 변환하면
    // 머리 회전이 두 번 먹어서 고개를 돌릴 때만 틀어지는 성가신 오차가 난다.
    // playspace 가 identity 면 raw 값이 곧 world 라 차이가 안 보이지만, teleport/재보정으로
    // 틀어지는 순간 갈린다. 부모가 없으면 raw 를 그대로 world 로 본다(= identity 가정).
    // 검증용으로 HUD 에 d2cam(=원점~카메라 거리) 을 찍는다. 정상이면 눈~PV 베이스라인 규모인
    // 0.0x m 여야 한다. 이게 카메라 world 위치 크기(방 규모)로 나오면 변환이 틀린 것.
    bool TryGetXrGazeRay(out Vector3 originWorld, out Vector3 dirWorld, out string tag)
    {
        originWorld = Vector3.zero; dirWorld = Vector3.forward;

        _xrEyeDevs.Clear();
        UnityEngine.XR.InputDevices.GetDevicesWithCharacteristics(
            UnityEngine.XR.InputDeviceCharacteristics.EyeTracking, _xrEyeDevs);
        int nDev = _xrEyeDevs.Count;
        if (nDev == 0) { tag = "dev=0"; return false; }
        var dev = _xrEyeDevs[0];

        Vector3 oT, dT;

        // [A-4] 주 경로: OpenXR eye gaze pose (devicePosition/deviceRotation).
        //   MRTK/OpenXR provider 가 Windows-MR 식 Eyes 구조체(eyesData)를 안 채워서 eyesData=N 이었다.
        //   OpenXR 표준 eye gaze interaction 은 gaze 를 '디바이스 pose'(pos+rot)로 노출한다 →
        //   forward = rot * +Z 가 시선 방향. 이게 우리가 쓸 값.
        Vector3 gpos; Quaternion grot;
        bool okPos = dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.devicePosition, out gpos);
        bool okRot = dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.deviceRotation, out grot);
        if (okPos && okRot && grot.normalized != new Quaternion(0, 0, 0, 0))
        {
            oT = gpos;
            dT = grot * Vector3.forward;
        }
        else
        {
            // 폴백: Windows-MR 식 Eyes 구조체 (구형 provider).
            UnityEngine.XR.Eyes eyes;
            if (!dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.eyesData, out eyes))
            { tag = $"dev={nDev} pose=N eyesData=N"; return false; }
            Vector3 fix, lPos, rPos;
            bool okFix = eyes.TryGetFixationPoint(out fix);
            bool okL = eyes.TryGetLeftEyePosition(out lPos);
            bool okR = eyes.TryGetRightEyePosition(out rPos);
            if (!okFix || !okL || !okR)
            {
                tag = $"dev={nDev} pose=N fix={(okFix ? "Y" : "N")} pos={((okL && okR) ? "Y" : "N")}";
                return false;
            }
            oT = (lPos + rPos) * 0.5f;
            dT = fix - oT;
        }
        if (dT.sqrMagnitude < 1e-8f) { tag = $"dev={nDev} deg(dir=0)"; return false; }

        // tracking(playspace) → world
        Camera cam = _cam != null ? _cam : (Camera.main != null ? Camera.main : FindObjectOfType<Camera>());
        Transform playspace = (cam != null && cam.transform.parent != null) ? cam.transform.parent : null;
        if (playspace != null)
        {
            originWorld = playspace.TransformPoint(oT);
            dirWorld = playspace.TransformDirection(dT).normalized;
        }
        else { originWorld = oT; dirWorld = dT.normalized; }

        float d2cam = cam != null ? Vector3.Distance(originWorld, cam.transform.position) : -1f;
        tag = $"dev={nDev} fix=Y pos=Y d2cam={d2cam:F2}";
        return true;
    }

    // [대조] 기존 MRTK 경로. 지우지 않고 HUD 에 같이 띄워 둘 중 뭐가 살아나는지 본다.
    // en=False  → capability(GazeInput) 또는 OpenXR Eye Gaze Interaction Profile 누락
    // valid=False → 기기 아이 캘리브레이션 미실행, 또는 동의 팝업 거부
    // cal=?     → 보정 상태조차 못 읽음 = eye gaze data provider 미등록(현 증상)
    void UpdateMrtkGazeDiag()
    {
        var inputSys = CoreServices.InputSystem;
        if (inputSys == null) { _gazeMrtkHud = "mrtk: InputSystem null"; return; }
        var eg = inputSys.EyeGazeProvider;
        if (eg == null) { _gazeMrtkHud = "mrtk: EyeGazeProvider null"; return; }

        // 활성 프로필이 패키지 내장 read-only 라 isEyeTrackingEnabled=0 으로 굳어 있음
        // (DefaultMixedRealityInputPointerProfile.asset:32). 프로필 트리 복제 대신 런타임에 켠다.
        if (!_gazeEnableTried) { _gazeEnableTried = true; eg.IsEyeTrackingEnabled = true; }

        string cal = eg.IsEyeCalibrationValid.HasValue
            ? (eg.IsEyeCalibrationValid.Value ? "cal=OK" : "cal=NO")
            : "cal=?";
        if (!eg.IsEyeTrackingEnabledAndValid)
        { _gazeMrtkHud = $"mrtk INVALID en={eg.IsEyeTrackingEnabled} valid={eg.IsEyeTrackingDataValid} {cal}"; return; }

        Vector3 o = eg.GazeOrigin, d = eg.GazeDirection;
        _gazeMrtkHud = $"mrtk OK {cal} o=({o.x:F2},{o.y:F2},{o.z:F2}) d=({d.x:F2},{d.y:F2},{d.z:F2})";
    }

    [System.Serializable] class InitResp
    {
        public bool ok;
        public float[] pose;
        public float[] render_correction;
    }

    bool AcceptInitResponse(InitResp response)
    {
        if (response == null || !response.ok || response.pose == null || response.pose.Length < 16)
            return false;
        _fpPose = response.pose;
        _renderCorrection = response.render_correction != null && response.render_correction.Length >= 16
            ? PoseToMatrix(response.render_correction)
            : Matrix4x4.identity;
        return true;
    }

    // [DIAG] ob_in_cam(row-major 16) 의 프레임간 변화량. 카메라/객체가 움직이는데 Δ≈0 이면
    // srt3d 가 pose 를 갱신 못 하는 것(=내부 K 로 투영 실패 가능성) → 근본 원인 확정용.
    void ComputePoseDelta(float[] cur)
    {
        if (_prevPose != null)
        {
            // translation Δ (m)
            float dx = cur[3] - _prevPose[3], dy = cur[7] - _prevPose[7], dz = cur[11] - _prevPose[11];
            _dT = Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
            // rotation Δ (도): trace(R_prevᵀ·R_cur) = 두 회전행렬의 Frobenius 내적(원소별 곱의 합).
            // 회전각 A 는 cosA = (trace - 1) / 2.
            int[] ri = { 0, 1, 2, 4, 5, 6, 8, 9, 10 };
            float tr = 0f;
            for (int i = 0; i < 9; i++) tr += _prevPose[ri[i]] * cur[ri[i]];
            float cosA = Mathf.Clamp((tr - 1f) * 0.5f, -1f, 1f);
            _dR = Mathf.Acos(cosA) * Mathf.Rad2Deg;
        }
        if (_prevPose == null) _prevPose = new float[16];
        System.Array.Copy(cur, _prevPose, 16);
    }

    // ── 두 모서리 pinch box 선택 (MRTK air-tap) ─────────────────────────────
    // 흐름: RegisterPointer → _selecting=true → 사용자가 두 모서리를 air-tap →
    //       OnPointerClicked 가 이미지 정규화 좌표로 모아 2개 되면 _boxCxcywh 확정.
    // 좌표: MRTK 포인터 초점(레이 끝점) → 헤드 카메라 viewport → 이미지 정규화(원점 top-left).
    //       hl2ss grab 프레임(≈헤드 시야)과 대략 정합 — SAM box 프롬프트는 소폭 오차에 관대.

    IEnumerator CollectTwoCorners()
    {
        _corners.Clear(); _boxReady = false;
        RegisterPointer();
        _selecting = true;
        Hud("크롭 등록\n① 객체 왼쪽위 모서리를 air-tap");
        // InputSystem 이 늦게 뜨는 경우 대비, 등록 전까지 매 프레임 재시도.
        while (!_boxReady) { if (!_handlerRegistered) RegisterPointer(); yield return null; }
        _selecting = false;
        UnregisterPointer();
    }

    public void OnPointerClicked(MixedRealityPointerEventData e)
    {
        _clickCount++;                                      // [DIAG] MRTK 전역 클릭이 오는지 카운트
        if (!_selecting || _boxReady) return;
        if (_centerBoxMode) { _boxReady = true; return; }   // 가운데 박스: 탭 1회로 확정
        if (!TryPointerToImageNorm(e, out Vector2 p)) { Hud("모서리 인식 실패 — 다시 air-tap"); return; }
        _corners.Add(p);
        if (_corners.Count == 1) Hud($"① 기록 ({p.x:F2},{p.y:F2})\n② 반대쪽(오른아래) 모서리를 air-tap");
        else if (_corners.Count >= 2)
        {
            _boxCxcywh = CornersToBox(_corners[0], _corners[1]);
            _boxReady = true;
            Hud($"box cx={_boxCxcywh[0]:F2} cy={_boxCxcywh[1]:F2} w={_boxCxcywh[2]:F2} h={_boxCxcywh[3]:F2}");
        }
    }
    public void OnPointerDown(MixedRealityPointerEventData e) { }
    public void OnPointerDragged(MixedRealityPointerEventData e) { }
    public void OnPointerUp(MixedRealityPointerEventData e) { }

    // 손관절(엄지끝↔검지끝) 거리로 핀치 직접 감지 — MRTK 의 'Select' 분류/전역 클릭에 의존 안 함.
    // 히스테리시스(on<off)로 채터링 방지. 양손 중 더 가까운 값 사용. 손 미추적 시 dist=-1.
    bool IsPinching(out float dist)
    {
        dist = -1f;
        foreach (var h in new[] { Handedness.Right, Handedness.Left })
        {
            if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexTip, h, out MixedRealityPose ip) &&
                HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbTip, h, out MixedRealityPose tp))
            {
                float d = Vector3.Distance(ip.Position, tp.Position);
                if (dist < 0f || d < dist) dist = d;
            }
        }
        if (dist < 0f) return false;
        bool prevInside = _wasPinch;
        return prevInside ? (dist < _pinchOff) : (dist < _pinchOn);   // 히스테리시스
    }

    // 검지끝 world 위치 → 헤드 카메라 이미지 정규화 좌표(top-left). 손 미추적 시 false.
    bool TryIndexTipImageNorm(out Vector2 img)
    {
        img = default;
        Camera cam = _cam != null ? _cam : (Camera.main != null ? Camera.main : FindObjectOfType<Camera>());
        if (cam == null) return false;
        foreach (var h in new[] { Handedness.Right, Handedness.Left })
            if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexTip, h, out MixedRealityPose ip))
            {
                Vector3 vp = cam.WorldToViewportPoint(ip.Position);
                if (vp.z <= 0f) return false;
                img = new Vector2(Mathf.Clamp01(vp.x), Mathf.Clamp01(1f - vp.y));
                return true;
            }
        return false;
    }

    // 매 프레임(선택 중) 핀치 상승엣지 → 확정 탭. CollectTwoCorners/CenterBoxRegister 루프에서 호출.
    void PollPinch()
    {
        bool pinch = IsPinching(out _pinchDist);
        if (pinch && !_wasPinch) { _pinchCount++; OnConfirmTap(); }  // 상승엣지 1회
        _wasPinch = pinch;
    }

    // 확정 신호(핀치 폴백 경로). 가운데 박스는 위치 무관이라 여기서 처리.
    void OnConfirmTap()
    {
        if (!_selecting || _boxReady) return;
        if (_centerBoxMode) _boxReady = true;
        // CropPinch(두 모서리)는 위치가 필요 → OnPointerClicked(레이) 경로 사용.
    }

    // 포인터 초점(콜라이더 없으면 MRTK 기본 거리의 레이 끝점) → 이미지 정규화 좌표(top-left 원점).
    bool TryPointerToImageNorm(MixedRealityPointerEventData e, out Vector2 img)
    {
        img = default;
        Camera cam = _cam != null ? _cam : (Camera.main != null ? Camera.main : FindObjectOfType<Camera>());
        if (cam == null || e == null || e.Pointer == null) return false;
        Vector3 world;
        var res = e.Pointer.Result;
        if (res != null && res.Details.Point != Vector3.zero) world = res.Details.Point;
        else world = e.Pointer.Position + (e.Pointer.Rotation * Vector3.forward) * 1.5f;  // 초점 없으면 1.5m 전방
        Vector3 vp = cam.WorldToViewportPoint(world);
        if (vp.z <= 0f) return false;   // 카메라 뒤
        // Unity viewport 원점 좌하(y↑) → 이미지 원점 좌상 : y 뒤집기.
        img = new Vector2(Mathf.Clamp01(vp.x), Mathf.Clamp01(1f - vp.y));
        return true;
    }

    // 두 모서리(이미지 정규화) → box [cx,cy,w,h] (정규화). 최소 크기 보장.
    float[] CornersToBox(Vector2 a, Vector2 b)
    {
        float x0 = Mathf.Min(a.x, b.x), x1 = Mathf.Max(a.x, b.x);
        float y0 = Mathf.Min(a.y, b.y), y1 = Mathf.Max(a.y, b.y);
        float w = Mathf.Clamp(x1 - x0, 0.03f, 1f), h = Mathf.Clamp(y1 - y0, 0.03f, 1f);
        float cx = Mathf.Clamp01(x0 + w * 0.5f), cy = Mathf.Clamp01(y0 + h * 0.5f);
        return new float[] { cx, cy, w, h };
    }

    void RegisterPointer()
    {
        if (_handlerRegistered) return;
        if (CoreServices.InputSystem != null)
        { CoreServices.InputSystem.RegisterHandler<IMixedRealityPointerHandler>(this); _handlerRegistered = true; }
        else Debug.LogWarning("[Srt3dTracker] InputSystem 없음 — air-tap 수신 불가(MRTK 미초기화?)");
    }
    void UnregisterPointer()
    {
        if (!_handlerRegistered) return;
        CoreServices.InputSystem?.UnregisterHandler<IMixedRealityPointerHandler>(this);
        _handlerRegistered = false;
    }

    // POST /init_box -> raw tracking pose + optional object-local render correction.
    IEnumerator PostBox(float[] box)
    {
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        string b = string.Format(ci, "[{0:0.####},{1:0.####},{2:0.####},{3:0.####}]",
                                 box[0], box[1], box[2], box[3]);
        string json = "{\"box\":" + b + "}";
        using (var req = new UnityWebRequest(InitBoxUrl, "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(json));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = 130;
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success)
            {
                var j = JsonUtility.FromJson<InitResp>(req.downloadHandler.text);
                if (AcceptInitResponse(j)) { }
                else Hud("등록 응답 오류\n" + req.downloadHandler.text);
            }
            else Hud("등록 요청 실패\n" + req.error);
        }
    }

    // [TUNE] HUD 색: 시안 기본. 흰색 시험하려면 Color.white 로.
    Color _hudColor = Color.cyan;
    TextMesh CreateHud()
    {
        var go = new GameObject("Srt3dHud");
        var tm = go.AddComponent<TextMesh>();
        tm.characterSize = 0.0075f; tm.fontSize = 120;
        tm.fontStyle = FontStyle.Bold;                    // OST 에선 얇은 획이 사라짐 → 굵게
        tm.anchor = TextAnchor.UpperCenter; tm.alignment = TextAlignment.Center; tm.color = _hudColor;
        return tm;
    }

    // GO 는 identity 유지. mesh 정점은 ApplyPose 에서 매 프레임 world 로 갱신(per-vertex).
    // RGB 축은 LineRenderer(useWorldSpace) 로 world 끝점 직접 → 어느 축이 어디로 갔는지 캡처로 읽음.
    Transform CreateTarget()
    {
        var tmp = GameObject.CreatePrimitive(PrimitiveType.Cube);  // Standard 셰이더 추출용(임시)
        _stdShader = tmp.GetComponent<Renderer>().sharedMaterial.shader;
        Destroy(tmp);

        var go = new GameObject("Srt3dTarget");
        var mf = go.AddComponent<MeshFilter>();
        var mr = go.AddComponent<MeshRenderer>();
        _mesh = new Mesh { name = "solidWorld" };
        mf.mesh = _mesh;
        var mat = new Material(_stdShader);
        // 반투명 wireframe → opaque solid 로 전환. AR 시스루라 emission 으로 형태가 잘 보이게.
        SetupMat(mat, new Color(0.55f, 0.75f, 0.95f, 1f), false);
        mr.material = mat;

        _axes = new LineRenderer[3];
        _axes[0] = MakeAxis(go.transform, new Color(1f, 0f, 0f)); // +X 빨강
        _axes[1] = MakeAxis(go.transform, new Color(0f, 1f, 0f)); // +Y 초록
        _axes[2] = MakeAxis(go.transform, new Color(0f, 0f, 1f)); // +Z 파랑

        go.SetActive(false);
        return go.transform;
    }

    LineRenderer MakeAxis(Transform parent, Color col)
    {
        var g = new GameObject("axis");
        g.transform.SetParent(parent, false);
        var lr = g.AddComponent<LineRenderer>();
        lr.useWorldSpace = true;        // world 끝점 직접 지정 (parent transform 무관)
        lr.positionCount = 2;
        lr.widthMultiplier = 0.004f;
        var mat = new Material(_stdShader);
        SetupMat(mat, col, false);
        lr.material = mat;
        lr.startColor = lr.endColor = col;
        return lr;
    }

    // Standard 셰이더(큐브에서 보장) + emission 으로 normal/조명 무관하게 색 가독. transparent 시 알파블렌드.
    static void SetupMat(Material m, Color col, bool transparent)
    {
        if (transparent)
        {
            m.SetFloat("_Mode", 3f);
            m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetInt("_ZWrite", 0);
            m.EnableKeyword("_ALPHABLEND_ON");
            m.renderQueue = 3000;
        }
        m.color = col;
        m.EnableKeyword("_EMISSION");
        m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
        m.SetColor("_EmissionColor", new Color(col.r, col.g, col.b) * 0.8f);
    }

    Camera _cam;
    void LateUpdate()
    {
        if (_hud == null) return;
        // Camera.main 이 null 일 수 있음(MRTK 씬 아님 → MainCamera 태그 없을 수 있음) → fallback.
        if (_cam == null) _cam = Camera.main != null ? Camera.main : FindObjectOfType<Camera>();
        if (_cam == null) return;
        var c = _cam.transform;
        // 시야 아래쪽에 배치하되 FOV 안(HL2 수직 반각 ~14.5°). 1.6m 앞 0.30m 아래 = ~11° → 안전.
        //   (이전 2m/0.75m = 20° 는 화면 밖이라 안 보였음.)
        // 회전은 카메라 up 을 써서 항상 똑바로 — LookRotation 기본 up 은 머리 기울일 때 기울어 보임.
        Vector3 target = c.position + c.forward * 1.6f - c.up * 0.30f;
        // 첫 프레임엔 원점(0,0,0)에서 Lerp 시작하면 뒤에 떴다가 날아옴 → 멀면 스냅.
        if (Vector3.Distance(_hud.transform.position, target) > 3f) _hud.transform.position = target;
        else _hud.transform.position = Vector3.Lerp(_hud.transform.position, target, 1f - Mathf.Exp(-10f * Time.unscaledDeltaTime));
        _hud.transform.rotation = Quaternion.LookRotation(_hud.transform.position - c.position, c.up);
        UpdateBoxVisual();
    }

    // 박스 시각화: 카메라 1m 앞 view plane 에 세로 사각형. 반투명 채움(quad) + 밝은 테두리(line).
    //   중심 = _boxCenterNorm (gaze 있으면 gaze, 없으면 0.5). 크기 = 현재 프리셋(세로).
    //   dwell 진행에 따라 테두리 색이 노랑→초록으로 차오름(발사 시점 예측 가능하게).
    void UpdateBoxVisual()
    {
        bool show = _selecting && (_centerBoxMode || _drawMode);
        if (_boxLr == null)
        {
            var g = new GameObject("Srt3dBox");
            _boxLr = g.AddComponent<LineRenderer>();
            _boxLr.useWorldSpace = true; _boxLr.loop = true;
            _boxLr.positionCount = 4; _boxLr.widthMultiplier = 0.008f;   // 굵은 테두리 (OST 가독)
            var mat = new Material(_stdShader); SetupMat(mat, Color.yellow, false);
            _boxLr.material = mat;

            // 채움 quad 복원 — 테두리만으론 안 보인다는 피드백. 흰 반투명(잘 보이던 원래 것).
            var fq = GameObject.CreatePrimitive(PrimitiveType.Quad);
            Destroy(fq.GetComponent<Collider>());
            _boxFill = fq.GetComponent<MeshRenderer>();
            var fmat = new Material(_stdShader);
            SetupMat(fmat, new Color(1f, 1f, 1f, 0.22f), true);   // 흰 반투명
            _boxFill.material = fmat;
        }
        _boxLr.enabled = show;
        if (_boxFill != null) _boxFill.enabled = show;
        if (!show || _cam == null) return;

        Vector2 c = (_centerBoxMode) ? _boxCenterNorm : (_dragA + _dragB) * 0.5f;
        float hw = _centerBoxW * 0.5f, hh = _centerBoxH * 0.5f;
        if (!_centerBoxMode) { hw = Mathf.Abs(_dragB.x - _dragA.x) * 0.5f; hh = Mathf.Abs(_dragB.y - _dragA.y) * 0.5f; }
        float ax = Mathf.Clamp01(c.x - hw), bx = Mathf.Clamp01(c.x + hw);
        float ay = Mathf.Clamp01(c.y - hh), by = Mathf.Clamp01(c.y + hh);
        const float d = 1.0f;
        Vector3 p00 = ImgNormToWorld(ax, ay, d), p10 = ImgNormToWorld(bx, ay, d);
        Vector3 p11 = ImgNormToWorld(bx, by, d), p01 = ImgNormToWorld(ax, by, d);
        _boxLr.SetPosition(0, p00); _boxLr.SetPosition(1, p10);
        _boxLr.SetPosition(2, p11); _boxLr.SetPosition(3, p01);

        // dwell 진행 → 테두리 색 (노랑→초록)
        float prog = _gazeForBox ? Mathf.Clamp01(_dwellT / _dwellSec) : 0f;
        Color edge = Color.Lerp(Color.yellow, Color.green, prog);
        _boxLr.startColor = _boxLr.endColor = edge;

        // 흰 채움 quad — 카메라 향해, 박스 크기로.
        if (_boxFill != null)
        {
            _boxFill.transform.position = (p00 + p11) * 0.5f;
            _boxFill.transform.rotation = Quaternion.LookRotation(_cam.transform.forward, _cam.transform.up);
            _boxFill.transform.localScale = new Vector3((p10 - p00).magnitude, (p01 - p00).magnitude, 1f);
        }
    }

    // 이미지 정규화(top-left 원점) → 카메라 view plane(거리 d) world 점. y 뒤집어 viewport 로.
    Vector3 ImgNormToWorld(float nx, float ny, float d)
    {
        return _cam.ViewportToWorldPoint(new Vector3(Mathf.Clamp01(nx), 1f - Mathf.Clamp01(ny), d));
    }

    void Hud(string s) { Debug.Log("[Srt3dTracker] " + s); if (_hud != null) _hud.text = s; }

    // srt3d ob_in_cam (row-major 16, OpenCV) → Matrix4x4. 좌표계 변환은 호출부에서 후보별로.
    static Matrix4x4 PoseToMatrix(float[] pose16)
    {
        Matrix4x4 M = new Matrix4x4();
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) M[r, c] = pose16[r * 4 + c];
        return M;
    }

    // [DIAG] 후보 world 행렬의 원점을 head-local 로 → "(x,y,z)" 문자열. Z>0 = 카메라 앞.
    string Oih(Matrix4x4 T)
    {
        if (_cam == null) return "(no cam)";
        Vector3 p = _cam.transform.InverseTransformPoint(T.MultiplyPoint3x4(Vector3.zero));
        return $"({p.x:F2},{p.y:F2},{p.z:F2})";
    }

    // StreamingAssets 의 .obj 를 받아 파싱 → _target MeshFilter 에 주입. (UWP 는 StreamingAssets 가
    // 패키지 안이라 File.IO 불가 → UnityWebRequest 로 읽고 텍스트만 파싱, 디스크 영속 불필요.)
    IEnumerator BuildSolidMesh(string rel)
    {
        string src = Application.streamingAssetsPath + "/" + rel;
        string uri = src.Contains("://") ? src : "file:///" + src.Replace("\\", "/");
        using (var req = UnityWebRequest.Get(uri))
        {
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success) { Hud("mesh obj 로드 실패\n" + req.error); yield break; }
            ParseObj(req.downloadHandler.text);   // _objVerts, _tris 채움
            _worldVerts = new Vector3[_objVerts.Length];
            _mesh.Clear();
            if (_objVerts.Length > 65000) _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            _mesh.SetVertices(_objVerts);          // 초기(object 공간; 첫 pose 에서 world 로 덮임)
            _mesh.SetTriangles(_tris, 0);
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();
            Debug.Log($"[Srt3dTracker] solid mesh: {_objVerts.Length}v {_tris.Length / 3}tri bounds={_mesh.bounds.size}");
        }
    }

    // .obj 좌표 그대로 → _objVerts/_tris. f a b c (1-indexed, v/vt/vn 슬래시 허용). 양면(역 winding 추가)
    // 으로 backface culling/normal 방향 무관하게 항상 보이게 → 좌표진단 변수 제거.
    void ParseObj(string text)
    {
        var verts = new List<Vector3>();
        var tris = new List<int>();
        var sep = new[] { ' ', '\t', '\r' };
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        foreach (var raw in text.Split('\n'))
        {
            if (raw.Length < 2) continue;
            char c0 = raw[0];
            if (c0 == 'v' && raw[1] == ' ')
            {
                var t = raw.Split(sep, System.StringSplitOptions.RemoveEmptyEntries);
                if (t.Length < 4) continue;
                verts.Add(new Vector3(float.Parse(t[1], ci), float.Parse(t[2], ci), float.Parse(t[3], ci)));
            }
            else if (c0 == 'f' && raw[1] == ' ')
            {
                var t = raw.Split(sep, System.StringSplitOptions.RemoveEmptyEntries);
                int n = t.Length - 1;
                if (n < 3) continue;
                int[] idx = new int[n];
                for (int k = 0; k < n; k++)
                {
                    string s = t[k + 1];
                    int sl = s.IndexOf('/');
                    if (sl >= 0) s = s.Substring(0, sl);
                    idx[k] = int.Parse(s, ci) - 1;
                }
                for (int k = 1; k + 1 < n; k++)   // fan triangulation, 양면
                {
                    tris.Add(idx[0]); tris.Add(idx[k]); tris.Add(idx[k + 1]);
                    tris.Add(idx[0]); tris.Add(idx[k + 1]); tris.Add(idx[k]);
                }
            }
        }
        _objVerts = verts.ToArray();
        _tris = tris.ToArray();
    }

    // object 정점/축을 full 4x4(T=cam2world·S·M)로 world 변환. MultiplyPoint3x4 라 반사행렬도 정확.
    void ApplyPose(Matrix4x4 T)
    {
        if (_objVerts == null || _mesh == null) return;
        for (int i = 0; i < _objVerts.Length; i++)
            _worldVerts[i] = T.MultiplyPoint3x4(_objVerts[i]);
        _mesh.SetVertices(_worldVerts);
        _mesh.RecalculateBounds();
        if (_axes == null) return;
        Vector3 o = T.MultiplyPoint3x4(Vector3.zero); const float L = 0.12f;
        SetAxis(_axes[0], o, T.MultiplyPoint3x4(new Vector3(L, 0, 0)));
        SetAxis(_axes[1], o, T.MultiplyPoint3x4(new Vector3(0, L, 0)));
        SetAxis(_axes[2], o, T.MultiplyPoint3x4(new Vector3(0, 0, L)));
    }
    static void SetAxis(LineRenderer lr, Vector3 a, Vector3 b) { lr.SetPosition(0, a); lr.SetPosition(1, b); }

    IEnumerator Copy(string rel, string dst)
    {
        if (File.Exists(dst)) yield break;
        string src = Application.streamingAssetsPath + "/" + rel;
        string uri = src.Contains("://") ? src : "file:///" + src.Replace("\\", "/");
        using (var req = UnityWebRequest.Get(uri))
        {
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success) File.WriteAllBytes(dst, req.downloadHandler.data);
            else Debug.LogError("[Srt3dTracker] copy fail " + rel + ": " + req.error);
        }
    }

#if !UNITY_EDITOR
    void OnDestroy()
    {
        UnregisterPointer();
        try { if (_kw != null) { if (_kw.IsRunning) _kw.Stop(); _kw.Dispose(); _kw = null; } } catch { }
#if ENABLE_WINMD_SUPPORT
        if (_pvCap != null) _pvCap.Stop();
#endif
        if (_inited) Srt3dNative.srt3d_release();
    }
#endif
}
