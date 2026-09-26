// Srt3dPvCapture — MediaFrameReader 기반 PV 캡처 (2단계: 프레임 → srt3d 투입).
// PhotoCapture 는 4K(3904x2196)만 줘서 fps≈0. MediaFrameReader 비디오 소스는 896x504@30fps 확보
// (검증 완료: ExclusiveControl 로 hl2ss 와 mutex 충돌 없이 OPEN, fps≈29.7).
//
// 프레임마다 추출: BGRA→RGB, CameraIntrinsics(fx,fy,cx,cy 실제값), cam2world(Unity world).
//   - K 는 기기가 주는 실제 intrinsics → PhotoCapture projection 추측(cx/cy 부호) 제거.
//   - cam2world = frame.CoordinateSystem.TryGetTransformTo(scs). RH(Windows)→LH(Unity) 변환은 [TUNE].
// 백그라운드 스레드(FrameArrived)에서 채우고, 메인 스레드가 TryGetLatest 로 소비 → srt3d 호출.
//
// WinRT 는 Unity UWP 빌드(ENABLE_WINMD_SUPPORT)에서만 컴파일.

using System;
using UnityEngine;
#if ENABLE_WINMD_SUPPORT
using System.Threading.Tasks;
using System.Runtime.InteropServices.WindowsRuntime;   // IBuffer.ToArray
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;                   // MediaEncodingSubtypes
using Windows.Graphics.Imaging;
using Windows.Perception.Spatial;
#endif

public class Srt3dPvCapture
{
    // HUD/로그용 상태 (메인 스레드 읽기)
    public string Status = "(idle)";
    public string Formats = "";
    public int FrameCount = 0;
    public int Width = 0, Height = 0;
    public bool Running = false, Opened = false;
    public string DiagIntr = "?";     // intrinsics 나오는지 진단
    public bool FlipHandedness = false; // transpose-only 전달 (flip 후보는 Srt3dTracker 가 계산·판별)
    // ⚠️ FlipH 를 켜지 말 것. 2026-07-22 기기 실측으로 반증됨.
    //
    // 예전 주석은 "MFR SoftwareBitmap 이 좌우 미러라 conf=0 (PNG 로 확인)" 이었으나 둘 다 틀렸다:
    //   (1) "PNG 로 확인" — 그 PNG 는 DumpFrame 의 Texture2D.LoadRawTextureData + EncodeToPNG
    //       경로라 Unity 텍스처가 bottom-up 이어서 **상하가 뒤집혀** 저장된다. 뒤집힌 글자를
    //       거울상으로 오독한 것으로 보인다. 좌우 미러는 관측된 적이 없다.
    //   (2) "conf=0" — 캡처 순간 객체가 시야 밖이었던 일시적 값이지 지속적 실패가 아니었다.
    //
    // FlipH=true 로 되살렸을 때의 실측 결과: **추적이 좌우 반대로 따라간다.**
    // (물체를 오른쪽으로 옮기면 홀로그램이 왼쪽으로.) 수식으로도 정확히 그렇게 나온다 —
    //   미러 이미지에서 물체는  u' = (w-1) - u = cx' - fx·X/Z   (cx' = (w-1) - cx)
    //   srt3d 모델은            u' = fx·X'/Z' + cx'
    //   ∴ X'/Z' = -X/Z          → 복원 pose 의 X 가 반전된다.
    //
    // 핵심: flip + cx 반전은 "전달 버퍼에 대한 서술"로는 일관되지만, 원본이 미러가 아닌데
    // 적용하면 **미러된 뷰를 만들어낸다.** 미러 뷰는 어떤 강체 pose 로도 만들 수 없으므로
    // srt3d 는 X 반전으로 억지 근사하게 되고, 그게 위 증상이다.
    // 되살리려면 먼저 원본이 실제로 미러인지 증명할 것(글자가 있는 대상 + 상하 반전을 보정한 덤프).
    public bool FlipH = false, FlipV = false;
    // [DIAG] (B)stride: Stride 가 W4(=w*4)보다 크면 패딩 → CopyToBuffer 어긋나 이미지 사선 → conf=0
    public int Stride = 0, W4 = 0;
    public float NonzeroPct = 0f;     // rgb 유효 픽셀 % (0 근처면 검은/깨진 이미지)
    public string DiagC2W = "?";      // cam2world raw(WinRT) vs 변환 후(camPos) 대조
    public string DiagFmt = "?";      // [DIAG] SoftwareBitmap 실제 픽셀 포맷 (BGRA8 인지 NV12 인지 확정)

#if ENABLE_WINMD_SUPPORT
    MediaCapture _mc;
    MediaFrameReader _reader;
    SpatialCoordinateSystem _scs;      // Unity world (메인 스레드에서 주입)

    // 최신 프레임 홀더 (lock 로 보호)
    readonly object _lock = new object();
    byte[] _rgb; int _fw, _fh; float _fx, _fy, _cx, _cy;
    Matrix4x4 _c2w = Matrix4x4.identity;
    bool _hasNew = false, _hasK = false, _hasPose = false;

    public void SetWorldCoordinateSystem(SpatialCoordinateSystem scs) { _scs = scs; }

    // 메인 스레드: 최신 프레임 소비. rgb 는 dst 로 복사(dst 재사용). 반환 true=새 프레임.
    public bool TryGetLatest(ref byte[] dst, out int w, out int h,
                             out float fx, out float fy, out float cx, out float cy,
                             out Matrix4x4 c2w, out bool hasK, out bool hasPose)
    {
        lock (_lock)
        {
            w = _fw; h = _fh; fx = _fx; fy = _fy; cx = _cx; cy = _cy;
            c2w = _c2w; hasK = _hasK; hasPose = _hasPose;
            if (!_hasNew || _rgb == null) return false;
            if (dst == null || dst.Length != _rgb.Length) dst = new byte[_rgb.Length];
            Array.Copy(_rgb, dst, _rgb.Length);
            _hasNew = false;
            return true;
        }
    }

    public async Task<bool> TryOpenAsync(MediaCaptureSharingMode sharing)
    {
        try
        {
            Status = "find source groups...";
            var groups = await MediaFrameSourceGroup.FindAllAsync();
            MediaFrameSourceGroup pvGroup = null; MediaFrameSourceInfo pvInfo = null;
            foreach (var g in groups)
            {
                foreach (var si in g.SourceInfos)
                    if (si.SourceKind == MediaFrameSourceKind.Color) { pvGroup = g; pvInfo = si; break; }
                if (pvGroup != null) break;
            }
            if (pvGroup == null) { Status = "no Color(PV) source"; return false; }

            _mc = new MediaCapture();
            var settings = new MediaCaptureInitializationSettings
            {
                SourceGroup = pvGroup, SharingMode = sharing,
                MemoryPreference = MediaCaptureMemoryPreference.Cpu,
                StreamingCaptureMode = StreamingCaptureMode.Video,
            };
            Status = $"init MediaCapture ({sharing})...";
            await _mc.InitializeAsync(settings);

            var source = _mc.FrameSources[pvInfo.Id];
            SummarizeFormats(source);
            if (sharing == MediaCaptureSharingMode.ExclusiveControl)
                await TrySetLowFormat(source);

            _reader = await _mc.CreateFrameReaderAsync(source, MediaEncodingSubtypes.Bgra8);
            _reader.FrameArrived += OnFrameArrived;
            var st = await _reader.StartAsync();
            Opened = true; Running = st == MediaFrameReaderStartStatus.Success;
            Status = $"reader={st} ({sharing})";
            return Running;
        }
        catch (Exception e) { Status = $"ERR({sharing}): " + e.Message; return false; }
    }

    void SummarizeFormats(MediaFrameSource source)
    {
        int minW = int.MaxValue, maxW = 0, count = 0, minH = 0, minFps = 0;
        foreach (var f in source.SupportedFormats)
        {
            if (f.VideoFormat == null) continue;
            int w = (int)f.VideoFormat.Width, h = (int)f.VideoFormat.Height;
            int fps = (f.FrameRate != null && f.FrameRate.Denominator != 0) ? (int)(f.FrameRate.Numerator / f.FrameRate.Denominator) : 0;
            count++;
            if (w < minW) { minW = w; minH = h; minFps = fps; }
            if (w > maxW) maxW = w;
        }
        Formats = count == 0 ? "no video formats" : $"{count}fmts min {minW}x{minH}@{minFps} max {maxW}";
    }

    async Task TrySetLowFormat(MediaFrameSource source)
    {
        MediaFrameFormat best = null; int bestScore = int.MaxValue;
        foreach (var f in source.SupportedFormats)
        {
            if (f.VideoFormat == null) continue;
            int w = (int)f.VideoFormat.Width;
            int fps = (f.FrameRate != null && f.FrameRate.Denominator != 0) ? (int)(f.FrameRate.Numerator / f.FrameRate.Denominator) : 0;
            int score = Math.Abs(w - 760) + (fps >= 25 ? 0 : 1000);   // 760 근처 + 30fps 우선
            if (score < bestScore) { bestScore = score; best = f; }
        }
        if (best != null)
            try { await source.SetFormatAsync(best); } catch (Exception e) { Status = "setfmt fail: " + e.Message; }
    }

    void OnFrameArrived(MediaFrameReader r, MediaFrameArrivedEventArgs a)
    {
        var frame = r.TryAcquireLatestFrame();
        if (frame == null) return;
        try
        {
            var vmf = frame.VideoMediaFrame;
            if (vmf == null) return;
            FrameCount++;

            // 1) 픽셀 BGRA → RGB
            var bmp = vmf.SoftwareBitmap;
            if (bmp == null) return;
            int w = bmp.PixelWidth, h = bmp.PixelHeight;
            Width = w; Height = h;
            DiagFmt = bmp.BitmapPixelFormat.ToString();   // [DIAG] 실제 포맷 (Bgra8 / Nv12 등)

            // [DIAG-B] 실제 stride 관측 (패딩 여부). w*4 와 다르면 CopyToBuffer 정렬 어긋남 의심.
            try { using (var bb = bmp.LockBuffer(BitmapBufferAccessMode.Read)) { var pd = bb.GetPlaneDescription(0); Stride = pd.Stride; W4 = w * 4; } } catch { }

            var buffer = new Windows.Storage.Streams.Buffer((uint)(w * h * 4));
            bmp.CopyToBuffer(buffer);
            byte[] bgra = buffer.ToArray();
            byte[] rgb = new byte[w * h * 3];
            // BGRA→RGB (+ 좌우/상하 flip). MFR 는 미러가 아니므로 둘 다 false 가 정답 — 위 주석 참조.
            for (int y = 0; y < h; y++)
            {
                int row = y * w;
                for (int x = 0; x < w; x++)
                {
                    int s = (row + x) * 4;
                    int dx = FlipH ? (w - 1 - x) : x;
                    int dy = FlipV ? (h - 1 - y) : y;
                    int d = (dy * w + dx) * 3;
                    rgb[d] = bgra[s + 2]; rgb[d + 1] = bgra[s + 1]; rgb[d + 2] = bgra[s];
                }
            }

            // [DIAG-B] rgb 유효 픽셀 % (검은/깨진 이미지면 0 근처)
            int nz = 0, samp = 0, step = Math.Max(1, rgb.Length / 30000);
            for (int i = 0; i < rgb.Length; i += step) { if (rgb[i] != 0) nz++; samp++; }
            NonzeroPct = samp > 0 ? 100f * nz / samp : 0f;

            // 2) 실제 K (intrinsics)
            float fx = 0, fy = 0, cx = 0, cy = 0; bool hasK = false;
            var intr = vmf.CameraIntrinsics;
            if (intr != null)
            {
                fx = intr.FocalLength.X; fy = intr.FocalLength.Y;
                cx = intr.PrincipalPoint.X; cy = intr.PrincipalPoint.Y;
                if (FlipH) cx = (w - 1) - cx;   // 이미지 좌우 flip 에 맞춰 principal point 반전
                if (FlipV) cy = (h - 1) - cy;
                hasK = true;
                DiagIntr = $"OK fx={fx:F1} fy={fy:F1} cx={cx:F1} cy={cy:F1}";
            }
            else DiagIntr = "NULL(no intrinsics)";

            // 3) cam2world (Unity world). RH→LH 는 FlipHandedness [TUNE].
            Matrix4x4 c2w = Matrix4x4.identity; bool hasPose = false;
            if (_scs != null && frame.CoordinateSystem != null)
            {
                var t = frame.CoordinateSystem.TryGetTransformTo(_scs);
                if (t.HasValue)
                {
                    var nm = t.Value;
                    c2w = NumericsToUnity(nm); hasPose = true;
                    // [DIAG-3] raw WinRT translation vs 변환 후 camera world pos(≈헤드 위치여야 정상).
                    DiagC2W = $"rawT({nm.M41:F2},{nm.M42:F2},{nm.M43:F2}) camPos({c2w.m03:F2},{c2w.m13:F2},{c2w.m23:F2})";
                }
            }

            lock (_lock)
            {
                _rgb = rgb; _fw = w; _fh = h;
                _fx = fx; _fy = fy; _cx = cx; _cy = cy; _hasK = hasK;
                _c2w = c2w; _hasPose = hasPose; _hasNew = true;
            }
        }
        finally { frame.Dispose(); }
    }

    // Microsoft 공식 변환 (Mixed Reality native objects in Unity → "Converting between coordinate systems").
    // Unity Matrix4x4 생성자는 열(column)을 받음 → System.Numerics 행을 열로 = transpose.
    // 부호 패턴 = handedness conjugation F·A·F (F=diag(1,1,-1)). 한쪽 flip 아님, 양쪽 정확.
    static Matrix4x4 NumericsToUnity(System.Numerics.Matrix4x4 m)
        => new Matrix4x4(
            new Vector4( m.M11,  m.M12, -m.M13,  m.M14),
            new Vector4( m.M21,  m.M22, -m.M23,  m.M24),
            new Vector4(-m.M31, -m.M32,  m.M33, -m.M34),
            new Vector4( m.M41,  m.M42, -m.M43,  m.M44));

    public void Stop()
    {
        try { if (_reader != null) { _reader.FrameArrived -= OnFrameArrived; var _ = _reader.StopAsync(); _reader = null; } } catch { }
        try { if (_mc != null) { _mc.Dispose(); _mc = null; } } catch { }
        Running = false;
    }
#endif
}
