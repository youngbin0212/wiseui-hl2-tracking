// SRT3D 온디바이스 네이티브 플러그인 (srt3d_uwp.dll) C# 바인딩.
// DLL: Assets/Plugins/WSA/ARM64/srt3d_uwp.dll (UWP ARM64, source_nogl 코어).
// C ABI 6함수를 P/Invoke. Editor 에는 DLL 이 없으니 호출은 빌드(UWP)에서만 동작
// — 선언 자체는 무해(지연 로드)하지만, 호출부는 #if !UNITY_EDITOR 로 감쌀 것.
//
// 좌표계: srt3d 의 pose 는 OpenCV 카메라 기준 ob_in_cam (row-major 4x4).
//         Unity world 로 옮기는 변환은 Srt3dTracker 에서 처리.

using System;
using System.Runtime.InteropServices;

public static class Srt3dNative
{
    const string DLL = "srt3d_uwp";

    // mesh(.obj) + <mesh>.meta 로드 + tracker setup.
    // K9 = row-major [fx,0,cx, 0,fy,cy, 0,0,1]. 반환 1=성공 0=실패.
    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int srt3d_init(string meshPath, float[] K9, int width, int height);

    // 초기 pose 주입 (OpenCV 좌표, row-major 16). 보통 FoundationPose register 결과.
    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int srt3d_reset_pose(float[] pose16);

    // 추적 1회: raw RGB(width*height*3, R,G,B 순) → out17 = pose 16(row-major) + conf.
    // 반환 1=성공 0=실패. out17 은 length>=17 으로 미리 할당.
    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int srt3d_track_rgb(byte[] rgb, int width, int height, [Out] float[] out17);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern void srt3d_release();

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    static extern IntPtr srt3d_last_error();

    // init/track 이 0 을 반환했을 때 원인 메시지.
    public static string LastError()
    {
        try { return Marshal.PtrToStringAnsi(srt3d_last_error()) ?? ""; }
        catch { return "(srt3d_last_error unavailable)"; }
    }
}
