// SRT3D on-device C bridge for HoloLens 2 (UWP / IL2CPP).
// XRHandEyeTracker 의 srt3d_jni.cpp(Galaxy XR, 검증됨)를 UWP용으로 변환:
//   - JNI 시그니처 → extern "C" __declspec(dllexport) C ABI (Unity DllImport)
//   - 입력을 JPEG → raw RGB 로 변경 (cv::imdecode 불사용 → imgcodecs 의존성 제거,
//     OpenCV 는 core+imgproc 만으로 충분)
//   - android/log → 선택적 로그 콜백 + last_error 문자열 (기기 디버깅용)
// 코어 로직(Body/Model/RegionModality/Tracker)은 JNI 와 동일. 단일 세션(전역 상태).
//
// Unity 측 호출 순서:
//   srt3d_init(meshPath, K9, w, h) -> srt3d_reset_pose(pose16, FP init) ->
//   매 프레임 srt3d_track_rgb(rgb, w, h, out17) -> 종료 시 srt3d_release()

#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <exception>

#include <opencv2/core.hpp>

#include <srt3d/body.h>
#include <srt3d/model.h>
#include <srt3d/region_modality.h>
#include <srt3d/tracker.h>
#include <srt3d/common.h>
#include "srt3d/virtual_camera.hpp"

#if defined(_WIN32)
#define SRT3D_API extern "C" __declspec(dllexport)
#else
#define SRT3D_API extern "C"
#endif

namespace {
std::shared_ptr<srt3d::Body> g_body;
std::shared_ptr<srt3d::Model> g_model;
std::shared_ptr<srt3d::VirtualCamera> g_camera;
std::shared_ptr<srt3d::RegionModality> g_rm;
std::shared_ptr<srt3d::Tracker> g_tracker;

void (*g_log)(const char*) = nullptr;
std::string g_last_error;

// [회복 부스트] reset_pose 직후 N프레임은 탐색을 넓게 해서 FP 초기 자세(크게 틀린 tilt)를 끌어온다.
//   기본 scales {5,2,2,1}(최대 95px) + n_corr=7 은 '추적' 값이라 회복 basin 이 좁다.
//   부스트: scales {12,8,5,3,2,1}(최대 12*19=228px) + n_corr=30 → basin 확장. 수렴 후 평소 값 복귀.
int g_boost_frames = 0;
const std::vector<int> kScalesTrack = {5, 2, 2, 1};
const std::vector<int> kScalesBoost = {12, 8, 5, 3, 2, 1};
const int kBoostFramesTotal = 12;

void logf(const std::string& s) { if (g_log) g_log(s.c_str()); }
void set_err(const std::string& s) { g_last_error = s; logf("[srt3d] " + s); }

Eigen::Matrix4f arr16_to_mat(const float* a) {
    Eigen::Matrix4f m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) m(r, c) = a[r * 4 + c];   // row-major in
    return m;
}
}  // namespace

// 로그 콜백 등록 (Unity 콘솔로 라우팅). 선택사항.
SRT3D_API void srt3d_set_log_callback(void (*cb)(const char*)) { g_log = cb; }

// 마지막 에러 메시지 (init/track 이 0 반환 시 원인 확인용).
SRT3D_API const char* srt3d_last_error() { return g_last_error.c_str(); }

// 초기화: mesh(.obj) + <mesh>.meta 로드 + tracker setup.
// K9 = row-major [fx,0,cx, 0,fy,cy, 0,0,1]. 반환 1=성공 0=실패.
SRT3D_API int srt3d_init(const char* meshPath, const float* K9, int width, int height) {
    try {
        std::string mesh(meshPath ? meshPath : "");
        const float fx = K9[0], cx = K9[2], fy = K9[4], cy = K9[5];

        const std::string name = "obj";
        g_body = std::make_shared<srt3d::Body>(
            name, mesh, 1.0f, true, true, srt3d::Transform3fA::Identity());
        g_model = std::make_shared<srt3d::Model>(name, g_body, mesh + ".meta");
        if (!g_model->SetUp()) { set_err("model SetUp failed (.meta load?)"); return 0; }

        g_camera = std::make_shared<srt3d::VirtualCamera>("vcam");
        srt3d::Intrinsics intr{fx, fy, cx, cy, (int)width, (int)height};
        g_camera->set_intrinsics(intr);
        g_camera->SetUp();
        g_camera->set_realtime(true);

        g_tracker = std::make_shared<srt3d::Tracker>("tracker");
        g_tracker->set_n_corr_iterations(7);
        g_tracker->set_n_update_iterations(2);
        g_rm = std::make_shared<srt3d::RegionModality>(name, g_body, g_model, g_camera);
        g_rm->set_kl_threshold(1.0f);
        g_tracker->AddRegionModality(g_rm);
        g_tracker->SetUpTracker();

        g_last_error.clear();
        logf("srt3d_init OK: " + mesh + " " + std::to_string(width) + "x" + std::to_string(height));
        return 1;
    } catch (const std::exception& e) {
        set_err(std::string("srt3d_init exception: ") + e.what());
        return 0;
    }
}

// 초기 pose 주입 (OpenCV 좌표, 4x4 row-major 16개). 보통 FP register 결과.
SRT3D_API int srt3d_reset_pose(const float* pose16) {
    if (!g_body || !pose16) { set_err("reset_pose: not initialized"); return 0; }
    g_body->set_body2world_pose(srt3d::Transform3fA(arr16_to_mat(pose16)));
    g_boost_frames = kBoostFramesTotal;   // 다음 N프레임 회복 부스트
    return 1;
}

// 추적 1회: raw RGB(width*height*3, R,G,B 순) → out17 = pose 16(row-major) + conf.
// 반환 1=성공 0=실패. (JNI 는 jpeg 를 받아 RGB 로 변환 후 push 했음 — 여기선 이미 RGB 라 그대로 push.)
SRT3D_API int srt3d_track_rgb(const unsigned char* rgb, int width, int height, float* out17) {
    if (!g_tracker || !g_body) { set_err("track: not initialized"); return 0; }
    if (!rgb || !out17) { set_err("track: null arg"); return 0; }
    try {
        // 외부 버퍼를 감싼 뒤 clone — 카메라가 소유하도록 복사본 push (RGB 순서).
        cv::Mat view(height, width, CV_8UC3, const_cast<unsigned char*>(rgb));
        g_camera->PushImage(view.clone());

        // [회복 부스트] reset 직후 N프레임: 넓은 scales + 많은 corr iter 로 basin 확장.
        //   scales/n_corr 는 TrackIter 가 매 프레임 읽는 멤버라 재-setup 없이 즉시 반영됨.
        if (g_boost_frames > 0) {
            g_rm->set_scales(kScalesBoost);
            g_tracker->set_n_corr_iterations(30);
            g_boost_frames--;
        } else {
            g_rm->set_scales(kScalesTrack);
            g_tracker->set_n_corr_iterations(7);
        }

        if (!g_tracker->set_up()) g_tracker->SetUpTracker();
        g_tracker->TrackIter(cv::Mat());
        std::vector<float> conf = g_tracker->EvaluateDistribution();

        Eigen::Matrix4f pose = g_body->body2world_pose().matrix();
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) out17[r * 4 + c] = pose(r, c);   // row-major out
        out17[16] = conf.empty() ? 0.0f : conf[0];
        return 1;
    } catch (const std::exception& e) {
        set_err(std::string("srt3d_track exception: ") + e.what());
        return 0;
    }
}

SRT3D_API void srt3d_release() {
    g_rm.reset(); g_tracker.reset(); g_camera.reset(); g_model.reset(); g_body.reset();
    logf("srt3d_release");
}
