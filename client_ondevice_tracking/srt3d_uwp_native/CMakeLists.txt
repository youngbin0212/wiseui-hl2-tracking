# SRT3D 온디바이스 네이티브 플러그인 — HoloLens 2 (UWP/WindowsStore, ARM64).
# XRHandEyeTracker(Android NDK)의 CMake 를 UWP 로 변환:
#   - find_package(OpenCV): Android SDK → 우리가 빌드한 UWP ARM64 install
#   - SHARED 라이브러리(.dll) — Unity 네이티브 플러그인으로 로드
#   - android/log 링크 제거
# GL-free: SRT3D_NO_GL 정의, renderer/viewer/occlusion .cpp 없음(코어 6개만).
cmake_minimum_required(VERSION 3.18)
project(srt3d_uwp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 우리가 빌드한 UWP ARM64 OpenCV (core+imgproc, static). build 스크립트에서 -DOpenCV_DIR 로 전달.
if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "D:/ProjectsTracking/opencv_uwp/install")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# SRT3D 코어 6개 .cpp (source_nogl, XRHandEyeTracker 에서 복사한 검증본)
file(GLOB SRT3D_CORE ${CMAKE_CURRENT_SOURCE_DIR}/srt3d/src/*.cpp)

add_library(srt3d_uwp SHARED
    ${SRT3D_CORE}
    ${CMAKE_CURRENT_SOURCE_DIR}/srt3d_uwp.cpp
)

target_compile_definitions(srt3d_uwp PRIVATE SRT3D_NO_GL)

target_include_directories(srt3d_uwp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}                    # "srt3d/virtual_camera.hpp"
    ${CMAKE_CURRENT_SOURCE_DIR}/srt3d/include       # <srt3d/*.h>
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party         # <tiny_obj_loader/*.h>
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3  # <Eigen/*>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(srt3d_uwp ${OpenCV_LIBS})

# UWP(WindowsStore) 링커 이슈 해결: vccorlib.lib 가 msvcrt.lib 보다 먼저 링크돼야 함.
# (LNK2038 vccorlib_lib_should_be_specified_before_msvcrt_lib / LNK2005 __crtWinrtInitType)
# MS 공식 해법: 기본 자동링크를 끄고(NODEFAULTLIB) 올바른 순서로 명시.
if(CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
    target_link_options(srt3d_uwp PRIVATE
        $<$<CONFIG:Release>:/NODEFAULTLIB:vccorlib.lib>
        $<$<CONFIG:Release>:/NODEFAULTLIB:msvcrt.lib>
        $<$<CONFIG:Release>:vccorlib.lib>
        $<$<CONFIG:Release>:msvcrt.lib>
        $<$<CONFIG:Debug>:/NODEFAULTLIB:vccorlibd.lib>
        $<$<CONFIG:Debug>:/NODEFAULTLIB:msvcrtd.lib>
        $<$<CONFIG:Debug>:vccorlibd.lib>
        $<$<CONFIG:Debug>:msvcrtd.lib>)
endif()

set_target_properties(srt3d_uwp PROPERTIES OUTPUT_NAME "srt3d_uwp" PREFIX "")
