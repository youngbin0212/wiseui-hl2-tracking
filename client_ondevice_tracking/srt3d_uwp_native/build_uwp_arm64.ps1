# srt3d_uwp.dll (HoloLens 2, UWP/WindowsStore ARM64) 빌드.
# OpenCV UWP ARM64(core+imgproc) 가 먼저 빌드돼 있어야 함:
#   D:\ProjectsTracking\opencv_uwp\install
#
# 실행:  powershell -ExecutionPolicy Bypass -File D:\ProjectsTracking\srt3d_uwp\build_uwp_arm64.ps1
# 산출:  build_arm64_uwp\Release\srt3d_uwp.dll  (Unity Plugins\WSA\ARM64 에 넣을 것)

$ErrorActionPreference = "Stop"
$root    = "D:\ProjectsTracking\srt3d_uwp"
$build   = "$root\build_arm64_uwp"
# 정적 빌드의 arch별 config 폴더를 직접 지정 (루트 dispatcher는 STATIC 인식 실패함).
$opencv  = "D:\ProjectsTracking\opencv_uwp\install\ARM64\vc17\staticlib"

if (-not (Test-Path "$opencv\OpenCVConfig.cmake")) { throw "OpenCV UWP install 없음: $opencv" }

Write-Host "=== configure (UWP ARM64 DLL) ===" -ForegroundColor Cyan
cmake -S $root -B $build `
  -G "Visual Studio 17 2022" -A ARM64 `
  -DCMAKE_SYSTEM_NAME=WindowsStore `
  -DCMAKE_SYSTEM_VERSION="10.0" `
  -DCMAKE_SYSTEM_PROCESSOR=ARM64 `
  -DCMAKE_POLICY_VERSION_MINIMUM="3.5" `
  -DOpenCV_STATIC=ON `
  -DOpenCV_DIR="$opencv"
if ($LASTEXITCODE -ne 0) { throw "configure 실패 ($LASTEXITCODE)" }

Write-Host "=== build (Release) ===" -ForegroundColor Cyan
cmake --build $build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "build 실패 ($LASTEXITCODE)" }

Write-Host "=== 산출 DLL ===" -ForegroundColor Green
Get-ChildItem $build -Recurse -Filter 'srt3d_uwp.dll' | Select-Object FullName, @{n='KB';e={[math]::Round($_.Length/1KB,1)}}
