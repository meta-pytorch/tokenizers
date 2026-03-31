@echo off
:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the BSD-style license found in the
:: LICENSE file in the root directory of this source tree.

:: Derived from pytorch/vision approach for Windows wheel builds.
:: Sets up Visual C++ environment and creates a shorter path symlink
:: to avoid Windows MAX_PATH (260 char) issues during CMake builds.

:: Find Visual Studio installation
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_INSTALL_DIR=%%i"
)

if not defined VS_INSTALL_DIR (
    echo ERROR: Could not find Visual Studio installation
    exit /b 1
)

:: Initialize the VC environment for x64
call "%VS_INSTALL_DIR%\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1

set DISTUTILS_USE_SDK=1

:: Create a short path symlink to avoid MAX_PATH issues.
:: The CI checkout path can be very long, e.g.:
::   C:\actions-runner\_work\tokenizers\tokenizers\meta-pytorch\tokenizers
:: We create a junction at a shorter path.
set "SHORT_DIR=%GITHUB_WORKSPACE%\tk"
if exist "%SHORT_DIR%" rmdir "%SHORT_DIR%"
mklink /J "%SHORT_DIR%" "%CD%"
cd /d "%SHORT_DIR%"

:: Execute the build command passed as arguments
%*
if errorlevel 1 exit /b 1
