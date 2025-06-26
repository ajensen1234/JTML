---
description: Environment setup runbook — from scratch to working build on Linux.
status: active
---
# Setup Runbook

## Requirements
- Linux (Fedora 43 tested; Ubuntu 22.04 should work)
- NVIDIA GPU with driver supporting CUDA 12.4
- `pixi` package manager

## Step 1: Install pixi
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Step 2: Clone and install conda environment
```bash
git clone <repo-url> JTML
cd JTML
pixi install
```
This downloads CUDA 12.4, Qt 5.15.8, VTK 9.3.0, PyTorch 2.4.1, OpenCV 4.10.0, Eigen 3.4.0, clang-tools 19 from conda-forge/nvidia channels.

## Step 3: Build VTK from source (first time only)
```bash
pixi run build-vtk
```
Takes 10–30 minutes. Output: `./_deps/vtk/build/`. Guarded by output check on `./_deps/vtk/build/CMakeCache.txt`.

## Step 4: Build JTML
```bash
pixi run build
```
Chains: configure → compile. Output: `.build/bin/joint-track-machine-learning`.

## Step 5: Verify
```bash
pixi run run
```
Qt window should appear.

## Display setup (Wayland/X11)
If the app doesn't display, check:
```bash
pixi run check-display
```
Environment sets `QT_QPA_PLATFORM=xcb;wayland` and `GDK_BACKEND=x11;wayland`.

## Incremental development
```bash
pixi run build    # incremental rebuild (skips VTK/configure if unchanged)
pixi run format   # auto-format sources
pixi run tidy     # clang-tidy analysis
```
