---
description: Product brief (C4 L1): what JTML is, audience, core value, constraints.
status: active
---
# Product: Joint Track Machine Learning (JTML)

## What this is
JointTrack Auto GPU (JTML) is a GPU-accelerated Qt5 desktop application for extracting 6-degree-of-freedom (6-DOF) joint kinematics from clinical fluoroscopic (X-ray) images. It uses the DIRECT global optimization algorithm over a 6D pose hyperrectangle, comparing GPU-rendered implant silhouettes/DRRs against fluoroscopy images via configurable registration metrics. A PyTorch/TorchScript neural network provides an initial pose estimate for the optimizer.

## Core value
**Accurate, automated 6-DOF pose estimation of orthopedic implants from fluoroscopy images** — the thing that MUST work: the DIRECT optimizer converges to correct implant pose using GPU-rendered metrics.

## Audience
Orthopaedic biomechanics researchers (Gary J. Miller Orthopaedic Biomechanics Lab, University of Florida). Not a clinical/consumer product — research tool.

## Primary user flow
1. Load fluoroscopy image(s) + 3D implant STL model(s)
2. Set camera calibration parameters
3. Set optimization settings (metric type, DIRECT stages, bounds)
4. Run NN pose estimate → optimizer → converge to 6-DOF pose
5. Export kinematics results (6-DOF pose tables, .jts/.jtak format)

## Tech stack
- Language: C++20, CUDA 20
- GPU: CUDA 12.4, cuDNN 9.3.0.75, PyTorch 2.4.1 (CUDA 12.0)
- UI: Qt 5.15.8
- Rendering: VTK 9.3.0 (built from source), OpenCV 4.10.0
- Math: Eigen 3.4.0
- Build: CMake 3.31.4 + Ninja via pixi (conda-forge + nvidia channels)
- Tooling: clang-format 19.1.2, clang-tidy 19

## Constraints
- Linux only (Fedora tested; conda-based environment)
- Requires NVIDIA GPU (CUDA 12.4, native arch — not portable)
- No automated test suite
- No functional CI
- VTK must be built from source (slow first-build step)

## Key decisions
| Decision | Rationale | Status |
|---|---|---|
| DIRECT optimizer | Global optimizer suitable for non-convex 6D pose space | active |
| pixi/conda environment | Reproducible deps including CUDA/Qt/VTK | active |
| Smart pointer conversion | Modernize raw-ptr ownership, reduce leak risk | in progress |
| native CUDA arch | Optimize for local GPU | accepted tradeoff |
