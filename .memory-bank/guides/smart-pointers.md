---
description: HOW to do smart pointer conversion in JTML — rules, patterns, current status.
status: active
---
# Smart Pointer Conversion Guide

See also: [architecture/overview.md](../architecture/overview.md) for WHY (memory management modernization).

## Current status (2026-04-18)
- 73 conversion tasks generated across 68 classes
- ~30 classes: "CONVERTED – no raw pointers found" (already clean)
- ~10 classes: "SKIPPED" (Qt autogen / stateless functors)
- ~30 classes: "READY" (scaffold created, not yet converted)
- `include/gpu/cuda_deleters.cuh` added (untracked) — custom deleters for CUDA device memory, not yet wired in

## Rules

### MUST convert
- Owning member variables: `T* member_` → `std::unique_ptr<T>` or `std::shared_ptr<T>`
- Use `std::make_unique<T>()` / `std::make_shared<T>()` instead of `new T()`
- Remove destructor `delete` calls (handled automatically)

### MUST NOT convert
- Qt widget pointers with parent-child relationships (Qt manages lifetime)
- CUDA device pointers (`T*` for GPU memory) — keep raw or use custom deleter from `cuda_deleters.cuh`
- Non-owning parameters (function args that just borrow a pointer)

### Passing smart pointers
- `func(member_.get())` — when function expects `T*`
- `func(*member_)` — when function expects `T&`
- `shared_ptr` can be passed/copied directly when sharing ownership

## CUDA-specific patterns
```cpp
// Use custom deleters from include/gpu/cuda_deleters.cuh
std::unique_ptr<float, CudaFreeDeleter> device_buf_;     // cudaMalloc memory
std::unique_ptr<float, CudaFreeHostDeleter> host_buf_;   // cudaMallocHost memory
```

## Validation after each conversion
```bash
pixi run build   # must pass before moving to next class
pixi run tidy    # check no new warnings
```

## Priority order for remaining ~30 classes
1. Leaf classes (fewest dependencies) first
2. Fix BUG-007 (OptimizerManager leaks) — high value
3. Fix BUG-008 (MainScreen optimizer_manager re-alloc) — high risk
4. GPU classes: use cuda_deleters.cuh patterns

## Reference
Full troubleshooting guide: `smart_pointer_conversion.md` in repo root.
Conversion prompt template: `class_conversion_prompt.md` in repo root.
