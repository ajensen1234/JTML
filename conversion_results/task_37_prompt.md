# Smart Pointer Conversion Task: CameraInteractorStyle

## Your Mission
Convert the class `CameraInteractorStyle` to use smart pointers. You are responsible ONLY for this class.

## Files to Modify
- Header: `./include/gui/interactor.h`
- Source: `None`
- CUDA Header: `None`
- CUDA Source: `None`

## Pre-Conversion Analysis
Run these commands first:
```bash
# Check current clang-tidy issues for your non-CUDA files only
clang-tidy ./include/gui/interactor.h None -p .build --checks="modernize-*,cppcoreguidelines-owning-memory" 

# Look for raw pointer patterns in ALL your class files
grep -n "new\|delete\|\*.*=" ./include/gui/interactor.h None None None

# For CUDA files, check for device memory patterns
grep -n "cudaMalloc\|cudaFree\|thrust::" None None
```

## Conversion Rules (STRICT)
1. **Member Variables**: Convert owning raw pointers to `std::unique_ptr`
2. **Parameters**: 
   - Keep raw pointers for non-owning parameters
   - Use `.get()` when passing smart pointers to functions expecting raw pointers
   - Use `*smart_ptr` when passing to functions expecting references
3. **Return Values**: Return `std::unique_ptr` for newly created objects
4. **Qt Exclusions**: Do NOT convert Qt widget pointers with parent/child relationships
5. **CUDA Considerations**:
   - Device pointers (`T*` for GPU memory) should generally stay raw
   - Host-side management of device pointers can use smart pointers with custom deleters
   - Thrust smart pointers (`thrust::device_ptr`) may be appropriate for some cases
   - CUDA streams and events are typically managed by CUDA runtime

## Common Fixes Reference
```cpp
// Member variable conversion
T* member_;              → std::unique_ptr<T> member_;

// Constructor
member_ = new T();       → member_ = std::make_unique<T>();

// Destructor  
delete member_;          → // Remove - automatic cleanup

// Function calls
func(member_);           → func(member_.get());     // if func expects T*
func(*member_);          → func(*member_);          // if func expects T&

// Null checks
if (member_)             → if (member_)             // works the same
if (member_ == nullptr)  → if (!member_)            // or keep as is

// CUDA-specific patterns
// Device memory with custom deleter
std::unique_ptr<T, CudaDeleter> device_ptr_;

// Custom deleter example
struct CudaDeleter {
    void operator()(T* ptr) { 
        if (ptr) cudaFree(ptr); 
    }
};
```

## Validation Process
1. Make your changes
2. Run: `pixi run build` - must compile cleanly
3. Run: `pixi run tidy --path .build` - check for new issues
4. If errors, fix using the patterns above

## Success Criteria
- Class compiles without errors
- No new clang-tidy warnings about raw pointers in your class
- Existing functionality preserved (same public interface where possible)
- No memory leaks (verified by smart pointer usage)

## Report Format
When complete, provide:
1. Summary of changes made
2. Any challenges encountered
3. Build/tidy status (pass/fail)
4. Suggested review points

**Remember: Focus ONLY on CameraInteractorStyle. Do not modify other classes.**