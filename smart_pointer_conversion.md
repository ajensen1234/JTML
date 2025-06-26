# Smart Pointer Conversion Troubleshooting Guide

## Common Compiler Error Patterns & Solutions

### Error: "Cannot convert from 'std::shared_ptr<T>' to 'T*'"

**What happened:** Function expects raw pointer, you passed smart pointer directly

**❌ Wrong Fix:**
```cpp
// DON'T randomly add * everywhere
someFunction(*mySharedPtr);  // Wrong if function expects T*
```

**✅ Correct Analysis & Fix:**
```cpp
// 1. Check function signature first
void someFunction(MyClass* ptr);        // Expects T*
void otherFunction(const MyClass& ref); // Expects T&

// 2. Use appropriate conversion
someFunction(mySharedPtr.get());        // For T* parameters
otherFunction(*mySharedPtr);            // For T& parameters
```

### Error: "Cannot convert from 'T*' to 'std::shared_ptr<T>'"

**What happened:** Trying to assign raw pointer to smart pointer

**❌ Wrong Fix:**
```cpp
// DON'T do direct assignment
std::shared_ptr<T> ptr = someRawPointer;  // Compiler error
```

**✅ Correct Solutions:**

**Case 1: Taking ownership of existing pointer**
```cpp
T* rawPtr = getRawPointer();
std::shared_ptr<T> smartPtr(rawPtr);  // Take ownership
// or
auto smartPtr = std::shared_ptr<T>(rawPtr);
```

**Case 2: Creating new object**
```cpp
// Replace this pattern:
T* ptr = new T(args);
// With this:
auto ptr = std::make_shared<T>(args);
```

**Case 3: Non-owning reference (keep as raw pointer)**
```cpp
// If you don't own the object, don't convert:
void processData(T* data) {  // Keep as raw pointer
    // Just using the data, not managing lifetime
}
```

### Error: Copying shared_ptr (Much Easier Than unique_ptr!)

**What happened:** Unlike unique_ptr, shared_ptr CAN be copied safely

**✅ These all work fine with shared_ptr:**
```cpp
std::shared_ptr<T> ptr1 = std::make_shared<T>();
std::shared_ptr<T> ptr2 = ptr1;  // ✓ OK: Creates another reference
std::vector<std::shared_ptr<T>> vec;
vec.push_back(ptr1);  // ✓ OK: Copies the shared_ptr
```

**✅ Passing to functions is easy:**
```cpp
// Taking shared ownership:
void takeShared(std::shared_ptr<T> ptr);
takeShared(myPtr);  // Just pass directly, no std::move needed

// Not taking ownership (more common):
void borrowObject(T* ptr);
borrowObject(myPtr.get());  // Use .get() for raw pointer access
```

**✅ Function returns:**
```cpp
std::shared_ptr<T> createObject() { 
    return std::make_shared<T>(); 
}

// All callers can just assign directly:
auto obj1 = createObject();
auto obj2 = obj1;  // Both share ownership now
```

### Error: Function Return Type Mismatches

**Scenario:** Function returns raw pointer, you changed it to return smart pointer

**❌ Problematic Change:**
```cpp
// Original:
T* createObject() { return new T(); }

// Naive conversion:
std::unique_ptr<T> createObject() { return std::make_unique<T>(); }
// Now all callers break!
```

**✅ Strategy: Update Callers Systematically**

**Step 1: Find all callers**
```bash
grep -rn "createObject()" ./src ./include
```

**Step 2: Update callers based on usage pattern**
```cpp
// Pattern 1: Caller takes shared ownership
// Old:
T* obj = createObject();
// New:
auto obj = createObject();  // or std::shared_ptr<T> obj = createObject();

// Pattern 2: Caller doesn't take ownership (common)
// Old:
T* obj = createObject();
processData(obj);  // processData doesn't take ownership
delete obj;
// New:
auto obj = createObject();
processData(obj.get());  // Use .get() for raw pointer access
// No delete needed - automatic cleanup

// Pattern 3: Caller stores in container
// Old:
std::vector<T*> vec;
vec.push_back(createObject());
// New:
std::vector<std::shared_ptr<T>> vec;
vec.push_back(createObject());  // Direct assignment works!

// Pattern 4: Multiple references (shared_ptr advantage)
// Old:
T* obj = createObject();
manager1->setObject(obj);  // Dangerous - who owns it?
manager2->setObject(obj);  // Dangerous - double delete risk?
// New:
auto obj = createObject();
manager1->setObject(obj);  // Both managers share ownership
manager2->setObject(obj);  // Safe - reference counted
```

### Error: Member Initialization Issues

**Scenario:** Constructor initialization with smart pointers

**❌ Common Mistakes:**
```cpp
class MyClass {
    std::unique_ptr<T> member_;
public:
    // Wrong:
    MyClass() : member_(new T()) {}  // Don't use raw new
    
    // Wrong:
    MyClass(T* ptr) : member_(ptr) {}  // Unsafe ownership transfer
};
```

**✅ Correct Patterns:**
```cpp
class MyClass {
    std::shared_ptr<T> member_;
public:
    // Correct - create new object:
    MyClass() : member_(std::make_shared<T>()) {}
    
    // Correct - share ownership:
    MyClass(std::shared_ptr<T> ptr) : member_(ptr) {}  // No move needed!
    
    // Correct - factory pattern:
    static std::shared_ptr<MyClass> create() {
        return std::make_shared<MyClass>();
    }
    
    // Easy sharing:
    std::shared_ptr<T> getMember() { return member_; }  // Returns shared ownership
    T* getMemberPtr() { return member_.get(); }         // Returns raw pointer for non-owning access
};
```

### Error: Circular Dependencies with shared_ptr

**Scenario:** Two objects reference each other, causing memory leaks

**❌ Problematic Pattern:**
```cpp
class Parent {
    std::shared_ptr<Child> child_;
};

class Child {
    std::shared_ptr<Parent> parent_;  // Circular reference!
};
```

**✅ Solution: Use weak_ptr**
```cpp
class Parent {
    std::shared_ptr<Child> child_;
};

class Child {
    std::weak_ptr<Parent> parent_;  // Breaks the cycle
    
    void doSomething() {
        if (auto parent = parent_.lock()) {  // Check if parent still exists
            parent->someMethod();
        }
    }
};
```

### Error: Qt Parent-Child System Conflicts

**Scenario:** Converting Qt widget pointers to smart pointers

**❌ Don't Convert Qt Widgets:**
```cpp
class MyWindow : public QWidget {
    std::unique_ptr<QPushButton> button_;  // WRONG!
public:
    MyWindow() {
        button_ = std::make_unique<QPushButton>(this);  // Qt parent handles deletion
    }
};
```

**✅ Keep Qt Widgets as Raw Pointers:**
```cpp
class MyWindow : public QWidget {
    QPushButton* button_;  // Correct - Qt parent system manages this
public:
    MyWindow() {
        button_ = new QPushButton(this);  // Qt will delete when MyWindow is destroyed
    }
};
```

### Systematic Conversion Strategy

**1. Identify Ownership Patterns**
```bash
# Find classes that manage resources
grep -rn "delete\|new.*(" ./src ./include | grep -v "_autogen"
```

**2. Convert in Dependency Order**
```bash
# Start with leaf classes (fewest dependencies)
# Work up to root classes (most dependencies)
```

**3. For Each Class:**

a) **Convert member variables:**
```cpp
T* member_ → std::shared_ptr<T> member_
```

b) **Update constructor:**
```cpp
: member_(new T()) → : member_(std::make_shared<T>())
```

c) **Remove destructor if only doing deletes:**
```cpp
~MyClass() { delete member_; }  // Remove entirely
```

d) **Update method implementations:**
```cpp
member_->method()     // No change needed
member_             → member_.get()  // When raw pointer needed
*member_              // When reference needed
```

e) **Update callers (easier with shared_ptr):**
- Most function calls can just pass the shared_ptr directly
- Use `.get()` only when raw pointer specifically needed
- Multiple ownership is safe and easy

**4. Build and Test After Each Class**
```bash
pixi run build  # Must pass before moving to next class
```

### Decision Tree for Each Pointer

```
Is this pointer owned by this class?
├── YES → Convert to std::shared_ptr<T> (easier than unique_ptr for most cases)
│   ├── Need guaranteed exclusive ownership? → Use std::unique_ptr<T>
│   ├── Qt widget with parent? → Keep as T* (Qt manages)
│   └── CUDA device memory? → Keep as T* or custom deleter
└── NO → Keep as T* (non-owning reference)
    ├── Function parameter? → Keep as T*
    ├── Temporary reference? → Keep as T*
    └── Observer pattern? → Consider T* or std::weak_ptr<T>
```

### Why shared_ptr is Often Easier

**Advantages for complex codebases:**
- No move semantics required - just copy normally
- Multiple owners naturally supported
- Easier to pass around without ownership transfer headaches
- Less refactoring of calling code needed
- Reference counting handles cleanup automatically

**When to prefer unique_ptr:**
- Single clear owner
- Performance critical (slightly less overhead)
- Want to enforce exclusive ownership at compile time

### Build Error Investigation Process

**When build fails:**

1. **Read the exact error message** - don't guess
2. **Identify the function signature** that's causing issues
3. **Determine if caller or callee needs to change**
4. **Apply the appropriate pattern from above**
5. **Test the fix in isolation**
6. **Continue to next error**

**Never randomly add `.get()` or `*` without understanding why!**
