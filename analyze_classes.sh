#!/bin/bash
# Generate class-specific analysis for smart pointer conversion

BUILD_PATH=".build"
OUTPUT_DIR="smart_pointer_analysis"

mkdir -p "$OUTPUT_DIR"

# Get all classes that need smart pointer conversion
echo "Analyzing codebase for smart pointer opportunities..."

# Find all C++ and CUDA files and extract class names
find ./src ./include -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" | \
    xargs grep -l "class\|struct" | \
    while read file; do
        # Extract class/struct names from each file
        grep -n "^[[:space:]]*class\|^[[:space:]]*struct" "$file" | \
            sed 's/.*\(class\|struct\)[[:space:]]\+\([^[:space:]:{]*\).*/\2/' | \
            while read classname; do
                echo "$file:$classname"
            done
    done > "$OUTPUT_DIR/all_classes.txt"

# Run clang-tidy to get all raw pointer issues (excluding CUDA files for now)
echo "Running clang-tidy for raw pointer analysis..."
find ./src ./include -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | \
    xargs clang-tidy -p "$BUILD_PATH" \
    --checks="modernize-use-auto,modernize-make-*,cppcoreguidelines-owning-memory" \
    2>/dev/null | grep -E "warning:|error:" > "$OUTPUT_DIR/tidy_raw_output.txt"

# Separate analysis for CUDA files (clang-tidy may not work well with them)
echo "Analyzing CUDA files for raw pointer patterns..."
find ./src ./include -name "*.cu" -o -name "*.cuh" | \
    xargs grep -Hn "new\|delete\|\*.*=" 2>/dev/null > "$OUTPUT_DIR/cuda_raw_pointers.txt" || true

echo "Analysis complete. Check $OUTPUT_DIR/ for results."
echo "Note: CUDA files (.cu/.cuh) analyzed separately due to clang-tidy limitations"