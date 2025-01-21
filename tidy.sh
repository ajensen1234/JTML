#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

function print_usage() {
    echo "Usage: $0 [-f|--fix] [-p|--path build-dir] [--explain-config] [--dump-config] [--verify-config]"
    echo "  -f, --fix           Apply suggested fixes"
    echo "  -p, --path          Path to compile_commands.json directory"
    echo "  --explain-config    Explain which configuration was applied"
    echo "  --dump-config       Show effective configuration"
    echo "  --verify-config     Verify .clang-tidy configuration"
    echo "  --fix-errors        Apply fixes even if there are compilation errors"
    echo "  --quiet            Suppress output about ignored warnings"
}

FIX=""
BUILD_PATH=".build"
EXPLAIN_CONFIG=""
DUMP_CONFIG=""
VERIFY_CONFIG=""
FIX_ERRORS=""
QUIET=""

while [[ $# -gt 0 ]]; do
    case $1 in
    -f | --fix)
        FIX="--fix"
        shift
        ;;
    --fix-errors)
        FIX="--fix-errors"
        shift
        ;;
    -p | --path)
        BUILD_PATH="$2"
        shift 2
        ;;
    --explain-config)
        EXPLAIN_CONFIG="--explain-config"
        shift
        ;;
    --dump-config)
        DUMP_CONFIG="1"
        shift
        ;;
    --verify-config)
        VERIFY_CONFIG="1"
        shift
        ;;
    --quiet)
        QUIET="--quiet"
        shift
        ;;
    -h | --help)
        print_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        print_usage
        exit 1
        ;;
    esac
done

# Verify configuration if requested
if [ -n "$VERIFY_CONFIG" ]; then
    clang-tidy --verify-config
    exit $?
fi

# Dump configuration if requested
if [ -n "$DUMP_CONFIG" ]; then
    clang-tidy --dump-config
    exit $?
fi

# Check for compilation database
if [ ! -f "${BUILD_PATH}/compile_commands.json" ]; then
    echo -e "${RED}Error: compile_commands.json not found in ${BUILD_PATH}!${NC}"
    echo "Run CMake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first"
    exit 1
fi

# Find files
FILES=$(find ./src -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" \) \
    -not -path "*/build/*" -not -path "*/cmake-build*/*")

if [ -z "$FILES" ]; then
    echo -e "${RED}No files found to analyze!${NC}"
    exit 1
fi

# Export fixes to YAML if fixing
EXPORT_FIXES=""
if [ -n "$FIX" ]; then
    EXPORT_FIXES="--export-fixes=clang-tidy-fixes.yaml"
fi

# Count files
FILE_COUNT=$(echo "$FILES" | wc -l)
echo "Found $FILE_COUNT files to analyze"

# Run clang-tidy
echo "Running clang-tidy..."

echo "$FILES" | xargs -P $(nproc) -I{} clang-tidy {} \
    -p "${BUILD_PATH}" \
    $FIX \
    $EXPORT_FIXES \
    $EXPLAIN_CONFIG \
    $QUIET \
    --format-style=file

if [ -f "clang-tidy-fixes.yaml" ]; then
    echo -e "${GREEN}Fixes have been exported to clang-tidy-fixes.yaml${NC}"
fi
