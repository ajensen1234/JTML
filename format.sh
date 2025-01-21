#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Print usage information
function print_usage() {
    echo "Usage: $0 [-d|--dry-run] [-v|--verbose]"
    echo "  -d, --dry-run   Don't actually format files, just show what would be done"
    echo "  -v, --verbose   Show verbose output"
}

# Parse command line arguments
DRY_RUN=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
    -d | --dry-run)
        DRY_RUN="-n"
        shift
        ;;
    -v | --verbose)
        VERBOSE="--verbose"
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

# Find all C++ files in both src and include directories
FILES=$(find ./src ./include -type f \( \
    -name "*.cpp" -o \
    -name "*.hpp" -o \
    -name "*.h" -o \
    -name "*.cc" -o \
    -name "*.cxx" \
    \) -not -path "*/build/*" \
    -not -path "*/cmake-build*/*" \
    2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo -e "${RED}No files found to format!${NC}"
    echo "Make sure you're running this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Count files
FILE_COUNT=$(echo "$FILES" | wc -l)

if [ -n "$VERBOSE" ]; then
    echo "Found $FILE_COUNT files to process:"
    echo "$FILES"
    echo "Current directory: $(pwd)"
fi

# Run clang-format
if [ -n "$DRY_RUN" ]; then
    echo "Dry run - showing what would be formatted..."
fi

echo "$FILES" | xargs clang-format -i $DRY_RUN $VERBOSE

if [ -z "$DRY_RUN" ]; then
    echo -e "${GREEN}Successfully formatted $FILE_COUNT files${NC}"
fi
