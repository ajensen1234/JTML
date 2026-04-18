#!/bin/bash
# Orchestrate the smart pointer conversion

set -e

TASK_FILE="smart_pointer_analysis/conversion_tasks.json"
RESULTS_DIR="conversion_results"
CONCURRENT_JOBS=3

mkdir -p "$RESULTS_DIR"

# Function to run a single conversion task
run_single_conversion() {
    local task_index=$1
    local class_name=$2
    local header_file=$3
    local source_file=$4
    local cuda_header_file=$5
    local cuda_source_file=$6
    
    echo "Starting conversion for class: $class_name"
    
    # Create task-specific prompt
    sed -e "s/{CLASS_NAME}/$class_name/g" \
        -e "s|{HEADER_FILE}|$header_file|g" \
        -e "s|{SOURCE_FILE}|$source_file|g" \
        -e "s|{CUDA_HEADER_FILE}|$cuda_header_file|g" \
        -e "s|{CUDA_SOURCE_FILE}|$cuda_source_file|g" \
        class_conversion_prompt.md > "$RESULTS_DIR/task_${task_index}_prompt.md"
    
    # Create result file
    echo "Task: $class_name" > "$RESULTS_DIR/task_${task_index}_result.txt"
    echo "Status: READY" >> "$RESULTS_DIR/task_${task_index}_result.txt"
    echo "Files: $header_file $source_file $cuda_header_file $cuda_source_file" >> "$RESULTS_DIR/task_${task_index}_result.txt"
}

# Process tasks in dependency order
python3 -c "
import json
with open('$TASK_FILE') as f:
    tasks = json.load(f)
for i, task in enumerate(tasks):
    files = task['files']
    header = files.get('header', '')
    source = files.get('source', '')  
    cuda_header = files.get('cuda_header', '')
    cuda_source = files.get('cuda_source', '')
    print(f'{i} {task[\"class_name\"]} {header} {source} {cuda_header} {cuda_source}')
" | while read line; do
    if [ -n "$line" ]; then
        run_single_conversion $line
    fi
done

echo "Generated $(ls $RESULTS_DIR/task_*_prompt.md | wc -l) conversion tasks"
echo "Each task is independent and can be assigned to different agents"