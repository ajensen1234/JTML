#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path

def extract_class_files():
    """Map each class to its header and source files"""
    class_files = {}
    
    # Read the class list
    with open("smart_pointer_analysis/all_classes.txt", "r") as f:
        for line in f:
            if ":" in line:
                file_path, class_name = line.strip().split(":", 1)
                if class_name not in class_files:
                    class_files[class_name] = {"header": None, "source": None, "cuda_header": None, "cuda_source": None}
                
                if file_path.endswith((".h", ".hpp")):
                    class_files[class_name]["header"] = file_path
                elif file_path.endswith(".cuh"):
                    class_files[class_name]["cuda_header"] = file_path
                elif file_path.endswith((".cpp", ".cc", ".cxx")):
                    class_files[class_name]["source"] = file_path
                elif file_path.endswith(".cu"):
                    class_files[class_name]["cuda_source"] = file_path
    
    return class_files

def analyze_dependencies():
    """Find which classes depend on others (for ordering)"""
    # Simple dependency analysis - look for #include patterns
    dependencies = {}
    
    for root, dirs, files in os.walk("./src"):
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cu", ".cuh")):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', content)
                    dependencies[file_path] = includes
    
    return dependencies

def generate_conversion_tasks():
    """Generate individual tasks for each class"""
    class_files = extract_class_files()
    dependencies = analyze_dependencies()
    
    tasks = []
    
    for class_name, files in class_files.items():
        if any(files.values()):  # If any file exists for this class
            # Determine if this is a CUDA class
            is_cuda = files["cuda_header"] or files["cuda_source"]
            
            task = {
                "class_name": class_name,
                "files": files,
                "is_cuda": is_cuda,
                "priority": "high" if "manager" in class_name.lower() or "controller" in class_name.lower() else "normal",
                "estimated_complexity": "cuda" if is_cuda else "simple"
            }
            tasks.append(task)
    
    # Sort by dependency order (leaf classes first)
    # Simple heuristic: fewer dependencies = earlier
    def dependency_count(task):
        count = 0
        for file_path in [task["files"]["header"], task["files"]["source"], 
                         task["files"]["cuda_header"], task["files"]["cuda_source"]]:
            if file_path and file_path in dependencies:
                count += len(dependencies[file_path])
        return count
    
    tasks.sort(key=dependency_count)
    
    # Save tasks
    with open("smart_pointer_analysis/conversion_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Generated {len(tasks)} conversion tasks")
    return tasks

if __name__ == "__main__":
    tasks = generate_conversion_tasks()
    for i, task in enumerate(tasks[:5]):  # Show first 5
        print(f"Task {i+1}: {task['class_name']} - {task['files']}")