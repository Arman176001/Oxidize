import glob
import os
import re
import json

py_source = "data/Leetcode"
rust_source = "data/rustgym/leetcode/src"

py_source_path = os.path.abspath(py_source)
rust_source_path = os.path.abspath(rust_source)

output_data = []

import re

def remove_comments_from_code(code, language):
    if language == 'python':
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)
    elif language == 'rust':
        # Remove line and block comments
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove #[test] functions entirely
        code = re.sub(r'(?m)^#[ \t]*test[ \t]*\n[ \t]*fn[ \t]+\w+[ \t]*\([^)]*\)[ \t]*\{(?:[^{}]*|\{[^}]*\})*\}', '', code, flags=re.DOTALL)
        
    return code


for folder in next(os.walk(py_source_path))[1]:
    try:
        k = int(int(folder[-5:-1]) / 100)
        if k < 0:
            k = int(int(folder[-4:-1]) / 100)
        print(f"Processing folder: {folder}, k: {k}")
        
        py_folder_path = os.path.join(py_source_path, folder)
        rust_folder_pattern = os.path.join(rust_source_path, f'd{k-1}')
        rust_folders = glob.glob(rust_folder_pattern)
        
        if not rust_folders:
            print(f"No matching Rust folder for {folder}")
            continue
        
        rust_folder_path = rust_folders[0]
        
        found_count = 0
        skipped_count = 0
        skipped_files = []

        for py_file in os.listdir(py_folder_path):
            py_file_path = os.path.join(py_folder_path, py_file)
            if not os.path.isfile(py_file_path):
                continue

            with open(py_file_path, 'r') as py_f:
                py_code = py_f.read()

            # Remove comments from Python code
            py_code = remove_comments_from_code(py_code, 'python')

            # Normalize the Python file name for matching
            py_file_base = os.path.splitext(py_file)[0]  # Remove the .py extension
            rust_file_pattern = os.path.join(rust_folder_path, f"*_{py_file_base}*")  # Add leading underscore
            rust_files = glob.glob(rust_file_pattern)

            if not rust_files:
                skipped_count += 1
                skipped_files.append(py_file)
                continue

            rust_file_path = rust_files[0]
            with open(rust_file_path, 'r') as rust_f:
                rust_code = rust_f.read()

            # Remove comments from Rust code
            rust_code = remove_comments_from_code(rust_code, 'rust')

            output_data.append({
                "input": py_code,
                "output": rust_code
            })
            found_count += 1

        # Print summary
        print(f"Number of files found: {found_count}")
        print(f"Number of files skipped: {skipped_count}")
        if skipped_files:
            print("Skipped files:")
            for skipped_file in skipped_files:
                print(f"  - {skipped_file}")

    except ValueError:
        print(folder, "is not a true folder")
        continue

# Save the collected data to a JSON file
output_json_path = os.path.join(os.getcwd(), "code_mapping.json")
with open(output_json_path, 'w') as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Code mapping saved to {output_json_path}")