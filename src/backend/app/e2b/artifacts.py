"""E2B file and artifact management functions."""

import os
import json
import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from e2b_code_interpreter import Sandbox


@dataclass
class FileEntry:
    path: str
    is_dir: bool
    size: Optional[int] = None


def extract_output_from_execution(execution) -> str:
    """Extract text output from E2B execution result."""
    if execution is None:
        return ""

    # The E2B SDK returns Execution objects with logs.stdout as a list
    if hasattr(execution, 'logs') and hasattr(execution.logs, 'stdout'):
        stdout_list = execution.logs.stdout
        if isinstance(stdout_list, list):
            return '\n'.join(stdout_list)
        else:
            return str(stdout_list) if stdout_list else ""

    # Fallback checks
    if hasattr(execution, 'text') and execution.text:
        return execution.text

    if hasattr(execution, 'stdout'):
        stdout = execution.stdout
        if isinstance(stdout, list):
            return '\n'.join(stdout)
        else:
            return str(stdout) if stdout else ""

    return str(execution) if execution else ""


def list_tree(sbx: Sandbox, root: str = "/home/user") -> List[Dict[str, Any]]:
    """
    Return a list[dict] with entries in 'root' (recursive).
    Uses a Python script executed via run_code to walk the filesystem.
    """
    # Use run_code to execute Python directly in the sandbox
    code = f'''
import os, json
root = {json.dumps(root)}
results = []
try:
    for dp, dns, fns in os.walk(root):
        for name in dns:
            p = os.path.join(dp, name)
            try:
                st = os.lstat(p)
                results.append({{"path": p, "type": "dir", "size": st.st_size, "mtime": int(st.st_mtime)}})
            except Exception as e:
                results.append({{"path": p, "type": "dir", "error": str(e)}})
        for name in fns:
            p = os.path.join(dp, name)
            try:
                st = os.lstat(p)
                results.append({{"path": p, "type": "file", "size": st.st_size, "mtime": int(st.st_mtime)}})
            except Exception as e:
                results.append({{"path": p, "type": "file", "error": str(e)}})
except Exception as e:
    results.append({{"path": root, "type": "error", "error": str(e)}})

print("RESULTS_START")
print(json.dumps(results))
print("RESULTS_END")
'''

    execution = sbx.run_code(code)
    output = extract_output_from_execution(execution)

    # Parse results from output
    lines = output.strip().split('\n')
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "RESULTS_START":
            start_idx = i + 1
        elif line.strip() == "RESULTS_END":
            end_idx = i
            break

    if start_idx != -1 and end_idx != -1:
        json_str = '\n'.join(lines[start_idx:end_idx])
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return []
    else:
        print("Failed to extract results from output")
        return []


def print_tree(entries: List[Dict[str, Any]], root: str = "/home/user"):
    """Pretty-print entries from list_tree output."""
    from os.path import relpath

    for e in sorted(entries, key=lambda x: x.get("path", "")):
        rel = relpath(e["path"], root) if e.get("path") else "?"
        suffix = f"  [ERR {e.get('error')}]" if e.get("error") else ""
        size_info = f" ({e.get('size', 0)} bytes)" if e.get('type') == 'file' else ""
        print(f"{(e.get('type') or '?'):4}  {rel}{size_info}{suffix}")


def download_all_as_tar(sbx: Sandbox, remote_root: str = "/home/user", local_tar_path: str = None) -> str:
    """Create a tar.gz in the sandbox and stream it locally. Preserves structure."""
    # Use a safe, writable local path in the current working directory
    if local_tar_path is None:
        local_tar_path = "artifacts/e2b_demo_project.tar.gz"

    local_tar_path = Path(local_tar_path)
    local_tar_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a gzip tar inside the sandbox using run_code with subprocess
    remote_tar = "/tmp/bundle.tar.gz"
    tar_code = f'''
import subprocess
import os
try:
    result = subprocess.run([
        "tar", "-czf", "{remote_tar}",
        "-C", "{remote_root}", "."
    ], capture_output=True, text=True, check=True)
    print(f"Tar created successfully: {{result.returncode}}")
except subprocess.CalledProcessError as e:
    print(f"Tar creation failed: {{e.stderr}}")
    raise
except Exception as e:
    print(f"Unexpected error: {{e}}")
    raise
'''

    result = sbx.run_code(tar_code)
    output = extract_output_from_execution(result)
    print(f"Tar creation output: {output}")

    # Now read the tar file and transfer it
    read_code = f'''
with open("{remote_tar}", "rb") as f:
    import base64
    data = f.read()
    encoded = base64.b64encode(data).decode()
    print("BASE64_START")
    print(encoded)
    print("BASE64_END")
'''

    read_result = sbx.run_code(read_code)
    read_output = extract_output_from_execution(read_result)

    # Extract base64 data
    lines = read_output.strip().split('\n')
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "BASE64_START":
            start_idx = i + 1
        elif line.strip() == "BASE64_END":
            end_idx = i
            break

    if start_idx != -1 and end_idx != -1:
        encoded_data = ''.join(lines[start_idx:end_idx])
        data = base64.b64decode(encoded_data)
        local_tar_path.write_bytes(data)
        return str(local_tar_path)
    else:
        raise RuntimeError("Failed to extract tar data from sandbox")


def download_folder_recursive(sbx: Sandbox, remote_root: str, local_root: str = "artifacts/e2b_demo_project") -> str:
    """Recursively mirror files from sandbox -> local path. Use when you need direct files.
    Prefer tar for large trees."""
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    entries = list_tree(sbx, root=remote_root)
    for entry in entries:
        if entry.get("error"):
            print(f"Skipping {entry['path']} due to error: {entry['error']}")
            continue

        rel_path = os.path.relpath(entry["path"], remote_root)
        local_path = local_root / rel_path

        if entry["type"] == "dir":
            local_path.mkdir(parents=True, exist_ok=True)
        elif entry["type"] == "file":
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Read file using run_code
                read_code = f'''
import base64
try:
    with open("{entry["path"]}", "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
        print("FILE_START")
        print(encoded)
        print("FILE_END")
except Exception as e:
    print(f"Error reading file: {{str(e)}}")
'''
                read_result = sbx.run_code(read_code)
                output = extract_output_from_execution(read_result)

                if output:
                    # Extract base64 encoded data
                    lines = output.strip().split('\n')
                    start_idx = -1
                    end_idx = -1
                    for i, line in enumerate(lines):
                        if line.strip() == "FILE_START":
                            start_idx = i + 1
                        elif line.strip() == "FILE_END":
                            end_idx = i
                            break

                    if start_idx != -1 and end_idx != -1:
                        encoded_data = ''.join(lines[start_idx:end_idx])
                        data = base64.b64decode(encoded_data)
                        local_path.write_bytes(data)
                    else:
                        print(f"Failed to extract data for {entry['path']}")
                else:
                    print(f"No output received for {entry['path']}")

            except Exception as e:
                print(f"Failed to download {entry['path']}: {e}")

    return str(local_root)