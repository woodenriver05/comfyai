import os
import json
import httpx
import hashlib
from pathlib import Path

# --- Configuration ---
API_URL = "http://localhost:7001/ingest/workflow"
DATA_DIR = Path("data/external_workflows")
EXTENSIONS = [".json"]

def get_client_run_id(file_path: Path) -> str:
    """Generate a unique but stable ID based on file path."""
    path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
    return f"bulk-{file_path.stem}-{path_hash}"

def extract_tags(file_path: Path) -> list:
    """Extract tags from directory structure."""
    # Relative to DATA_DIR
    rel_path = file_path.relative_to(DATA_DIR)
    tags = list(rel_path.parts[:-1]) # Folder names as tags
    return tags

def ingest_file(client: httpx.Client, file_path: Path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            workflow_json = json.load(f)
        
        # Basic validation: ComfyUI workflows are usually dicts
        if not isinstance(workflow_json, dict):
            return False, "Not a valid dict"

        payload = {
            "client_run_id": get_client_run_id(file_path),
            "workflow_json": workflow_json,
            "workflow_name": file_path.stem,
            "tags": extract_tags(file_path),
            "status": "success"
        }

        response = client.post(API_URL, json=payload, timeout=30.0)
        
        if response.status_code == 200:
            return True, response.json().get("status")
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return False, str(e)

def main():
    print(f"🚀 Starting Bulk Ingest from: {DATA_DIR}")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0

    with httpx.Client() as client:
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if any(file.endswith(ext) for ext in EXTENSIONS):
                    file_path = Path(root) / file
                    
                    # Skip very large files or non-workflow JSONs if needed
                    # For now, try all .json
                    
                    ok, msg = ingest_file(client, file_path)
                    
                    if ok:
                        success_count += 1
                        print(f"✅ [{success_count}] {file_path.name} -> {msg}")
                    else:
                        fail_count += 1
                        print(f"❌ ERR: {file_path.name} -> {msg}")
                else:
                    # Non-json files
                    pass

    print("\n" + "="*30)
    print(f"📊 Ingest Task Complete")
    print(f"   - Success: {success_count}")
    print(f"   - Failed:  {fail_count}")
    print("="*30)

if __name__ == "__main__":
    main()
