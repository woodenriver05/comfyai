import os
import re
import argparse
import hashlib
import httpx
from pathlib import Path
from typing import List, Dict, Tuple

# Default Config
DEFAULT_API_URL = "http://localhost:7001/knowledge/ingest"
CHUNK_SIZE = 600  # Approx tokens
CHUNK_OVERLAP = 100

def parse_markdown(text: str) -> List[Tuple[str, str]]:
    """
    Split markdown by headers (##).
    Returns list of (title, content_chunk).
    """
    chunks = []
    lines = text.split('\n')
    current_title = "Introduction"
    current_content = []
    
    for line in lines:
        if line.startswith('## '):
            if current_content:
                chunks.append((current_title, '\n'.join(current_content)))
            current_title = line.strip('# ').strip()
            current_content = []
        elif line.startswith('# '):
            # Main title, just skip or use as context
            pass
        else:
            current_content.append(line)
            
    if current_content:
        chunks.append((current_title, '\n'.join(current_content)))
        
    return chunks

def ingest_file(client: httpx.Client, api_url: str, file_path: Path):
    print(f"📄 Processing: {file_path.name}")
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"❌ Read Error: {e}")
        return

    # 1. Determine Type & Tags from filename
    doc_type = "snippet"
    tags = []
    
    fname = file_path.name.upper()
    if "WORKLOG" in fname:
        doc_type = "worklog"
        tags.append("log")
    elif "ARCHITECTURE" in fname:
        doc_type = "architecture"
        tags.append("design")
    elif "GEMINI" in fname:
        doc_type = "persona"
        tags.append("instruction")
    
    # 2. Chunking
    chunks = parse_markdown(content)
    
    # 3. Upload Chunks
    success_cnt = 0
    for idx, (section_title, chunk_content) in enumerate(chunks):
        if not chunk_content.strip():
            continue
            
        payload = {
            "type": doc_type,
            "title": f"{file_path.stem} - {section_title}",
            "content": chunk_content,
            "source_path": str(file_path),
            "chunk_index": idx,
            "metadata": {"section": section_title},
            "tags": tags
        }
        
        try:
            resp = client.post(api_url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                success_cnt += 1
                # print(f"  - Chunk {idx}: OK")
            else:
                print(f"  - Chunk {idx}: HTTP {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"  - Chunk {idx}: Error {e}")

    print(f"✅ Uploaded {success_cnt}/{len(chunks)} chunks.")

def main():
    parser = argparse.ArgumentParser(description="Ingest Markdown documents into RAG Knowledge Base")
    parser.add_argument("path", help="File or Directory path")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API Endpoint URL")
    
    args = parser.parse_args()
    target_path = Path(args.path)
    
    if not target_path.exists():
        print(f"❌ Path not found: {target_path}")
        return

    with httpx.Client() as client:
        if target_path.is_file():
            ingest_file(client, args.url, target_path)
        elif target_path.is_dir():
            for f in target_path.glob("**/*.md"):
                ingest_file(client, args.url, f)

if __name__ == "__main__":
    main()
