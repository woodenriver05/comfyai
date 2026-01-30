"""
Smoke Test for ComfyUI RAG SDK
Tests both Sync and Async clients against localhost:7001
"""
import asyncio
import sys
from src.sdk import ComfyRagClient, AsyncComfyRagClient

BASE_URL = "http://localhost:7001"

def test_sync():
    print("🚀 [Sync] Starting Smoke Test...")
    with ComfyRagClient(BASE_URL) as client:
        # 1. Knowledge Ingest
        print("  - Ingesting doc...")
        ingest_res = client.knowledge_ingest(
            type="test",
            title="SDK Sync Test",
            content="Sync client test content with transaction keyword",
            tags=["sdk", "test"]
        )
        print(f"    ✅ Created: {ingest_res.doc_hash}")

        # 2. Knowledge Search
        print("  - Searching doc...")
        search_res = client.knowledge_search("Sync client test", limit=1)
        assert search_res.total > 0
        print(f"    ✅ Found: {search_res.results[0].title}")

        # 3. Workflow Search (MMR)
        print("  - Searching workflow (MMR)...")
        wf_res = client.search("anime", diversify=True, mmr_lambda=0.5)
        print(f"    ✅ Found {len(wf_res.results)} workflows")

        # 4. Recall Info (New)
        if wf_res.results:
            wf_id = wf_res.results[0].workflow_id
            print(f"  - Checking workflow info for {wf_id}...")
            wf_info = client.recall_info(wf_id)
            print(f"    ✅ Info: {wf_info.name or 'No Name'} (Success: {wf_info.success_count})")

    print("✅ [Sync] Test Passed!\n")

async def test_async():
    print("🚀 [Async] Starting Smoke Test...")
    async with AsyncComfyRagClient(BASE_URL) as client:
        # 1. Knowledge Ingest
        print("  - Ingesting doc...")
        ingest_res = await client.knowledge_ingest(
            type="test",
            title="SDK Async Test",
            content="Async client test content",
            tags=["sdk", "async"]
        )
        print(f"    ✅ Created: {ingest_res.doc_hash}")

        # 2. Recall Outbox
        print("  - Checking outbox...")
        outbox = await client.get_outbox(limit=5)
        print(f"    ✅ Outbox items: {len(outbox)}")

    print("✅ [Async] Test Passed!\n")

if __name__ == "__main__":
    try:
        test_sync()
        asyncio.run(test_async())
        print("🎉 ALL SDK TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        sys.exit(1)
