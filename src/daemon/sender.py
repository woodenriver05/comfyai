"""
ComfyUI Memory RAG - Sender Daemon
Polls recall_outbox and sends workflows to ComfyUI PC
"""
import asyncio
import asyncpg
import httpx
import os
import sys
import signal
import json
from datetime import datetime
from typing import Optional

# Config
POLL_INTERVAL = float(os.getenv("SENDER_POLL_INTERVAL", "5.0"))  # seconds
MAX_ATTEMPTS = int(os.getenv("SENDER_MAX_ATTEMPTS", "3"))
PROCESSING_TIMEOUT = float(os.getenv("SENDER_PROCESSING_TIMEOUT", "300"))
BATCH_SIZE = int(os.getenv("SENDER_BATCH_SIZE", "10"))
DATABASE_URL = os.getenv("DATABASE_URL")

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown"""
    global shutdown_flag
    print(f"\n🛑 Received signal {sig}. Shutting down gracefully...")
    shutdown_flag = True


async def send_to_comfyui(
    workflow_json: dict,
    target_host: str,
    request_id: str,
    client: httpx.AsyncClient,
) -> dict:
    """
    Send workflow to ComfyUI PC

    Args:
        workflow_json: Workflow JSON dict
        target_host: PC URL (e.g., http://192.168.1.100:8188)
        request_id: UUID for tracking
        client: Shared AsyncClient instance

    Returns:
        ComfyUI response dict

    Raises:
        httpx.HTTPError: On network/HTTP errors
    """
    url = f"{target_host.rstrip('/')}/prompt"

    payload = {
        "prompt": workflow_json,
        "client_id": str(request_id),
    }

    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


async def process_pending_items(pool: asyncpg.Pool, client: httpx.AsyncClient) -> int:
    """
    Process pending items in recall_outbox

    Returns:
        Number of items processed
    """
    async with pool.acquire() as conn:
        if PROCESSING_TIMEOUT > 0:
            stuck_rows = await conn.fetch(
                """
                UPDATE recall_outbox
                SET status = $$pending$$,
                    last_attempt_at = NOW(),
                    error_log = COALESCE(error_log, $$requeued after processing timeout$$)
                WHERE status = $$processing$$
                  AND (
                        (last_attempt_at IS NOT NULL AND last_attempt_at < NOW() - ($1 * interval $$1 second$$))
                     OR (last_attempt_at IS NULL AND created_at < NOW() - ($1 * interval $$1 second$$))
                  )
                RETURNING id
                """,
                PROCESSING_TIMEOUT,
            )
            if stuck_rows:
                print(f"↩️  Requeued {len(stuck_rows)} stuck items.")

        # Fetch and claim pending items
        rows = await conn.fetch(
            """
            UPDATE recall_outbox
            SET status = 'processing',
                last_attempt_at = NOW()
            WHERE id IN (
                SELECT id
                FROM recall_outbox
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, request_id, workflow_json, target_host, attempts
            """,
            BATCH_SIZE,
        )

        if not rows:
            return 0

        print(f"📦 Processing {len(rows)} pending items...")
        processed = 0

        for row in rows:
            if shutdown_flag:
                print("⚠️  Shutdown requested, stopping processing.")
                break

            item_id = row["id"]
            request_id = str(row["request_id"])
            workflow_json = row["workflow_json"]
            target_host = row["target_host"]
            attempts = row["attempts"]

            # Parse workflow_json if it's a string
            if isinstance(workflow_json, str):
                try:
                    workflow_json = json.loads(workflow_json)
                except json.JSONDecodeError as e:
                    # Invalid JSON, mark as failed
                    await conn.execute(
                        """
                        UPDATE recall_outbox
                        SET status = 'failed',
                            attempts = attempts + 1,
                            last_attempt_at = NOW(),
                            error_log = $1
                        WHERE id = $2
                        """,
                        f"Invalid JSON: {e}",
                        item_id,
                    )
                    print(f"❌ Invalid JSON for {item_id}: {e}")
                    processed += 1
                    continue

            try:
                # Send to ComfyUI
                result = await send_to_comfyui(
                    workflow_json,
                    target_host,
                    request_id,
                    client=client,
                )

                # Success
                await conn.execute(
                    """
                    UPDATE recall_outbox
                    SET status = 'sent',
                        sent_at = NOW(),
                        last_attempt_at = NOW(),
                        attempts = attempts + 1,
                        error_log = NULL
                    WHERE id = $1
                    """,
                    item_id,
                )

                print(f"✅ Sent {item_id} → {target_host}: {result}")
                processed += 1

            except Exception as e:
                new_attempts = attempts + 1

                if new_attempts >= MAX_ATTEMPTS:
                    new_status = "failed"
                    print(f"❌ Failed {item_id} (max attempts reached): {e}")
                else:
                    new_status = "pending"
                    print(f"⚠️  Retry {item_id} (attempt {new_attempts}/{MAX_ATTEMPTS}): {e}")

                await conn.execute(
                    """
                    UPDATE recall_outbox
                    SET status = $1,
                        attempts = $2,
                        last_attempt_at = NOW(),
                        error_log = $3
                    WHERE id = $4
                    """,
                    new_status,
                    new_attempts,
                    str(e)[:1000],  # Truncate error log
                    item_id,
                )
                processed += 1

        return processed


async def main_loop():
    """Main daemon loop"""
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set. Exiting.")
        sys.exit(1)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create connection pool
    print(f"🔌 Connecting to database...")
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    client = httpx.AsyncClient(timeout=30.0)

    print(f"🚀 Sender daemon started.")
    print(f"   Poll interval: {POLL_INTERVAL}s")
    print(f"   Max attempts: {MAX_ATTEMPTS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Press Ctrl+C to stop.\n")

    try:
        while not shutdown_flag:
            try:
                processed = await process_pending_items(pool, client)
                if processed > 0:
                    print(f"✨ Processed {processed} items.\n")
            except Exception as e:
                print(f"❌ Error in processing loop: {e}")

            # Sleep with shutdown check
            for _ in range(int(POLL_INTERVAL * 10)):
                if shutdown_flag:
                    break
                await asyncio.sleep(0.1)

        print("✅ Shutdown complete.")

    finally:
        await client.aclose()
        await pool.close()


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user.")
