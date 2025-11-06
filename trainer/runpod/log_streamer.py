"""
Background log streaming from RunPod dstack runs
"""
import asyncio
import sys

from dstack.api import Client

from trainer.tasks import get_task, log_task
from trainer.utils.logging import logger


async def stream_runpod_logs(
    run_id: str,
    task_id: str,
    hotkey: str,
    log_labels: dict[str, str] | None = None,
):
    """
    Stream logs from a RunPod dstack run and update task logs in real-time.
    
    Args:
        run_id: dstack run ID
        task_id: Training task ID
        hotkey: Hotkey
        log_labels: Labels for logging
    """
    try:
        client = Client.from_config()
        run = client.runs.get(run_id)
        
        if not run:
            await log_task(task_id, hotkey, f"[ERROR] RunPod run {run_id} not found")
            return
        
        await log_task(task_id, hotkey, f"Starting log stream from RunPod run {run_id}")
        
        buffer = []
        buffer_size = 50  # Buffer logs before sending
        last_buffer_time = asyncio.get_event_loop().time()
        buffer_timeout = 5  # Flush buffer every 5 seconds
        
        while True:
            try:
                # Check if task still exists and is running
                task = get_task(task_id, hotkey)
                if not task or task.status.value != "training":
                    logger.info(f"Task {task_id} no longer running, stopping log stream", extra=log_labels)
                    break
                
                # Get all available logs (streaming)
                try:
                    logs = run.logs()
                    
                    for log_entry in logs:
                        if isinstance(log_entry, bytes):
                            log_text = log_entry.decode('utf-8', errors='ignore')
                        else:
                            log_text = str(log_entry)
                        
                        if log_text.strip():  # Only add non-empty logs
                            buffer.append(log_text)
                except Exception as log_err:
                    # If logs() fails, continue - might be transient
                    logger.debug(f"Could not fetch logs (will retry): {log_err}", extra=log_labels)
                
                current_time = asyncio.get_event_loop().time()
                
                # Flush buffer if it's full or timeout reached
                if len(buffer) >= buffer_size or (buffer and current_time - last_buffer_time >= buffer_timeout):
                    combined_log = "".join(buffer)
                    if combined_log.strip():
                        await log_task(task_id, hotkey, combined_log)
                    buffer.clear()
                    last_buffer_time = current_time
                
                # Check run status
                run.refresh()
                status_str = str(run.status).lower()
                
                if "done" in status_str or "failed" in status_str:
                    # Get final logs
                    try:
                        final_logs = run.logs()
                        final_text = ""
                        for log_entry in final_logs:
                            if isinstance(log_entry, bytes):
                                final_text += log_entry.decode('utf-8', errors='ignore')
                            else:
                                final_text += str(log_entry)
                        
                        if final_text:
                            await log_task(task_id, hotkey, f"\n--- Final RunPod Logs ---\n{final_text}")
                    except Exception as e:
                        logger.warning(f"Could not get final logs: {e}", extra=log_labels)
                    
                    await log_task(task_id, hotkey, f"RunPod run completed with status: {run.status}")
                    break
                
                # Wait before next check
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error streaming logs: {e}", extra=log_labels)
                await log_task(task_id, hotkey, f"[ERROR] Log streaming error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
                
    except Exception as e:
        error_msg = f"Log streaming failed: {e}"
        logger.error(error_msg, extra=log_labels)
        await log_task(task_id, hotkey, f"[ERROR] {error_msg}")


async def get_runpod_status(
    run_id: str,
) -> dict:
    """
    Get current status of a RunPod dstack run.
    
    Args:
        run_id: dstack run ID
        
    Returns:
        Dictionary with run status information
    """
    try:
        client = Client.from_config()
        run = client.runs.get(run_id)
        
        if not run:
            return {"error": f"Run {run_id} not found"}
        
        run.refresh()
        
        return {
            "run_id": run_id,
            "status": str(run.status),
            "created_at": str(run.created_at) if hasattr(run, 'created_at') else None,
        }
    except Exception as e:
        return {"error": str(e)}

