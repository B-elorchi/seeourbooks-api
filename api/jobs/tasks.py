import asyncio
import logging
from typing import Any, Dict

from api.celery_app import celery_app
from api.routes.pipeline_v2 import _ingest_then_run, V2PipelineReq
from api.services.db import find

log = logging.getLogger(__name__)

@celery_app.task(name="api.jobs.tasks.run_pipeline_task")
def run_pipeline_task(job_id: str, req_data: Dict[str, Any], previous_result: Any = None):
    """
    Celery task to run the full pipeline asynchronously.
    """
    log.info("Celery task started for job: %s", job_id)
    
    async def _runner():
        req = V2PipelineReq(**req_data)
        book_id = req.book_id
        
        # Try to resolve bid (numeric id if possible)
        try:
            bid = int(book_id)
        except ValueError:
            bid = book_id
            
        # Re-fetch book_row
        try:
            rows = await find("books", filters={"book_id": bid}, limit=1)
            book_row = rows[0] if rows else None
        except Exception as e:
            log.warning("Could not fetch book_row for %s: %s", bid, e)
            book_row = None
            
        await _ingest_then_run(
            job_id=job_id,
            book_id=book_id,
            bid=bid,
            book_row=book_row,
            req=req,
            previous_result=previous_result
        )

    # Run the async pipeline runner inside the synchronous celery worker
    asyncio.run(_runner())
    log.info("Celery task completed for job: %s", job_id)
