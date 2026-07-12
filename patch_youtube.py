import re
import pathlib

p = pathlib.Path('api/routes/document.py')
text = p.read_text('utf-8')

youtube_code = '''

class YouTubeReq(BaseModel):
    url: str
    language: str = "en"
    steps: str = ""

@router.post("/youtube", status_code=202)
async def youtube_upload(
    req: YouTubeReq,
    background_tasks: BackgroundTasks,
    user: AuthUser | None = Depends(get_current_user),
):
    import re
    import uuid
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter

    # Extract video ID
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", req.url)
    if not match:
        raise HTTPException(400, "Invalid YouTube URL")
    video_id = match.group(1)

    try:
        # Fetch transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'ar'])
        formatter = TextFormatter()
        full_text = formatter.format_transcript(transcript_list)
    except Exception as e:
        log.exception("Failed to fetch YouTube transcript")
        raise HTTPException(400, f"Could not extract transcript: {e}")

    book_id = f"yt_{video_id}_{uuid.uuid4().hex[:4]}"
    title = f"YouTube Video {video_id}"
    steps_list = [s.strip() for s in req.steps.split(",") if s.strip()]

    placeholder_req = PipelineReq(
        book_id=book_id,
        title=title,
        language=req.language,
        steps=steps_list,
        options=PipelineOptions(length="10min", style="narrative"),
        source="youtube",
    )
    
    job_id = await create_job(book_id, placeholder_req.model_dump(), user_id=user.id if user else None)

    background_tasks.add_task(
        _run_youtube_pipeline,
        job_id, book_id, title, full_text, req.language, steps_list
    )

    return {
        "job_id": job_id,
        "book_id": book_id,
        "status": "queued",
        "status_url": f"/api/pipeline/status/{job_id}",
    }

async def _run_youtube_pipeline(
    job_id: str,
    book_id: str,
    title: str,
    full_text: str,
    language: str,
    steps_list: list[str],
) -> None:
    from api.services.usage_logger import set_job_context
    set_job_context(job_id)

    try:
        chapters = _split_chapters(full_text)
        if not chapters:
            chapters = [Chapter(index=1, title="Transcript", text=full_text)]
            
        req = PipelineReq(
            book_id=book_id,
            title=title,
            author="YouTube",
            language=language,
            chapters=chapters,
            steps=steps_list,
            options=PipelineOptions(length="10min", style="narrative"),
            source="youtube",
        )
        
        await db_update("pipeline_jobs", {"id": job_id}, {"input": req.model_dump()})
        await set_running(job_id)
        
        result = await run_pipeline(req, job_id=job_id)
        
        if result["status"] == "done":
            await set_done(job_id, result)
        elif result["status"] == "partial":
            await set_partial(job_id, result)
        else:
            await set_failed(job_id, str(result.get("errors", "unknown error")))
    except Exception as e:
        log.exception("youtube job %s failed", job_id)
        await set_failed(job_id, str(e))
'''

# Add BaseModel import if it doesn't exist
if 'from pydantic import BaseModel' not in text:
    text = text.replace('from pydantic import BaseModel', '') # clean up
    text = text.replace('from pydantic import', 'from pydantic import BaseModel,')
    if 'BaseModel' not in text: # If the whole import doesn't exist
        text = text.replace('from fastapi import', 'from pydantic import BaseModel\nfrom fastapi import')

text += youtube_code
p.write_text(text, 'utf-8')
print("OK")
