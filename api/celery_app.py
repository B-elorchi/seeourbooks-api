import os
from celery import Celery

# Create the celery app
# We use REDIS_URL from environment or fallback to localhost
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "seeourbook_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["api.jobs.tasks"] # We will create api/jobs/tasks.py for the actual tasks
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 2, # 2 hours max per task
)
