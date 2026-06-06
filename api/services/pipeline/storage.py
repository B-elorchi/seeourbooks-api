"""
File upload to DigitalOcean Spaces or MinIO (S3-compatible).
Provider selected from settings.STORAGE_PROVIDER.

Bucket bootstrap
────────────────
The bucket is auto-created on first upload if it doesn't exist.  This is the
fix for the common "PutObject 404 — NoSuchBucket" error people hit when they
spin up a fresh MinIO container and forget to create the bucket manually.

For DigitalOcean Spaces the auto-create is silent if it already exists, and
on MinIO it's free + idempotent.  After successful creation we also set a
permissive public-read policy so the URLs we return can actually be loaded
from a browser without signed URLs.
"""
import json
import logging

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from api.config.settings import settings

log = logging.getLogger(__name__)

# Cache the "this bucket exists / has been ensured" decision per-process so
# we don't re-check on every upload.
_BUCKET_READY: set[str] = set()


def _get_client():
    if settings.STORAGE_PROVIDER == "spaces":
        return boto3.client(
            "s3",
            region_name=settings.DO_SPACES_REGION,
            endpoint_url=f"https://{settings.DO_SPACES_REGION}.digitaloceanspaces.com",
            aws_access_key_id=settings.DO_SPACES_KEY,
            aws_secret_access_key=settings.DO_SPACES_SECRET,
        )
    else:
        return boto3.client(
            "s3",
            endpoint_url=settings.MINIO_ENDPOINT,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )


def _bucket() -> str:
    return settings.DO_SPACES_BUCKET if settings.STORAGE_PROVIDER == "spaces" else settings.MINIO_BUCKET


def _public_url(key: str) -> str:
    if settings.STORAGE_PROVIDER == "spaces":
        base = settings.DO_SPACES_CDN_URL or f"https://{settings.DO_SPACES_BUCKET}.{settings.DO_SPACES_REGION}.digitaloceanspaces.com"
        return f"{base.rstrip('/')}/{key}"
    return f"{settings.MINIO_ENDPOINT.rstrip('/')}/{settings.MINIO_BUCKET}/{key}"


def _ensure_bucket(client, bucket: str) -> None:
    """
    Make sure the bucket exists and has a public-read policy.
    Cached per-process so we only do this work once after startup.

    Idempotent on both DigitalOcean Spaces and MinIO.  Failures here are
    logged but not raised — the upload that follows will produce a much
    clearer error if the bucket really can't be created.
    """
    if bucket in _BUCKET_READY:
        return

    # Step 1: HEAD the bucket to see if it exists.
    try:
        client.head_bucket(Bucket=bucket)
        _BUCKET_READY.add(bucket)
        return
    except ClientError as exc:
        status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        code   = exc.response.get("Error", {}).get("Code", "")
        if status not in (404, 301, 400) and code not in ("NoSuchBucket", "404", "NotFound"):
            # Some other failure (auth, network) — let the actual upload surface it
            log.warning("head_bucket(%s) returned %s/%s — proceeding anyway",
                        bucket, status, code)
            return

    # Step 2: bucket doesn't exist — try to create it.
    log.info("Bucket %r does not exist — creating it now", bucket)
    try:
        # Spaces requires no CreateBucketConfiguration; MinIO accepts the
        # empty body too.  We do NOT pass LocationConstraint to keep this
        # compatible with both backends.
        client.create_bucket(Bucket=bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            log.info("Bucket %r already exists — continuing", bucket)
        else:
            log.warning("Could not create bucket %r (%s) — upload will retry",
                        bucket, exc)
            return

    # Step 3: relax the bucket policy so the public URLs we return work
    # without signed URLs.  Best-effort — some Spaces accounts disable this.
    try:
        public_read_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Sid":       "PublicReadGetObject",
                "Effect":    "Allow",
                "Principal": "*",
                "Action":    ["s3:GetObject"],
                "Resource":  [f"arn:aws:s3:::{bucket}/*"],
            }],
        }
        client.put_bucket_policy(
            Bucket=bucket,
            Policy=json.dumps(public_read_policy),
        )
        log.info("Public-read policy applied to bucket %r", bucket)
    except ClientError as exc:
        log.info("Bucket policy not applied (%s) — set it manually if uploads "
                 "are not publicly readable", exc)

    _BUCKET_READY.add(bucket)


def upload_file(local_path: str, remote_key: str, content_type: str) -> str:
    """Upload a file and return its public CDN URL."""
    client = _get_client()
    bucket = _bucket()

    # Auto-create the bucket on first use (idempotent + cached).  This is the
    # fix for the "PutObject 404 NoSuchBucket" hit on fresh MinIO installs.
    _ensure_bucket(client, bucket)

    try:
        client.upload_file(
            local_path,
            bucket,
            remote_key,
            ExtraArgs={
                "ACL":          "public-read",
                "ContentType":  content_type,
                "CacheControl": "max-age=31536000, immutable",
            },
        )
    except Exception as exc:
        msg = str(exc)
        # Give the developer an actionable hint for the most common mistakes
        if "NoSuchBucket" in msg or "404" in msg:
            provider = settings.STORAGE_PROVIDER
            raise RuntimeError(
                f"Storage bucket '{bucket}' does not exist and auto-create failed. "
                f"Create it manually in your {'DigitalOcean Spaces' if provider == 'spaces' else 'MinIO'} "
                f"console and make sure {'DO_SPACES_BUCKET' if provider == 'spaces' else 'MINIO_BUCKET'} "
                f"in your .env matches the bucket name exactly."
            ) from exc
        if "InvalidAccessKeyId" in msg or "SignatureDoesNotMatch" in msg or "403" in msg:
            raise RuntimeError(
                f"Storage credentials rejected (bucket='{bucket}'). "
                f"Check {'DO_SPACES_KEY / DO_SPACES_SECRET' if settings.STORAGE_PROVIDER == 'spaces' else 'MINIO_ACCESS_KEY / MINIO_SECRET_KEY'} "
                f"in your .env file."
            ) from exc
        raise
    return _public_url(remote_key)


CONTENT_TYPES = {
    ".mp3":  "audio/mpeg",
    ".jpg":  "image/jpeg",
    ".png":  "image/png",
    ".svg":  "image/svg+xml",
    ".json": "application/json",
    ".epub": "application/epub+zip",
    ".mp4":  "video/mp4",
    ".srt":  "application/x-subrip",
}
