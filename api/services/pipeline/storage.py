"""
File upload to DigitalOcean Spaces or MinIO (S3-compatible).
Provider selected from settings.STORAGE_PROVIDER.
"""
import boto3
from botocore.config import Config
from api.config.settings import settings


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


def upload_file(local_path: str, remote_key: str, content_type: str) -> str:
    """Upload a file and return its public CDN URL."""
    client = _get_client()
    bucket = _bucket()
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
        if "NoSuchBucket" in msg:
            provider = settings.STORAGE_PROVIDER
            raise RuntimeError(
                f"Storage bucket '{bucket}' does not exist. "
                f"Create it in your {'DigitalOcean Spaces' if provider == 'spaces' else 'MinIO'} dashboard "
                f"and make sure DO_SPACES_BUCKET / MINIO_BUCKET in .env matches the bucket name."
            ) from exc
        if "InvalidAccessKeyId" in msg or "SignatureDoesNotMatch" in msg or "403" in msg:
            raise RuntimeError(
                f"Storage credentials rejected (bucket='{bucket}'). "
                f"Check DO_SPACES_KEY / DO_SPACES_SECRET in your .env file."
            ) from exc
        raise
    return _public_url(remote_key)


CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}
