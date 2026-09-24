"""Fetches each service's latest documentation.md from S3-compatible storage
(MinIO locally, real S3 in prod — same client either way, just pointed at a
different endpoint via STORAGE_* env vars).

Replaces github_fetcher.py as the doc source. No local file is ever written —
content is read into memory and handed directly to the extraction functions.

RepoDocAI uploads to: <bucket>/RepoDocAI/documentation/<repo-slug>_<date>_documentation.md
Re-running doc generation on a new day creates a NEW file rather than
overwriting — so for each repo we have to find the most recent one ourselves.

Bucket name follows RepoDocAI's own convention: it's the tenant's bucket,
selected by TENANT_ID (falls back to STORAGE_AGENTS_BUCKET if TENANT_ID isn't set).
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import boto3

_PREFIX = "RepoDocAI/documentation/"
_PATTERN = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})_documentation\.md$")


def _client():
    """boto3 S3 client, pointed at MinIO (or real S3) via STORAGE_* env vars.

    common_lib.storage.storage_client is the SDK's own wrapper for this same
    setup, but it doesn't expose a list-objects operation — and we need to
    list to find the latest dated file per repo — so this talks to the
    endpoint directly instead.
    """
    endpoint = os.environ.get("STORAGE_ENDPOINT")
    if endpoint and not endpoint.startswith("http"):
        secure = os.environ.get("STORAGE_SECURE", "true").lower() == "true"
        endpoint = f"{'https' if secure else 'http'}://{endpoint}"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,  # None => real AWS S3; set => MinIO or other S3-compatible store
        aws_access_key_id=os.environ.get("STORAGE_ACCESS_KEY") or None,
        aws_secret_access_key=os.environ.get("STORAGE_SECRET_KEY") or None,
    )


def docs_bucket() -> str:
    return os.environ.get("TENANT_ID") or os.environ["STORAGE_AGENTS_BUCKET"]


def list_latest_docs(bucket: str) -> dict[str, str]:
    """Return {repo_slug: s3_key} for the most recent documentation.md per repo.

    Filters out *_answer.md files (a separate artifact RepoDocAI can also
    produce in the same folder) automatically, since they don't match _PATTERN.
    """
    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    docs_by_repo: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for page in paginator.paginate(Bucket=bucket, Prefix=_PREFIX):
        for obj in page.get("Contents", []):
            filename = obj["Key"].split("/")[-1]
            m = _PATTERN.match(filename)
            if not m:
                continue
            repo_slug, date_str = m.group(1), m.group(2)
            docs_by_repo[repo_slug].append((date_str, obj["Key"]))

    # ISO dates (YYYY-MM-DD) sort correctly as plain strings
    return {repo: max(entries, key=lambda x: x[0])[1] for repo, entries in docs_by_repo.items()}


def fetch_doc_from_s3(bucket: str, key: str) -> str:
    s3 = _client()
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
