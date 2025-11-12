#!/usr/bin/env python3
import os
import json
import boto3
import mimetypes
from urllib.parse import urlparse
from typing import List
from utils.log import log_run

def _parse_s3_uri(s3_uri: str):
    """
    Parse s3://bucket/optional/prefix -> (bucket, prefix or '')
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri}. Expected s3://<bucket>/<optional/prefix>")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix.endswith("/"):
        prefix = prefix[:-1]
    return bucket, prefix

def upload(s3_uri: str, files: List[str]) -> List[str]:
    """
    Upload a list of local files to the exact S3 URI (bucket/folder) provided,
    and log the run to logs/<calling-script>.jsonl with IST timestamp.
    """
    bucket, prefix = _parse_s3_uri(s3_uri)
    s3_client = boto3.client("s3")

    uploaded_uris: List[str] = []
    actually_uploaded: List[str] = []

    try:
        for local_path in files:
            if not os.path.isfile(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")

            key = f"{prefix}/{os.path.basename(local_path)}" if prefix else os.path.basename(local_path)

            content_type, _ = mimetypes.guess_type(local_path)
            extra_args = {"ContentType": content_type} if content_type else None

            if extra_args:
                s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
            else:
                s3_client.upload_file(local_path, bucket, key)

            dest = f"s3://{bucket}/{key}"
            print(f"Uploaded {local_path} -> {dest}")
            uploaded_uris.append(dest)
            actually_uploaded.append(local_path)

        return uploaded_uris
    finally:
        log_entry = {
            "files": actually_uploaded,
            "s3_uri": s3_uri
        }
        log_run(log_entry)



if __name__ == "__main__":
    # Use this inside jupyter to find the right bucket
    # output_bucket = sagemaker.Session().default_bucket()
    ID="version-1"
    S3_URI = f"s3://sagemaker-us-east-1-466279506647/{ID}"
    FILES = ["template.json"]

    upload(S3_URI, FILES)
