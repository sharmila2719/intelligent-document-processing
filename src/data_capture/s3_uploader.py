"""
Phase 1 – Data Capture
======================
Upload documents to Amazon S3 for downstream IDP processing.
Supports PDF, PNG, JPEG, TIFF formats.

Reference:
  https://aws.amazon.com/blogs/machine-learning/part-1-intelligent-document-processing-with-aws-ai-services/
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3DocumentUploader:
    """
    Uploads local documents to an Amazon S3 bucket for IDP processing.

    Attributes:
        bucket (str): Target S3 bucket name.
        prefix (str): S3 key prefix (folder) under which documents are stored.
        region (str): AWS region.
        s3_client: Boto3 S3 client.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(
        self,
        bucket: str,
        prefix: str = "idp/raw/",
        region: str = "us-east-1",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.region = region
        self.s3_client = boto3.client("s3", region_name=region)
        logger.info(
            "S3DocumentUploader initialised — bucket=%s prefix=%s",
            self.bucket,
            self.prefix,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_document(self, local_path: str, s3_key: Optional[str] = None) -> str:
        """
        Upload a single document to S3.

        Args:
            local_path: Absolute or relative local file path.
            s3_key: Optional explicit S3 key. Defaults to prefix + filename.

        Returns:
            Full S3 URI (s3://bucket/key) of the uploaded object.

        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If local_path does not exist.
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if s3_key is None:
            s3_key = self.prefix + path.name

        try:
            self.s3_client.upload_file(str(path), self.bucket, s3_key)
            s3_uri = f"s3://{self.bucket}/{s3_key}"
            logger.info("Uploaded %s → %s", local_path, s3_uri)
            return s3_uri
        except ClientError as exc:
            logger.error("Failed to upload %s: %s", local_path, exc)
            raise

    def upload_directory(self, directory: str) -> List[str]:
        """
        Recursively upload all supported documents in a directory to S3.

        Args:
            directory: Local directory path.

        Returns:
            List of S3 URIs for successfully uploaded files.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        uploaded: List[str] = []
        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                # Preserve relative sub-directory structure in S3
                relative = file_path.relative_to(dir_path)
                s3_key = self.prefix + str(relative).replace("\\", "/")
                try:
                    uri = self.upload_document(str(file_path), s3_key=s3_key)
                    uploaded.append(uri)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping %s: %s", file_path, exc)

        logger.info("Uploaded %d document(s) from %s", len(uploaded), directory)
        return uploaded

    def list_documents(self, prefix: Optional[str] = None) -> List[str]:
        """
        List all documents stored in the S3 bucket under the given prefix.

        Args:
            prefix: S3 prefix to list. Defaults to self.prefix.

        Returns:
            List of S3 keys.
        """
        search_prefix = prefix or self.prefix
        keys: List[str] = []

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=search_prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        logger.info(
            "Found %d object(s) under s3://%s/%s",
            len(keys),
            self.bucket,
            search_prefix,
        )
        return keys

    def download_document(self, s3_key: str, local_dir: str = "/tmp") -> str:
        """
        Download a document from S3 to a local directory.

        Args:
            s3_key: S3 object key.
            local_dir: Local directory to save the file.

        Returns:
            Local file path of the downloaded document.
        """
        filename = os.path.basename(s3_key)
        local_path = os.path.join(local_dir, filename)

        os.makedirs(local_dir, exist_ok=True)
        self.s3_client.download_file(self.bucket, s3_key, local_path)
        logger.info("Downloaded s3://%s/%s → %s", self.bucket, s3_key, local_path)
        return local_path
