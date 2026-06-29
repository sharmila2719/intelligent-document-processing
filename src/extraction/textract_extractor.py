"""
Phase 3 – Document Extraction
==============================
Use Amazon Textract to extract:
  - Raw text (unstructured)
  - Tables (structured)
  - Forms / key-value pairs (semi-structured)
  - Expense documents (invoices & receipts)
  - Identity documents (driver's licenses, passports)

Reference:
  https://aws.amazon.com/blogs/machine-learning/part-1-intelligent-document-processing-with-aws-ai-services/
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class TextractExtractor:
    """
    High-level wrapper around Amazon Textract APIs.

    Supports both synchronous (single-page images) and asynchronous
    (multi-page PDFs) document processing.

    Args:
        region (str): AWS region.
        bucket (str): Default S3 bucket for documents.
    """

    def __init__(self, region: str = "us-east-1", bucket: str = "") -> None:
        self.region = region
        self.bucket = bucket
        self.textract = boto3.client("textract", region_name=region)

    # ------------------------------------------------------------------
    # Raw text detection
    # ------------------------------------------------------------------

    def detect_text(
        self,
        s3_key: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect raw text from a document using DetectDocumentText.

        Provide either (s3_key) or (image_bytes).

        Args:
            s3_key: S3 key of the document.
            image_bytes: Raw bytes of the document image.
            bucket: S3 bucket (overrides self.bucket).

        Returns:
            Textract API response dict.
        """
        document = self._build_document(s3_key, image_bytes, bucket)
        response = self.textract.detect_document_text(Document=document)
        logger.info("detect_document_text complete — %d blocks", len(response.get("Blocks", [])))
        return response

    def get_text_lines(self, response: Dict[str, Any]) -> List[str]:
        """Extract LINE blocks as a plain list of strings from a Textract response."""
        return [
            block["Text"]
            for block in response.get("Blocks", [])
            if block.get("BlockType") == "LINE"
        ]

    # ------------------------------------------------------------------
    # Tables extraction
    # ------------------------------------------------------------------

    def extract_tables(
        self,
        s3_key: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        bucket: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[List[List[str]]]]:
        """
        Extract all tables from a document using AnalyzeDocument with TABLES.

        Args:
            s3_key: S3 key of the document.
            image_bytes: Raw document bytes.
            bucket: S3 bucket override.

        Returns:
            Tuple of (raw API response, list of tables as 2D cell lists).
        """
        document = self._build_document(s3_key, image_bytes, bucket)
        response = self.textract.analyze_document(
            Document=document,
            FeatureTypes=["TABLES"],
        )
        tables = self._parse_tables(response)
        logger.info("Extracted %d table(s) from document", len(tables))
        return response, tables

    # ------------------------------------------------------------------
    # Forms / key-value pairs extraction
    # ------------------------------------------------------------------

    def extract_forms(
        self,
        s3_key: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        bucket: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Extract form key-value pairs using AnalyzeDocument with FORMS.

        Args:
            s3_key: S3 key of the document.
            image_bytes: Raw document bytes.
            bucket: S3 bucket override.

        Returns:
            Tuple of (raw API response, dict of {key: value}).
        """
        document = self._build_document(s3_key, image_bytes, bucket)
        response = self.textract.analyze_document(
            Document=document,
            FeatureTypes=["FORMS"],
        )
        kv_pairs = self._parse_kv_pairs(response)
        logger.info("Extracted %d key-value pair(s) from form", len(kv_pairs))
        return response, kv_pairs

    # ------------------------------------------------------------------
    # Expense documents (invoices & receipts)
    # ------------------------------------------------------------------

    def analyze_expense(
        self,
        s3_key: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process an invoice or receipt using AnalyzeExpense.

        Args:
            s3_key: S3 key of the document.
            image_bytes: Raw document bytes.
            bucket: S3 bucket override.

        Returns:
            Parsed expense summary dict containing summary fields and line items.
        """
        document = self._build_document(s3_key, image_bytes, bucket)
        response = self.textract.analyze_expense(Document=document)
        return self._parse_expense(response)

    # ------------------------------------------------------------------
    # Identity documents
    # ------------------------------------------------------------------

    def analyze_id(
        self,
        s3_key: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Extract fields from identity documents (driver's licence, passport)
        using AnalyzeID.

        Args:
            s3_key: S3 key of the document.
            image_bytes: Raw document bytes.
            bucket: S3 bucket override.

        Returns:
            Dict of {field_type: value}.
        """
        document = self._build_document(s3_key, image_bytes, bucket)
        response = self.textract.analyze_id(DocumentPages=[document])
        fields: Dict[str, str] = {}
        for doc in response.get("IdentityDocuments", []):
            for field in doc.get("IdentityDocumentFields", []):
                key = field.get("Type", {}).get("Text", "UNKNOWN")
                value = field.get("ValueDetection", {}).get("Text", "")
                fields[key] = value
        logger.info("AnalyzeID extracted %d field(s)", len(fields))
        return fields

    # ------------------------------------------------------------------
    # Async processing for multi-page PDFs
    # ------------------------------------------------------------------

    def start_async_text_detection(self, s3_key: str, bucket: Optional[str] = None) -> str:
        """
        Start an asynchronous Textract text detection job for a multi-page PDF.

        Args:
            s3_key: S3 key of the PDF.
            bucket: S3 bucket override.

        Returns:
            Textract job ID.
        """
        bkt = bucket or self.bucket
        response = self.textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bkt, "Name": s3_key}}
        )
        job_id = response["JobId"]
        logger.info("Started async text detection job: %s", job_id)
        return job_id

    def get_async_results(
        self,
        job_id: str,
        poll_interval: int = 5,
        max_attempts: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Poll an async Textract job until completion and return all result pages.

        Args:
            job_id: Textract job ID.
            poll_interval: Seconds between polls.
            max_attempts: Maximum number of polling attempts.

        Returns:
            List of Textract response pages.

        Raises:
            RuntimeError: If the job fails or times out.
        """
        for attempt in range(max_attempts):
            response = self.textract.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]
            logger.debug("Async job %s status: %s (attempt %d)", job_id, status, attempt + 1)

            if status == "SUCCEEDED":
                pages = [response]
                # Handle pagination
                while "NextToken" in response:
                    response = self.textract.get_document_text_detection(
                        JobId=job_id, NextToken=response["NextToken"]
                    )
                    pages.append(response)
                logger.info("Async job %s succeeded — %d pages", job_id, len(pages))
                return pages

            if status == "FAILED":
                raise RuntimeError(f"Textract async job {job_id} FAILED")

            time.sleep(poll_interval)

        raise RuntimeError(
            f"Textract async job {job_id} timed out after "
            f"{max_attempts * poll_interval}s"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_document(
        self,
        s3_key: Optional[str],
        image_bytes: Optional[bytes],
        bucket: Optional[str],
    ) -> Dict[str, Any]:
        """Build the Document parameter for Textract APIs."""
        if image_bytes is not None:
            return {"Bytes": image_bytes}
        if s3_key is not None:
            bkt = bucket or self.bucket
            return {"S3Object": {"Bucket": bkt, "Name": s3_key}}
        raise ValueError("Provide either s3_key or image_bytes")

    def _parse_tables(self, response: Dict[str, Any]) -> List[List[List[str]]]:
        """Parse Textract TABLES feature response into a list of 2D cell grids."""
        blocks = {b["Id"]: b for b in response.get("Blocks", [])}
        tables: List[List[List[str]]] = []

        for block in response.get("Blocks", []):
            if block["BlockType"] != "TABLE":
                continue

            # Build {(row, col): text} map
            cell_map: Dict[Tuple[int, int], str] = {}
            max_row = max_col = 0

            for rel in block.get("Relationships", []):
                if rel["Type"] != "CHILD":
                    continue
                for cell_id in rel["Ids"]:
                    cell = blocks.get(cell_id, {})
                    if cell.get("BlockType") != "CELL":
                        continue
                    row = cell.get("RowIndex", 1)
                    col = cell.get("ColumnIndex", 1)
                    max_row = max(max_row, row)
                    max_col = max(max_col, col)

                    # Collect text from WORD children of the CELL
                    words: List[str] = []
                    for child_rel in cell.get("Relationships", []):
                        if child_rel["Type"] == "CHILD":
                            for word_id in child_rel["Ids"]:
                                word_block = blocks.get(word_id, {})
                                if word_block.get("BlockType") in {"WORD", "SELECTION_ELEMENT"}:
                                    if word_block.get("BlockType") == "SELECTION_ELEMENT":
                                        words.append(word_block.get("SelectionStatus", ""))
                                    else:
                                        words.append(word_block.get("Text", ""))
                    cell_map[(row, col)] = " ".join(words)

            # Convert map to 2D list
            grid = [
                [cell_map.get((r, c), "") for c in range(1, max_col + 1)]
                for r in range(1, max_row + 1)
            ]
            tables.append(grid)

        return tables

    def _parse_kv_pairs(self, response: Dict[str, Any]) -> Dict[str, str]:
        """Parse Textract FORMS feature response into a {key: value} dict."""
        blocks = {b["Id"]: b for b in response.get("Blocks", [])}
        kv: Dict[str, str] = {}

        for block in response.get("Blocks", []):
            if block["BlockType"] != "KEY_VALUE_SET":
                continue
            if "KEY" not in block.get("EntityTypes", []):
                continue

            key_text = self._get_text_for_block(block, blocks)
            value_text = ""

            for rel in block.get("Relationships", []):
                if rel["Type"] == "VALUE":
                    for val_id in rel["Ids"]:
                        value_block = blocks.get(val_id, {})
                        value_text = self._get_text_for_block(value_block, blocks)

            if key_text:
                kv[key_text.strip()] = value_text.strip()

        return kv

    def _get_text_for_block(
        self,
        block: Dict[str, Any],
        all_blocks: Dict[str, Dict[str, Any]],
    ) -> str:
        """Collect concatenated text from a block's WORD children."""
        words: List[str] = []
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for child_id in rel["Ids"]:
                    child = all_blocks.get(child_id, {})
                    if child.get("BlockType") == "WORD":
                        words.append(child.get("Text", ""))
        return " ".join(words)

    def _parse_expense(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AnalyzeExpense response into a structured summary."""
        result: Dict[str, Any] = {"documents": []}

        for doc in response.get("ExpenseDocuments", []):
            doc_summary: Dict[str, Any] = {"summary_fields": {}, "line_items": []}

            # Summary fields (vendor name, total, tax, etc.)
            for field in doc.get("SummaryFields", []):
                field_type = field.get("Type", {}).get("Text", "UNKNOWN")
                value = field.get("ValueDetection", {}).get("Text", "")
                doc_summary["summary_fields"][field_type] = value

            # Line items
            for line_item_group in doc.get("LineItemGroups", []):
                for line_item in line_item_group.get("LineItems", []):
                    item: Dict[str, str] = {}
                    for field in line_item.get("LineItemExpenseFields", []):
                        field_type = field.get("Type", {}).get("Text", "UNKNOWN")
                        value = field.get("ValueDetection", {}).get("Text", "")
                        item[field_type] = value
                    doc_summary["line_items"].append(item)

            result["documents"].append(doc_summary)

        return result
