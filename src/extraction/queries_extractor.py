"""
Phase 3 (extended) – Textract Queries
=======================================
Extract specific information from documents by asking natural-language
questions via Amazon Textract's QUERIES feature.

Example:
    queries = [
        {"Text": "What is the patient's date of birth?", "Alias": "DOB"},
        {"Text": "What is the total amount due?",        "Alias": "TOTAL"},
    ]
"""

import logging
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)


class TextractQueriesExtractor:
    """
    Uses Amazon Textract's AnalyzeDocument QUERIES feature to answer
    natural-language questions about a document.

    Args:
        region (str): AWS region.
        bucket (str): Default S3 bucket.
    """

    def __init__(self, region: str = "us-east-1", bucket: str = "") -> None:
        self.region = region
        self.bucket = bucket
        self.textract = boto3.client("textract", region_name=region)

    def query_document(
        self,
        queries: List[Dict[str, str]],
        s3_key: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        bucket: Optional[str] = None,
        additional_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run natural-language queries against a document.

        Args:
            queries: List of query dicts with keys "Text" (question string)
                     and optional "Alias" (short result label). Example:
                     [{"Text": "What is the invoice number?", "Alias": "INVOICE_NO"}]
            s3_key: S3 key of the source document.
            image_bytes: Raw bytes of the document image.
            bucket: S3 bucket override.
            additional_features: Other AnalyzeDocument feature types to enable
                                  alongside QUERIES (e.g., ["FORMS", "TABLES"]).

        Returns:
            Dict mapping alias (or query text) → extracted answer string.
        """
        document = self._build_document(s3_key, image_bytes, bucket)
        feature_types = list(set(["QUERIES"] + (additional_features or [])))

        response = self.textract.analyze_document(
            Document=document,
            FeatureTypes=feature_types,
            QueriesConfig={"Queries": queries},
        )

        answers = self._parse_query_answers(response, queries)
        logger.info("Queries completed — %d/%d answers found", len(answers), len(queries))
        return answers

    def query_document_with_pages(
        self,
        queries: List[Dict[str, str]],
        s3_key: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        bucket: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Convenience wrapper that returns answers as a flat list of dicts
        for easy tabular display.

        Returns:
            List of dicts: [{"query": ..., "alias": ..., "answer": ...,
                              "confidence": ...}]
        """
        document = self._build_document(s3_key, image_bytes, bucket)
        response = self.textract.analyze_document(
            Document=document,
            FeatureTypes=["QUERIES"],
            QueriesConfig={"Queries": queries},
        )
        return self._parse_query_answers_detailed(response, queries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_document(
        self,
        s3_key: Optional[str],
        image_bytes: Optional[bytes],
        bucket: Optional[str],
    ) -> Dict[str, Any]:
        if image_bytes is not None:
            return {"Bytes": image_bytes}
        if s3_key is not None:
            bkt = bucket or self.bucket
            return {"S3Object": {"Bucket": bkt, "Name": s3_key}}
        raise ValueError("Provide either s3_key or image_bytes")

    def _parse_query_answers(
        self,
        response: Dict[str, Any],
        queries: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """
        Extract query answers from the Textract response.

        Returns dict: {alias_or_query_text → answer_text}
        """
        blocks = {b["Id"]: b for b in response.get("Blocks", [])}
        answers: Dict[str, str] = {}

        for block in response.get("Blocks", []):
            if block.get("BlockType") != "QUERY":
                continue

            query_text = block.get("Query", {}).get("Text", "")
            alias = block.get("Query", {}).get("Alias", query_text)
            answer_text = ""

            # Find the QUERY_RESULT block linked to this QUERY
            for rel in block.get("Relationships", []):
                if rel["Type"] == "ANSWER":
                    for result_id in rel["Ids"]:
                        result_block = blocks.get(result_id, {})
                        if result_block.get("BlockType") == "QUERY_RESULT":
                            answer_text = result_block.get("Text", "")
                            break

            answers[alias] = answer_text

        return answers

    def _parse_query_answers_detailed(
        self,
        response: Dict[str, Any],
        queries: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Return answers as a list of detailed dicts including confidence."""
        blocks = {b["Id"]: b for b in response.get("Blocks", [])}
        results: List[Dict[str, str]] = []

        for block in response.get("Blocks", []):
            if block.get("BlockType") != "QUERY":
                continue

            query_text = block.get("Query", {}).get("Text", "")
            alias = block.get("Query", {}).get("Alias", query_text)
            answer_text = ""
            confidence = 0.0

            for rel in block.get("Relationships", []):
                if rel["Type"] == "ANSWER":
                    for result_id in rel["Ids"]:
                        result_block = blocks.get(result_id, {})
                        if result_block.get("BlockType") == "QUERY_RESULT":
                            answer_text = result_block.get("Text", "")
                            confidence = result_block.get("Confidence", 0.0)
                            break

            results.append(
                {
                    "query": query_text,
                    "alias": alias,
                    "answer": answer_text,
                    "confidence": f"{confidence:.1f}%",
                }
            )

        return results
