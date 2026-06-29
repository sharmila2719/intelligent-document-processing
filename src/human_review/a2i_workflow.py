"""
Phase 5 – Human Review & Validation
=====================================
Integrate Amazon Augmented AI (A2I) into the IDP pipeline to trigger
human review when document extraction confidence falls below a threshold.

The workflow:
  1. Check the confidence score(s) from Textract / Comprehend.
  2. If confidence < threshold, start a human review loop using A2I.
  3. Poll until reviewers complete the task.
  4. Retrieve the human-corrected output.

Reference:
  https://aws.amazon.com/blogs/machine-learning/part-2-intelligent-document-processing-with-aws-ai-services/
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)


class A2IHumanReview:
    """
    Manages Amazon A2I human review workflows for IDP.

    Use A2I to send documents (and their ML-extracted data) to a human
    workforce when extraction confidence is below an acceptable threshold.

    Args:
        region (str): AWS region.
        flow_definition_arn (str): ARN of the A2I flow definition.
        confidence_threshold (float): Trigger review if confidence < this value.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        flow_definition_arn: str = "",
        confidence_threshold: float = 0.90,
    ) -> None:
        self.region = region
        self.flow_definition_arn = flow_definition_arn
        self.confidence_threshold = confidence_threshold
        self.a2i = boto3.client(
            "sagemaker-a2i-runtime", region_name=region
        )

    # ------------------------------------------------------------------
    # Review trigger logic
    # ------------------------------------------------------------------

    def should_trigger_review(self, confidence: float) -> bool:
        """
        Determine whether a human review should be triggered based on
        the extraction confidence score.

        Args:
            confidence: Confidence score in range [0, 1].

        Returns:
            True if the score is below the configured threshold.
        """
        return confidence < self.confidence_threshold

    def evaluate_textract_results(
        self,
        kv_pairs: Dict[str, Any],
        confidence_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate Textract key-value extraction results and flag entries
        that fall below the confidence threshold.

        Args:
            kv_pairs: Dict of extracted {key: value} pairs.
            confidence_map: Optional dict of {key: confidence_score}.

        Returns:
            Dict with keys:
              - "review_needed": bool
              - "low_confidence_fields": list of field names needing review
              - "all_fields": the full kv_pairs dict
        """
        if not confidence_map:
            return {
                "review_needed": False,
                "low_confidence_fields": [],
                "all_fields": kv_pairs,
            }

        low_confidence = [
            key
            for key, score in confidence_map.items()
            if score < self.confidence_threshold
        ]
        return {
            "review_needed": len(low_confidence) > 0,
            "low_confidence_fields": low_confidence,
            "all_fields": kv_pairs,
        }

    # ------------------------------------------------------------------
    # Start human review loop
    # ------------------------------------------------------------------

    def start_human_loop(
        self,
        document_s3_uri: str,
        extracted_data: Dict[str, Any],
        human_loop_name: Optional[str] = None,
    ) -> str:
        """
        Start an Amazon A2I human review loop for a document.

        Args:
            document_s3_uri: S3 URI of the document to review.
            extracted_data: The ML-extracted data that needs human validation.
            human_loop_name: Optional unique loop name. Auto-generated if None.

        Returns:
            Human loop ARN.
        """
        if not self.flow_definition_arn:
            raise ValueError(
                "flow_definition_arn is not configured. "
                "Create an A2I flow definition first and set it in config.yaml."
            )

        loop_name = human_loop_name or f"idp-review-{uuid.uuid4().hex[:8]}"

        input_content = {
            "taskObject": document_s3_uri,
            "extractedData": extracted_data,
        }

        response = self.a2i.start_human_loop(
            HumanLoopName=loop_name,
            FlowDefinitionArn=self.flow_definition_arn,
            HumanLoopInput={"InputContent": json.dumps(input_content)},
        )
        loop_arn = response["HumanLoopArn"]
        logger.info("Human review loop started: %s (ARN=%s)", loop_name, loop_arn)
        return loop_arn

    # ------------------------------------------------------------------
    # Poll for completion
    # ------------------------------------------------------------------

    def wait_for_human_loop(
        self,
        human_loop_arn: str,
        poll_interval: int = 30,
        max_attempts: int = 60,
    ) -> Dict[str, Any]:
        """
        Poll an A2I human loop until a reviewer completes the task.

        Args:
            human_loop_arn: ARN of the human loop.
            poll_interval: Seconds between polls.
            max_attempts: Maximum polling attempts.

        Returns:
            Human loop status response dict.

        Raises:
            RuntimeError: If the loop fails or times out.
        """
        for attempt in range(max_attempts):
            response = self.a2i.describe_human_loop(HumanLoopArn=human_loop_arn)
            status = response["HumanLoopStatus"]
            logger.info(
                "Human loop status: %s (attempt %d/%d)",
                status,
                attempt + 1,
                max_attempts,
            )

            if status == "Completed":
                return response
            if status in {"Failed", "Stopped"}:
                raise RuntimeError(
                    f"Human loop {human_loop_arn} ended with status: {status}"
                )

            time.sleep(poll_interval)

        raise RuntimeError(
            f"Human loop {human_loop_arn} timed out after "
            f"{max_attempts * poll_interval}s"
        )

    # ------------------------------------------------------------------
    # Retrieve results
    # ------------------------------------------------------------------

    def get_human_review_results(
        self,
        human_loop_arn: str,
    ) -> Dict[str, Any]:
        """
        Retrieve the human-reviewed output from a completed A2I loop.

        Args:
            human_loop_arn: ARN of the completed human loop.

        Returns:
            Human review output dict parsed from the A2I output JSON.
        """
        response = self.a2i.describe_human_loop(HumanLoopArn=human_loop_arn)
        output_s3_uri = (
            response.get("HumanLoopOutput", {}).get("OutputS3Uri", "")
        )

        if not output_s3_uri:
            logger.warning("No output S3 URI found for loop %s", human_loop_arn)
            return {}

        # Download and parse the JSON output from S3
        s3 = boto3.client("s3", region_name=self.region)
        # Parse s3://bucket/key
        s3_path = output_s3_uri.replace("s3://", "")
        bucket, key = s3_path.split("/", 1)
        s3_response = s3.get_object(Bucket=bucket, Key=key)
        output_data = json.loads(s3_response["Body"].read().decode("utf-8"))

        logger.info("Retrieved human review output from %s", output_s3_uri)
        return output_data

    # ------------------------------------------------------------------
    # List human loops
    # ------------------------------------------------------------------

    def list_human_loops(
        self,
        status_filter: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        List A2I human loops for this flow definition.

        Args:
            status_filter: Filter by status ("InProgress", "Completed",
                           "Failed", "Stopped"). None = all.
            max_results: Maximum number of results to return.

        Returns:
            List of human loop summary dicts.
        """
        kwargs: Dict[str, Any] = {
            "FlowDefinitionArn": self.flow_definition_arn,
            "MaxResults": min(max_results, 100),
        }
        if status_filter:
            kwargs["StatusEquals"] = status_filter

        response = self.a2i.list_human_loops(**kwargs)
        loops = response.get("HumanLoopSummaries", [])
        logger.info("Listed %d human loop(s)", len(loops))
        return loops

    def stop_human_loop(self, human_loop_name: str) -> None:
        """Stop an in-progress human review loop."""
        self.a2i.stop_human_loop(HumanLoopName=human_loop_name)
        logger.info("Stopped human loop: %s", human_loop_name)
