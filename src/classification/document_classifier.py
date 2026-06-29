"""
Phase 2 – Document Classification
==================================
Train and run an Amazon Comprehend custom multi-class document classifier
to categorise documents (bank statement, invoice, receipt, etc.).

Reference:
  https://aws.amazon.com/blogs/machine-learning/part-1-intelligent-document-processing-with-aws-ai-services/
"""

import csv
import io
import logging
import time
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Wraps Amazon Comprehend custom classification to:
      1. Prepare training data from S3-stored documents (via Textract).
      2. Train a custom multi-class classifier.
      3. Deploy a real-time inference endpoint.
      4. Classify new documents.

    Attributes:
        region (str): AWS region.
        bucket (str): S3 bucket for training artefacts and documents.
        role_arn (str): IAM role ARN that Comprehend assumes for training.
    """

    def __init__(self, region: str, bucket: str, role_arn: str) -> None:
        self.region = region
        self.bucket = bucket
        self.role_arn = role_arn
        self.comprehend = boto3.client("comprehend", region_name=region)
        self.textract = boto3.client("textract", region_name=region)
        self.s3 = boto3.client("s3", region_name=region)

    # ------------------------------------------------------------------
    # Step 1 – Prepare training data
    # ------------------------------------------------------------------

    def extract_text_from_document(self, s3_key: str) -> str:
        """
        Use Amazon Textract DetectDocumentText to extract raw text from
        a document stored in S3.

        Args:
            s3_key: S3 key of the source document.

        Returns:
            Extracted plain text string.
        """
        try:
            response = self.textract.detect_document_text(
                Document={"S3Object": {"Bucket": self.bucket, "Name": s3_key}}
            )
            lines: List[str] = [
                block["Text"]
                for block in response.get("Blocks", [])
                if block["BlockType"] == "LINE"
            ]
            return " ".join(lines)
        except ClientError as exc:
            logger.error("Textract error for %s: %s", s3_key, exc)
            raise

    def build_training_csv(
        self,
        labeled_documents: List[Tuple[str, str]],
        output_s3_key: str,
    ) -> str:
        """
        Build a Comprehend CSV training file from labeled documents.

        Args:
            labeled_documents: List of (s3_key, label) tuples, e.g.
                               [("idp/raw/invoice1.pdf", "invoice"), ...]
            output_s3_key: S3 key where the CSV file will be saved.

        Returns:
            S3 URI of the uploaded CSV training file.
        """
        rows: List[List[str]] = []
        for s3_key, label in labeled_documents:
            logger.info("Extracting text from %s (label=%s)", s3_key, label)
            text = self.extract_text_from_document(s3_key)
            # Comprehend CSV format: label, text
            rows.append([label, text])

        # Write CSV to a string buffer and upload to S3
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)
        csv_bytes = buf.getvalue().encode("utf-8")

        self.s3.put_object(
            Bucket=self.bucket,
            Key=output_s3_key,
            Body=csv_bytes,
            ContentType="text/csv",
        )
        s3_uri = f"s3://{self.bucket}/{output_s3_key}"
        logger.info("Training CSV uploaded to %s (%d rows)", s3_uri, len(rows))
        return s3_uri

    # ------------------------------------------------------------------
    # Step 2 – Train classifier
    # ------------------------------------------------------------------

    def train_classifier(
        self,
        classifier_name: str,
        version_name: str,
        training_data_s3_uri: str,
        output_s3_uri: str,
        language_code: str = "en",
        mode: str = "MULTI_CLASS",
    ) -> str:
        """
        Create an Amazon Comprehend custom document classifier training job.

        Args:
            classifier_name: Unique name for the classifier.
            version_name: Version label.
            training_data_s3_uri: S3 URI to the training CSV.
            output_s3_uri: S3 URI prefix for Comprehend output artefacts.
            language_code: Language of documents (default: "en").
            mode: "MULTI_CLASS" or "MULTI_LABEL".

        Returns:
            ARN of the document classifier being trained.
        """
        try:
            response = self.comprehend.create_document_classifier(
                DocumentClassifierName=classifier_name,
                VersionName=version_name,
                DataAccessRoleArn=self.role_arn,
                InputDataConfig={
                    "DataFormat": "COMPREHEND_CSV",
                    "S3Uri": training_data_s3_uri,
                },
                OutputDataConfig={"S3Uri": output_s3_uri},
                LanguageCode=language_code,
                Mode=mode,
            )
            arn = response["DocumentClassifierArn"]
            logger.info("Classifier training started. ARN: %s", arn)
            return arn
        except ClientError as exc:
            logger.error("Failed to start classifier training: %s", exc)
            raise

    def wait_for_classifier(self, classifier_arn: str, poll_secs: int = 60) -> str:
        """
        Poll until the Comprehend classifier training job is complete.

        Args:
            classifier_arn: ARN of the classifier.
            poll_secs: Polling interval in seconds.

        Returns:
            Final status ("TRAINED" or "FAILED").
        """
        while True:
            response = self.comprehend.describe_document_classifier(
                DocumentClassifierArn=classifier_arn
            )
            status = response["DocumentClassifierProperties"]["Status"]
            logger.info("Classifier status: %s", status)
            if status in {"TRAINED", "FAILED", "DELETING", "STOP_REQUESTED", "STOPPED"}:
                return status
            time.sleep(poll_secs)

    # ------------------------------------------------------------------
    # Step 3 – Deploy endpoint
    # ------------------------------------------------------------------

    def create_endpoint(
        self,
        endpoint_name: str,
        classifier_arn: str,
        inference_units: int = 1,
    ) -> str:
        """
        Create a real-time inference endpoint for the trained classifier.

        Args:
            endpoint_name: Unique name for the endpoint.
            classifier_arn: Trained classifier ARN.
            inference_units: Number of inference units (capacity).

        Returns:
            Endpoint ARN.
        """
        try:
            response = self.comprehend.create_endpoint(
                EndpointName=endpoint_name,
                ModelArn=classifier_arn,
                DesiredInferenceUnits=inference_units,
                DataAccessRoleArn=self.role_arn,
            )
            endpoint_arn = response["EndpointArn"]
            logger.info("Classifier endpoint created: %s", endpoint_arn)
            return endpoint_arn
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ResourceInUseException":
                # Endpoint already exists — return its ARN
                account_id = boto3.client("sts").get_caller_identity()["Account"]
                endpoint_arn = (
                    f"arn:aws:comprehend:{self.region}:{account_id}"
                    f":document-classifier-endpoint/{endpoint_name}"
                )
                logger.warning("Endpoint already exists: %s", endpoint_arn)
                return endpoint_arn
            raise

    # ------------------------------------------------------------------
    # Step 4 – Classify documents
    # ------------------------------------------------------------------

    def classify_document(
        self,
        text: str,
        endpoint_arn: str,
    ) -> Dict[str, float]:
        """
        Classify a document using the real-time Comprehend endpoint.

        Args:
            text: Extracted text of the document to classify.
            endpoint_arn: Comprehend classifier endpoint ARN.

        Returns:
            Dictionary mapping document class → confidence score,
            sorted by descending confidence.
        """
        response = self.comprehend.classify_document(
            Text=text,
            EndpointArn=endpoint_arn,
        )
        classes = {
            c["Name"]: round(c["Score"], 4)
            for c in response.get("Classes", [])
        }
        # Sort by score descending
        sorted_classes = dict(
            sorted(classes.items(), key=lambda item: item[1], reverse=True)
        )
        logger.info("Classification result: %s", sorted_classes)
        return sorted_classes

    def get_top_class(
        self,
        text: str,
        endpoint_arn: str,
    ) -> Tuple[str, float]:
        """
        Return the top predicted document class and its confidence score.

        Args:
            text: Document text.
            endpoint_arn: Comprehend endpoint ARN.

        Returns:
            Tuple of (class_name, confidence_score).
        """
        classes = self.classify_document(text, endpoint_arn)
        top_class, top_score = next(iter(classes.items()))
        logger.info("Top class: %s (score=%.4f)", top_class, top_score)
        return top_class, top_score

    def delete_endpoint(self, endpoint_arn: str) -> None:
        """Delete a Comprehend inference endpoint to stop incurring charges."""
        self.comprehend.delete_endpoint(EndpointArn=endpoint_arn)
        logger.info("Deleted endpoint: %s", endpoint_arn)
