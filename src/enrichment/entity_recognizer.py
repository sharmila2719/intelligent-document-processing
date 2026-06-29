"""
Phase 4 – Document Enrichment: Named Entity Recognition
=========================================================
Use Amazon Comprehend built-in NER and custom entity recognizers
to extract named entities (people, places, dates, custom business terms)
from document text.

Reference:
  https://aws.amazon.com/blogs/machine-learning/part-2-intelligent-document-processing-with-aws-ai-services/
"""

import csv
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class EntityRecognizer:
    """
    Performs Named Entity Recognition (NER) on document text using:
      1. Amazon Comprehend built-in entities (PERSON, LOCATION, DATE, etc.)
      2. Custom entity recognizer (trained on domain-specific entities)

    Args:
        region (str): AWS region.
        bucket (str): S3 bucket for training data artefacts.
        role_arn (str): IAM role ARN that Comprehend assumes.
    """

    # Comprehend built-in entity types
    BUILTIN_ENTITY_TYPES = [
        "PERSON", "LOCATION", "ORGANIZATION", "COMMERCIAL_ITEM",
        "EVENT", "DATE", "QUANTITY", "TITLE", "OTHER",
    ]

    def __init__(
        self,
        region: str = "us-east-1",
        bucket: str = "",
        role_arn: str = "",
    ) -> None:
        self.region = region
        self.bucket = bucket
        self.role_arn = role_arn
        self.comprehend = boto3.client("comprehend", region_name=region)
        self.s3 = boto3.client("s3", region_name=region)

    # ------------------------------------------------------------------
    # Built-in NER
    # ------------------------------------------------------------------

    def detect_entities(
        self,
        text: str,
        language_code: str = "en",
    ) -> List[Dict[str, Any]]:
        """
        Detect built-in named entities from document text.

        Args:
            text: Plain text to analyse (max 5 000 UTF-8 bytes per call).
            language_code: Language code (default "en").

        Returns:
            List of entity dicts:
            [{"Text": ..., "Type": ..., "Score": ..., "BeginOffset": ...,
              "EndOffset": ...}]
        """
        # Comprehend has a 5 000 byte limit per synchronous call
        chunks = self._chunk_text(text, max_bytes=4800)
        all_entities: List[Dict[str, Any]] = []

        offset = 0
        for chunk in chunks:
            response = self.comprehend.detect_entities(
                LanguageCode=language_code,
                Text=chunk,
            )
            for entity in response.get("Entities", []):
                entity["BeginOffset"] += offset
                entity["EndOffset"] += offset
            all_entities.extend(response.get("Entities", []))
            offset += len(chunk)

        logger.info("detect_entities returned %d entities", len(all_entities))
        return all_entities

    def detect_entities_by_type(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        language_code: str = "en",
    ) -> Dict[str, List[str]]:
        """
        Detect entities and group them by type.

        Args:
            text: Document text.
            entity_types: Filter to specific entity types (default: all).
            language_code: Language code.

        Returns:
            Dict mapping entity type → list of entity text values.
        """
        entities = self.detect_entities(text, language_code)
        types_filter = set(entity_types or self.BUILTIN_ENTITY_TYPES)
        grouped: Dict[str, List[str]] = {t: [] for t in types_filter}

        for entity in entities:
            etype = entity.get("Type", "OTHER")
            if etype in types_filter:
                grouped[etype].append(entity["Text"])

        return grouped

    # ------------------------------------------------------------------
    # Custom entity recognizer training
    # ------------------------------------------------------------------

    def upload_entity_list(
        self,
        entity_list: List[Tuple[str, str]],
        s3_key: str,
    ) -> str:
        """
        Upload an entity list CSV to S3 for Comprehend custom NER training.

        The CSV format is: Text, Type
        e.g. "ACC-123456789", "SAVINGS_AC"

        Args:
            entity_list: List of (entity_text, entity_type) tuples.
            s3_key: S3 destination key.

        Returns:
            S3 URI of the uploaded CSV.
        """
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
        writer.writerow(["Text", "Type"])
        writer.writerows(entity_list)
        data = buf.getvalue().encode("utf-8")

        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=data)
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        logger.info("Entity list uploaded to %s (%d entries)", s3_uri, len(entity_list))
        return s3_uri

    def train_custom_ner(
        self,
        recognizer_name: str,
        entity_types: List[str],
        entity_list_s3_uri: str,
        documents_s3_uri: str,
        language_code: str = "en",
    ) -> str:
        """
        Train an Amazon Comprehend custom entity recognizer.

        Args:
            recognizer_name: Unique name for the recognizer.
            entity_types: List of entity type names, e.g. ["SAVINGS_AC", "CHECKING_AC"].
            entity_list_s3_uri: S3 URI of the entity list CSV.
            documents_s3_uri: S3 URI of the plain-text training documents.
            language_code: Language code.

        Returns:
            ARN of the entity recognizer being trained.
        """
        entity_types_config = [{"Type": t} for t in entity_types]

        response = self.comprehend.create_entity_recognizer(
            RecognizerName=recognizer_name,
            DataAccessRoleArn=self.role_arn,
            InputDataConfig={
                "DataFormat": "COMPREHEND_CSV",
                "EntityTypes": entity_types_config,
                "EntityList": {"S3Uri": entity_list_s3_uri},
                "Documents": {"S3Uri": documents_s3_uri},
            },
            LanguageCode=language_code,
        )
        arn = response["EntityRecognizerArn"]
        logger.info("Custom NER training started — ARN: %s", arn)
        return arn

    def wait_for_ner_training(self, recognizer_arn: str, poll_secs: int = 60) -> str:
        """Poll until custom NER training is done."""
        while True:
            response = self.comprehend.describe_entity_recognizer(
                EntityRecognizerArn=recognizer_arn
            )
            status = response["EntityRecognizerProperties"]["Status"]
            logger.info("NER training status: %s", status)
            if status in {"TRAINED", "FAILED", "STOPPED"}:
                return status
            time.sleep(poll_secs)

    def create_ner_endpoint(
        self,
        endpoint_name: str,
        recognizer_arn: str,
        inference_units: int = 1,
    ) -> str:
        """
        Create a real-time inference endpoint for a custom NER model.

        Args:
            endpoint_name: Unique endpoint name.
            recognizer_arn: Trained recognizer ARN.
            inference_units: Capacity units.

        Returns:
            Endpoint ARN.
        """
        try:
            response = self.comprehend.create_endpoint(
                EndpointName=endpoint_name,
                ModelArn=recognizer_arn,
                DesiredInferenceUnits=inference_units,
                DataAccessRoleArn=self.role_arn,
            )
            arn = response["EndpointArn"]
            logger.info("NER endpoint created: %s", arn)
            return arn
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ResourceInUseException":
                account_id = boto3.client("sts").get_caller_identity()["Account"]
                arn = (
                    f"arn:aws:comprehend:{self.region}:{account_id}"
                    f":entity-recognizer-endpoint/{endpoint_name}"
                )
                logger.warning("NER endpoint already exists: %s", arn)
                return arn
            raise

    # ------------------------------------------------------------------
    # Custom entity inference
    # ------------------------------------------------------------------

    def detect_custom_entities(
        self,
        text: str,
        endpoint_arn: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect custom entities from text using the trained NER endpoint.

        Args:
            text: Plain text to analyse.
            endpoint_arn: Custom NER endpoint ARN.

        Returns:
            List of entity dicts with Text, Type, Score.
        """
        response = self.comprehend.detect_entities(
            Text=text,
            EndpointArn=endpoint_arn,
        )
        entities = response.get("Entities", [])
        logger.info("Custom NER detected %d entities", len(entities))
        return entities

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, max_bytes: int = 4800) -> List[str]:
        """Split text into chunks that do not exceed max_bytes UTF-8 size."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for word in text.split():
            word_bytes = len(word.encode("utf-8")) + 1  # +1 for space
            if current_size + word_bytes > max_bytes and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_bytes
            else:
                current_chunk.append(word)
                current_size += word_bytes

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
