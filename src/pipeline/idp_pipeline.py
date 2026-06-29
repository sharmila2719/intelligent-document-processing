"""
End-to-End IDP Pipeline Orchestrator
=====================================
Ties together all IDP phases into a single configurable pipeline:

  Phase 1 – Data Capture       (S3 upload)
  Phase 2 – Classification     (Comprehend custom classifier)
  Phase 3 – Extraction         (Textract text/tables/forms/expense/ID)
  Phase 3b– Queries            (Textract natural-language queries)
  Phase 4 – Enrichment         (NER, PII redaction, Comprehend Medical)
  Phase 5 – GenAI Q&A          (Bedrock + LangChain RAG)
  Phase 6 – Human Review       (A2I human-in-the-loop)

Usage:
    pipeline = IDPPipeline.from_config("config/config.yaml")
    result = pipeline.process_document("path/to/document.pdf")
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from src.classification.document_classifier import DocumentClassifier
from src.data_capture.s3_uploader import S3DocumentUploader
from src.enrichment.entity_recognizer import EntityRecognizer
from src.enrichment.medical_extractor import MedicalExtractor
from src.enrichment.pii_redactor import PIIRedactor
from src.extraction.queries_extractor import TextractQueriesExtractor
from src.extraction.textract_extractor import TextractExtractor
from src.genai.bedrock_langchain_qa import BedrockDocumentQA
from src.human_review.a2i_workflow import A2IHumanReview

logger = logging.getLogger(__name__)


@dataclass
class IDPResult:
    """Container for IDP pipeline results for a single document."""

    document_path: str
    s3_uri: str = ""
    document_type: str = ""
    classification_confidence: float = 0.0
    raw_text: str = ""
    tables: List[Any] = field(default_factory=list)
    kv_pairs: Dict[str, str] = field(default_factory=dict)
    expense_data: Dict[str, Any] = field(default_factory=dict)
    id_fields: Dict[str, str] = field(default_factory=dict)
    query_answers: Dict[str, str] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    pii_entities: List[Dict[str, Any]] = field(default_factory=list)
    redacted_text: str = ""
    genai_answers: List[Dict[str, Any]] = field(default_factory=list)
    human_review_triggered: bool = False
    human_review_arn: str = ""
    errors: List[str] = field(default_factory=list)


class IDPPipeline:
    """
    Orchestrates the full IDP workflow.

    Args:
        region (str): AWS region.
        bucket (str): S3 bucket for documents.
        role_arn (str): IAM role for AWS AI services.
        classifier_endpoint_arn (str): Comprehend classifier endpoint ARN.
        ner_endpoint_arn (str): Comprehend custom NER endpoint ARN.
        bedrock_model_id (str): Bedrock LLM model ID.
        embedding_model_id (str): Bedrock embedding model ID.
        a2i_flow_definition_arn (str): A2I flow definition ARN.
        confidence_threshold (float): A2I trigger threshold.
        enable_genai (bool): Whether to run Bedrock GenAI Q&A.
        enable_a2i (bool): Whether to trigger A2I review.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        bucket: str = "",
        role_arn: str = "",
        classifier_endpoint_arn: str = "",
        ner_endpoint_arn: str = "",
        bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v1",
        a2i_flow_definition_arn: str = "",
        confidence_threshold: float = 0.90,
        enable_genai: bool = True,
        enable_a2i: bool = False,
    ) -> None:
        self.region = region
        self.bucket = bucket
        self.confidence_threshold = confidence_threshold
        self.classifier_endpoint_arn = classifier_endpoint_arn
        self.ner_endpoint_arn = ner_endpoint_arn
        self.enable_genai = enable_genai
        self.enable_a2i = enable_a2i

        # Initialise individual phase components
        self.uploader = S3DocumentUploader(bucket=bucket, region=region)
        self.extractor = TextractExtractor(region=region, bucket=bucket)
        self.queries_extractor = TextractQueriesExtractor(region=region, bucket=bucket)
        self.classifier = DocumentClassifier(
            region=region, bucket=bucket, role_arn=role_arn
        )
        self.ner = EntityRecognizer(region=region, bucket=bucket, role_arn=role_arn)
        self.pii_redactor = PIIRedactor(region=region)
        self.medical_extractor = MedicalExtractor(region=region)

        if enable_genai:
            self.genai_qa = BedrockDocumentQA(
                region=region,
                bucket=bucket,
                bedrock_model_id=bedrock_model_id,
                embedding_model_id=embedding_model_id,
            )
        else:
            self.genai_qa = None  # type: ignore[assignment]

        if enable_a2i and a2i_flow_definition_arn:
            self.a2i = A2IHumanReview(
                region=region,
                flow_definition_arn=a2i_flow_definition_arn,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.a2i = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Factory: load from YAML config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "IDPPipeline":
        """
        Create an IDPPipeline from a YAML configuration file.

        Args:
            config_path: Path to config/config.yaml.

        Returns:
            Configured IDPPipeline instance.
        """
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        aws = cfg.get("aws", {})
        comp = cfg.get("comprehend", {})
        bedrock = cfg.get("bedrock", {})
        a2i = cfg.get("a2i", {})

        return cls(
            region=aws.get("region", "us-east-1"),
            bucket=aws.get("s3_bucket", ""),
            role_arn=aws.get("iam_role_arn", ""),
            classifier_endpoint_arn=comp.get("classifier_endpoint_arn", ""),
            ner_endpoint_arn=comp.get("ner_endpoint_arn", ""),
            bedrock_model_id=bedrock.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0"),
            embedding_model_id=bedrock.get("embedding_model_id", "amazon.titan-embed-text-v1"),
            a2i_flow_definition_arn=a2i.get("flow_definition_arn", ""),
            confidence_threshold=float(a2i.get("confidence_threshold", 0.90)),
        )

    # ------------------------------------------------------------------
    # Main pipeline entrypoint
    # ------------------------------------------------------------------

    def process_document(
        self,
        document_path: str,
        queries: Optional[List[Dict[str, str]]] = None,
        genai_questions: Optional[List[str]] = None,
        upload_first: bool = True,
    ) -> IDPResult:
        """
        Run the full IDP pipeline on a single document.

        Args:
            document_path: Local file path to the document.
            queries: Optional Textract natural-language query list.
            genai_questions: Optional questions for Bedrock GenAI Q&A.
            upload_first: Upload the document to S3 before processing.

        Returns:
            IDPResult containing all extracted data.
        """
        result = IDPResult(document_path=document_path)

        # ── Phase 1: Data Capture ─────────────────────────────────────
        logger.info("=== Phase 1: Data Capture ===")
        if upload_first:
            try:
                result.s3_uri = self.uploader.upload_document(document_path)
                s3_key = result.s3_uri.replace(f"s3://{self.bucket}/", "")
            except Exception as exc:
                result.errors.append(f"Upload failed: {exc}")
                logger.error("Upload failed: %s", exc)
                return result
        else:
            # Assume document_path IS the S3 key
            s3_key = document_path
            result.s3_uri = f"s3://{self.bucket}/{s3_key}"

        # ── Phase 3: Extraction (always run first to get raw text) ────
        logger.info("=== Phase 3: Extraction ===")
        try:
            textract_response = self.extractor.detect_text(s3_key=s3_key)
            result.raw_text = "\n".join(self.extractor.get_text_lines(textract_response))

            # Tables
            _, result.tables = self.extractor.extract_tables(s3_key=s3_key)
            logger.info("Extracted %d table(s)", len(result.tables))

            # Forms / key-value pairs
            _, result.kv_pairs = self.extractor.extract_forms(s3_key=s3_key)
            logger.info("Extracted %d KV pair(s)", len(result.kv_pairs))

            # Expense documents (invoices/receipts)
            doc_ext = os.path.splitext(document_path)[1].lower()
            if doc_ext in {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}:
                try:
                    result.expense_data = self.extractor.analyze_expense(s3_key=s3_key)
                except Exception:
                    pass  # Not all documents are expense docs

        except Exception as exc:
            result.errors.append(f"Extraction failed: {exc}")
            logger.error("Extraction failed: %s", exc)

        # ── Phase 2: Classification ────────────────────────────────────
        logger.info("=== Phase 2: Classification ===")
        if self.classifier_endpoint_arn and result.raw_text:
            try:
                doc_type, score = self.classifier.get_top_class(
                    text=result.raw_text,
                    endpoint_arn=self.classifier_endpoint_arn,
                )
                result.document_type = doc_type
                result.classification_confidence = score
                logger.info("Document classified as '%s' (confidence=%.2f)", doc_type, score)
            except Exception as exc:
                result.errors.append(f"Classification failed: {exc}")
                logger.warning("Classification skipped: %s", exc)
        else:
            logger.info("Classification skipped — no endpoint configured or no text extracted")

        # ── Phase 3b: Textract Queries ─────────────────────────────────
        if queries:
            logger.info("=== Phase 3b: Textract Queries ===")
            try:
                result.query_answers = self.queries_extractor.query_document(
                    queries=queries, s3_key=s3_key
                )
            except Exception as exc:
                result.errors.append(f"Queries failed: {exc}")
                logger.warning("Queries failed: %s", exc)

        # ── Phase 4: Enrichment ────────────────────────────────────────
        logger.info("=== Phase 4: Enrichment ===")
        if result.raw_text:
            try:
                # Built-in NER
                result.entities = self.ner.detect_entities(result.raw_text)
                logger.info("Detected %d entities", len(result.entities))
            except Exception as exc:
                result.errors.append(f"NER failed: {exc}")
                logger.warning("NER failed: %s", exc)

            try:
                # PII detection and text redaction
                result.pii_entities = self.pii_redactor.detect_pii(result.raw_text)
                if result.pii_entities:
                    result.redacted_text = self.pii_redactor.redact_text(
                        result.raw_text, result.pii_entities
                    )
                    logger.info("Redacted %d PII entities", len(result.pii_entities))
            except Exception as exc:
                result.errors.append(f"PII redaction failed: {exc}")
                logger.warning("PII redaction failed: %s", exc)

            # Custom NER (if endpoint configured)
            if self.ner_endpoint_arn:
                try:
                    custom_entities = self.ner.detect_custom_entities(
                        result.raw_text, self.ner_endpoint_arn
                    )
                    result.entities.extend(custom_entities)
                except Exception as exc:
                    logger.warning("Custom NER failed: %s", exc)

        # ── Phase 5: GenAI Q&A ─────────────────────────────────────────
        if self.enable_genai and self.genai_qa and result.raw_text:
            logger.info("=== Phase 5: GenAI Q&A ===")
            questions = genai_questions or [
                "What is this document about?",
                "What are the key data points in this document?",
                "Are there any dates or deadlines mentioned?",
            ]
            try:
                docs = self.genai_qa.load_text_as_documents(
                    result.raw_text, source=document_path
                )
                self.genai_qa.build_vector_store(docs)
                self.genai_qa.build_qa_chain()
                result.genai_answers = self.genai_qa.answer_questions_batch(questions)
            except Exception as exc:
                result.errors.append(f"GenAI Q&A failed: {exc}")
                logger.warning("GenAI Q&A failed: %s", exc)

        # ── Phase 6: Human Review (A2I) ────────────────────────────────
        if (
            self.enable_a2i
            and self.a2i
            and result.classification_confidence > 0
            and self.a2i.should_trigger_review(result.classification_confidence)
        ):
            logger.info("=== Phase 6: Human Review (A2I) — low confidence ===")
            try:
                loop_arn = self.a2i.start_human_loop(
                    document_s3_uri=result.s3_uri,
                    extracted_data={
                        "document_type": result.document_type,
                        "kv_pairs": result.kv_pairs,
                        "confidence": result.classification_confidence,
                    },
                )
                result.human_review_triggered = True
                result.human_review_arn = loop_arn
                logger.info("Human review triggered — ARN: %s", loop_arn)
            except Exception as exc:
                result.errors.append(f"A2I review failed: {exc}")
                logger.warning("A2I review failed: %s", exc)

        logger.info(
            "Pipeline complete for '%s' — %d errors",
            document_path,
            len(result.errors),
        )
        return result

    def process_documents_batch(
        self,
        document_paths: List[str],
        **kwargs: Any,
    ) -> List[IDPResult]:
        """
        Process multiple documents through the IDP pipeline.

        Args:
            document_paths: List of local file paths.
            **kwargs: Additional arguments passed to process_document.

        Returns:
            List of IDPResult objects.
        """
        results = []
        for i, path in enumerate(document_paths, start=1):
            logger.info("Processing document %d/%d: %s", i, len(document_paths), path)
            try:
                result = self.process_document(path, **kwargs)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to process %s: %s", path, exc)
                failed = IDPResult(document_path=path)
                failed.errors.append(str(exc))
                results.append(failed)
        return results
