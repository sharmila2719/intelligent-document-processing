"""
Phase 4 – Document Enrichment: PII Detection and Redaction
============================================================
Detect personally identifiable information (PII) using Amazon Comprehend
and redact it from document images using Amazon Textract bounding-box
geometry.

Reference:
  https://aws.amazon.com/blogs/machine-learning/part-2-intelligent-document-processing-with-aws-ai-services/
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import boto3

logger = logging.getLogger(__name__)


class PIIRedactor:
    """
    Detects PII / sensitive entities from document text (via Amazon Comprehend)
    and optionally redacts them from the source image using bounding-box data
    from Amazon Textract.

    Supported PII types include: NAME, ADDRESS, CREDIT_DEBIT_NUMBER,
    SSN, EMAIL, PHONE, DATE_TIME, BANK_ACCOUNT_NUMBER, and more.

    Args:
        region (str): AWS region.
    """

    # Default PII entity types to redact
    DEFAULT_PII_TYPES = [
        "NAME",
        "ADDRESS",
        "CREDIT_DEBIT_NUMBER",
        "CREDIT_DEBIT_CVV",
        "CREDIT_DEBIT_EXPIRY",
        "PIN",
        "EMAIL",
        "ADDRESS",
        "PHONE",
        "SSN",
        "DATE_TIME",
        "BANK_ACCOUNT_NUMBER",
        "BANK_ROUTING",
        "PASSPORT_NUMBER",
        "DRIVER_ID",
        "IP_ADDRESS",
        "MAC_ADDRESS",
        "URL",
        "USERNAME",
        "PASSWORD",
    ]

    def __init__(self, region: str = "us-east-1") -> None:
        self.region = region
        self.comprehend = boto3.client("comprehend", region_name=region)
        self.textract = boto3.client("textract", region_name=region)

    # ------------------------------------------------------------------
    # PII detection
    # ------------------------------------------------------------------

    def detect_pii(
        self,
        text: str,
        language_code: str = "en",
        pii_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect PII entities in text using Amazon Comprehend.

        Args:
            text: Plain document text.
            language_code: Language code (default "en").
            pii_types: Filter to specific PII types. None = all types.

        Returns:
            List of PII entity dicts:
            [{"Type": ..., "Score": ..., "BeginOffset": ..., "EndOffset": ...,
              "Text": ...}]
        """
        chunks = self._chunk_text(text, max_bytes=4800)
        all_pii: List[Dict[str, Any]] = []
        offset = 0

        for chunk in chunks:
            response = self.comprehend.detect_pii_entities(
                Text=chunk,
                LanguageCode=language_code,
            )
            for entity in response.get("Entities", []):
                pii_type = entity.get("Type", "")
                if pii_types and pii_type not in pii_types:
                    continue
                entity["BeginOffset"] += offset
                entity["EndOffset"] += offset
                # Extract the actual PII text span
                entity["Text"] = text[entity["BeginOffset"]: entity["EndOffset"]]
                all_pii.append(entity)
            offset += len(chunk)

        logger.info("Detected %d PII entities", len(all_pii))
        return all_pii

    def redact_text(
        self,
        text: str,
        pii_entities: List[Dict[str, Any]],
        redact_char: str = "*",
    ) -> str:
        """
        Redact PII entities from plain text by replacing them with a
        mask character.

        Args:
            text: Original document text.
            pii_entities: List of PII entities (from detect_pii).
            redact_char: Character to use for masking (default "*").

        Returns:
            Redacted text string.
        """
        # Sort by BeginOffset descending so replacements don't shift offsets
        sorted_entities = sorted(
            pii_entities, key=lambda e: e["BeginOffset"], reverse=True
        )
        text_list = list(text)

        for entity in sorted_entities:
            begin = entity["BeginOffset"]
            end = entity["EndOffset"]
            span_len = end - begin
            text_list[begin:end] = list(redact_char * span_len)

        return "".join(text_list)

    # ------------------------------------------------------------------
    # Image-level redaction using Textract geometry
    # ------------------------------------------------------------------

    def get_redaction_boxes(
        self,
        textract_response: Dict[str, Any],
        pii_text_values: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Find bounding boxes for PII words in a Textract response.

        Used to draw black rectangles over PII in the original document image.

        Args:
            textract_response: Full Textract API response (from AnalyzeDocument
                               or DetectDocumentText).
            pii_text_values: List of PII text strings to locate in the document.

        Returns:
            List of bounding box dicts:
            [{"Left": ..., "Top": ..., "Width": ..., "Height": ..., "Text": ...}]
        """
        pii_set = {v.lower() for v in pii_text_values}
        boxes: List[Dict[str, Any]] = []

        for block in textract_response.get("Blocks", []):
            if block.get("BlockType") not in {"WORD", "LINE"}:
                continue
            block_text = block.get("Text", "").lower()
            if any(pii in block_text for pii in pii_set):
                geometry = block.get("Geometry", {}).get("BoundingBox", {})
                if geometry:
                    boxes.append(
                        {
                            "Left": geometry["Left"],
                            "Top": geometry["Top"],
                            "Width": geometry["Width"],
                            "Height": geometry["Height"],
                            "Text": block.get("Text", ""),
                        }
                    )

        logger.info("Found %d bounding boxes for PII redaction", len(boxes))
        return boxes

    def redact_image(
        self,
        image_path: str,
        bounding_boxes: List[Dict[str, Any]],
        output_path: str,
        fill_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> str:
        """
        Draw filled rectangles over PII bounding boxes in a document image.

        Requires Pillow (PIL) to be installed.

        Args:
            image_path: Local path to the source image (PNG, JPEG, TIFF).
            bounding_boxes: List of bounding boxes from get_redaction_boxes.
            output_path: Local path to save the redacted image.
            fill_color: RGB tuple for the redaction colour (default black).

        Returns:
            Path to the saved redacted image.
        """
        try:
            from PIL import Image, ImageDraw  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Pillow is required for image redaction: pip install Pillow") from exc

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)

        for box in bounding_boxes:
            left = int(box["Left"] * width)
            top = int(box["Top"] * height)
            right = left + int(box["Width"] * width)
            bottom = top + int(box["Height"] * height)
            draw.rectangle([left, top, right, bottom], fill=fill_color)

        image.save(output_path)
        logger.info("Redacted image saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, max_bytes: int = 4800) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []
        current_size = 0

        for word in text.split():
            word_bytes = len(word.encode("utf-8")) + 1
            if current_size + word_bytes > max_bytes and current:
                chunks.append(" ".join(current))
                current = [word]
                current_size = word_bytes
            else:
                current.append(word)
                current_size += word_bytes

        if current:
            chunks.append(" ".join(current))
        return chunks
