"""
Unit tests for the enrichment modules (NER, PII redaction, Medical).
Uses mocked boto3 clients so no real AWS calls are made.
"""

import unittest
from unittest.mock import MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enrichment.entity_recognizer import EntityRecognizer
from src.enrichment.pii_redactor import PIIRedactor
from src.enrichment.medical_extractor import MedicalExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_ENTITIES_RESPONSE = {
    "Entities": [
        {"Text": "John Doe",    "Type": "PERSON",       "Score": 0.99, "BeginOffset": 0,  "EndOffset": 8},
        {"Text": "New York",   "Type": "LOCATION",     "Score": 0.97, "BeginOffset": 20, "EndOffset": 28},
        {"Text": "January 5",  "Type": "DATE",         "Score": 0.95, "BeginOffset": 33, "EndOffset": 42},
    ]
}

MOCK_PII_RESPONSE = {
    "Entities": [
        {"Type": "NAME",    "Score": 0.99, "BeginOffset": 5,  "EndOffset": 13},
        {"Type": "EMAIL",   "Score": 0.98, "BeginOffset": 20, "EndOffset": 38},
        {"Type": "PHONE",   "Score": 0.97, "BeginOffset": 45, "EndOffset": 57},
    ]
}

MOCK_MEDICAL_RESPONSE = {
    "Entities": [
        {"Text": "metformin", "Category": "MEDICATION", "Type": "GENERIC_NAME", "Score": 0.98,
         "BeginOffset": 0, "EndOffset": 9, "Traits": [], "Attributes": []},
        {"Text": "diabetes",  "Category": "MEDICAL_CONDITION", "Type": "DX_NAME", "Score": 0.97,
         "BeginOffset": 14, "EndOffset": 22, "Traits": [], "Attributes": []},
    ]
}

MOCK_PHI_RESPONSE = {
    "Entities": [
        {"Text": "John Smith", "Type": "NAME",  "Score": 0.99, "BeginOffset": 0, "EndOffset": 10},
        {"Text": "555-0100",   "Type": "PHONE", "Score": 0.98, "BeginOffset": 15, "EndOffset": 23},
    ]
}


# ---------------------------------------------------------------------------
# Tests — EntityRecognizer
# ---------------------------------------------------------------------------

class TestEntityRecognizer(unittest.TestCase):

    def setUp(self):
        with patch("boto3.client"):
            self.ner = EntityRecognizer(region="us-east-1", bucket="test-bucket")
            self.ner.comprehend = MagicMock()

    def test_detect_entities_returns_list(self):
        self.ner.comprehend.detect_entities.return_value = MOCK_ENTITIES_RESPONSE
        entities = self.ner.detect_entities("John Doe lives in New York on January 5")
        self.assertEqual(len(entities), 3)

    def test_detect_entities_by_type_groups_correctly(self):
        self.ner.comprehend.detect_entities.return_value = MOCK_ENTITIES_RESPONSE
        grouped = self.ner.detect_entities_by_type(
            "John Doe lives in New York",
            entity_types=["PERSON", "LOCATION"],
        )
        self.assertIn("PERSON", grouped)
        self.assertIn("LOCATION", grouped)
        self.assertEqual(grouped["PERSON"], ["John Doe"])
        self.assertEqual(grouped["LOCATION"], ["New York"])

    def test_chunk_text_splits_correctly(self):
        """Large text should be split into chunks not exceeding max_bytes."""
        long_text = "word " * 2000  # ~10 000 bytes
        chunks = EntityRecognizer._chunk_text(long_text, max_bytes=4800)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.encode("utf-8")), 4800 + 10)  # tolerance


# ---------------------------------------------------------------------------
# Tests — PIIRedactor
# ---------------------------------------------------------------------------

class TestPIIRedactor(unittest.TestCase):

    def setUp(self):
        with patch("boto3.client"):
            self.redactor = PIIRedactor(region="us-east-1")
            self.redactor.comprehend = MagicMock()

    def test_detect_pii_returns_entities(self):
        self.redactor.comprehend.detect_pii_entities.return_value = MOCK_PII_RESPONSE
        text = "Hi, John Doe (john.doe@example.com) — 555-123-4567"
        pii = self.redactor.detect_pii(text)
        self.assertEqual(len(pii), 3)
        types = [e["Type"] for e in pii]
        self.assertIn("NAME", types)
        self.assertIn("EMAIL", types)
        self.assertIn("PHONE", types)

    def test_redact_text_replaces_pii(self):
        """redact_text should mask PII spans with asterisks."""
        text = "Call John Doe at 555-123-4567"
        # Manually specify PII entities with text spans
        pii_entities = [
            {"Type": "NAME",  "BeginOffset": 5,  "EndOffset": 13,
             "Score": 0.99, "Text": "John Doe"},
            {"Type": "PHONE", "BeginOffset": 17, "EndOffset": 29,
             "Score": 0.98, "Text": "555-123-4567"},
        ]
        redacted = self.redactor.redact_text(text, pii_entities)
        self.assertNotIn("John Doe",     redacted)
        self.assertNotIn("555-123-4567", redacted)
        self.assertIn("*", redacted)

    def test_redact_text_preserves_non_pii(self):
        """Non-PII parts of the text should remain unchanged."""
        text = "Call John Doe now"
        pii_entities = [
            {"Type": "NAME", "BeginOffset": 5, "EndOffset": 13,
             "Score": 0.99, "Text": "John Doe"},
        ]
        redacted = self.redactor.redact_text(text, pii_entities)
        self.assertIn("Call", redacted)
        self.assertIn("now",  redacted)


# ---------------------------------------------------------------------------
# Tests — MedicalExtractor
# ---------------------------------------------------------------------------

class TestMedicalExtractor(unittest.TestCase):

    def setUp(self):
        with patch("boto3.client"):
            self.med = MedicalExtractor(region="us-east-1")
            self.med.comprehend_medical = MagicMock()

    def test_detect_medical_entities_returns_entities(self):
        self.med.comprehend_medical.detect_entities_v2.return_value = MOCK_MEDICAL_RESPONSE
        entities = self.med.detect_medical_entities("Patient takes metformin for diabetes")
        self.assertEqual(len(entities), 2)

    def test_get_entities_by_category(self):
        self.med.comprehend_medical.detect_entities_v2.return_value = MOCK_MEDICAL_RESPONSE
        grouped = self.med.get_entities_by_category(
            "Patient takes metformin for diabetes",
            categories=["MEDICATION", "MEDICAL_CONDITION"],
        )
        self.assertIn("MEDICATION", grouped)
        self.assertIn("MEDICAL_CONDITION", grouped)
        meds = [e["Text"] for e in grouped["MEDICATION"]]
        self.assertIn("metformin", meds)

    def test_detect_phi(self):
        self.med.comprehend_medical.detect_phi.return_value = MOCK_PHI_RESPONSE
        phi = self.med.detect_phi("John Smith called at 555-0100")
        self.assertEqual(len(phi), 2)
        phi_types = [e["Type"] for e in phi]
        self.assertIn("NAME", phi_types)
        self.assertIn("PHONE", phi_types)


if __name__ == "__main__":
    unittest.main()
