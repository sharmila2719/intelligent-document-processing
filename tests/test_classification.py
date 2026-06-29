"""
Unit tests for the document classification module.
Uses mocked boto3 clients so no real AWS calls are made.
"""

import unittest
from unittest.mock import MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification.document_classifier import DocumentClassifier


MOCK_CLASSIFY_RESPONSE = {
    "Classes": [
        {"Name": "invoice",       "Score": 0.9543},
        {"Name": "bank-statement","Score": 0.0312},
        {"Name": "receipt",       "Score": 0.0145},
    ]
}

MOCK_DETECT_TEXT_RESPONSE = {
    "Blocks": [
        {"BlockType": "LINE", "Text": "INVOICE"},
        {"BlockType": "LINE", "Text": "Bill To: Acme Corp"},
        {"BlockType": "LINE", "Text": "Total: $1,200.00"},
    ]
}


class TestDocumentClassifier(unittest.TestCase):

    def setUp(self):
        with patch("boto3.client"):
            self.clf = DocumentClassifier(
                region="us-east-1",
                bucket="test-bucket",
                role_arn="arn:aws:iam::123456789012:role/TestRole",
            )
            self.clf.comprehend = MagicMock()
            self.clf.textract = MagicMock()
            self.clf.s3 = MagicMock()

    # ------------------------------------------------------------------

    def test_classify_document_returns_sorted_classes(self):
        """classify_document should return classes sorted by score descending."""
        self.clf.comprehend.classify_document.return_value = MOCK_CLASSIFY_RESPONSE
        result = self.clf.classify_document("Sample invoice text", "arn:aws:comprehend::endpoint/test")
        classes = list(result.keys())
        scores = list(result.values())
        # First entry must be the highest score
        self.assertEqual(classes[0], "invoice")
        self.assertGreater(scores[0], scores[1])

    def test_get_top_class_returns_invoice(self):
        """get_top_class should return 'invoice' with highest confidence."""
        self.clf.comprehend.classify_document.return_value = MOCK_CLASSIFY_RESPONSE
        top_class, score = self.clf.get_top_class(
            "Sample invoice text", "arn:aws:comprehend::endpoint/test"
        )
        self.assertEqual(top_class, "invoice")
        self.assertAlmostEqual(score, 0.9543, places=3)

    def test_extract_text_from_document(self):
        """extract_text_from_document should concatenate LINE blocks."""
        self.clf.textract.detect_document_text.return_value = MOCK_DETECT_TEXT_RESPONSE
        text = self.clf.extract_text_from_document("test/invoice.pdf")
        self.assertIn("INVOICE", text)
        self.assertIn("Total: $1,200.00", text)

    def test_classify_document_calls_comprehend(self):
        """Ensure classify_document sends the text and endpoint to Comprehend."""
        self.clf.comprehend.classify_document.return_value = MOCK_CLASSIFY_RESPONSE
        endpoint_arn = "arn:aws:comprehend:us-east-1:123:document-classifier-endpoint/test"
        self.clf.classify_document("some text", endpoint_arn)
        call_kwargs = self.clf.comprehend.classify_document.call_args.kwargs
        self.assertEqual(call_kwargs["Text"], "some text")
        self.assertEqual(call_kwargs["EndpointArn"], endpoint_arn)


if __name__ == "__main__":
    unittest.main()
