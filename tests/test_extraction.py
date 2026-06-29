"""
Unit tests for the Textract extraction modules.
Uses mocked boto3 clients so no real AWS calls are made.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.textract_extractor import TextractExtractor
from src.extraction.queries_extractor import TextractQueriesExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_DETECT_TEXT_RESPONSE = {
    "Blocks": [
        {"BlockType": "LINE", "Text": "Invoice Number: INV-001", "Id": "b1"},
        {"BlockType": "LINE", "Text": "Total Amount: $500.00",   "Id": "b2"},
        {"BlockType": "WORD", "Text": "Invoice",                  "Id": "b3"},
    ]
}

MOCK_QUERY_RESPONSE = {
    "Blocks": [
        {
            "BlockType": "QUERY",
            "Id": "q1",
            "Query": {"Text": "What is the invoice number?", "Alias": "INV_NO"},
            "Relationships": [{"Type": "ANSWER", "Ids": ["qa1"]}],
        },
        {
            "BlockType": "QUERY_RESULT",
            "Id": "qa1",
            "Text": "INV-001",
            "Confidence": 98.5,
        },
    ]
}

MOCK_EXPENSE_RESPONSE = {
    "ExpenseDocuments": [
        {
            "SummaryFields": [
                {"Type": {"Text": "VENDOR_NAME"}, "ValueDetection": {"Text": "Acme Corp"}},
                {"Type": {"Text": "TOTAL"},       "ValueDetection": {"Text": "$500.00"}},
            ],
            "LineItemGroups": [
                {
                    "LineItems": [
                        {
                            "LineItemExpenseFields": [
                                {"Type": {"Text": "ITEM"},     "ValueDetection": {"Text": "Widget A"}},
                                {"Type": {"Text": "QUANTITY"}, "ValueDetection": {"Text": "2"}},
                                {"Type": {"Text": "PRICE"},    "ValueDetection": {"Text": "$250.00"}},
                            ]
                        }
                    ]
                }
            ],
        }
    ]
}

MOCK_ID_RESPONSE = {
    "IdentityDocuments": [
        {
            "IdentityDocumentFields": [
                {"Type": {"Text": "FIRST_NAME"},  "ValueDetection": {"Text": "John"}},
                {"Type": {"Text": "LAST_NAME"},   "ValueDetection": {"Text": "Doe"}},
                {"Type": {"Text": "DATE_OF_BIRTH"}, "ValueDetection": {"Text": "01/01/1990"}},
            ]
        }
    ]
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTextractExtractor(unittest.TestCase):

    def setUp(self):
        with patch("boto3.client") as mock_client:
            self.mock_textract = MagicMock()
            mock_client.return_value = self.mock_textract
            self.extractor = TextractExtractor(region="us-east-1", bucket="test-bucket")
            self.extractor.textract = self.mock_textract

    def test_get_text_lines(self):
        """Test LINE block extraction from Textract response."""
        lines = self.extractor.get_text_lines(MOCK_DETECT_TEXT_RESPONSE)
        self.assertEqual(len(lines), 2)
        self.assertIn("Invoice Number: INV-001", lines)
        self.assertIn("Total Amount: $500.00", lines)

    def test_detect_text_calls_api(self):
        """Test that detect_text calls Textract DetectDocumentText."""
        self.mock_textract.detect_document_text.return_value = MOCK_DETECT_TEXT_RESPONSE
        response = self.extractor.detect_text(s3_key="test/doc.pdf")
        self.mock_textract.detect_document_text.assert_called_once()
        self.assertIn("Blocks", response)

    def test_detect_text_with_bytes(self):
        """Test detect_text with image bytes."""
        self.mock_textract.detect_document_text.return_value = MOCK_DETECT_TEXT_RESPONSE
        response = self.extractor.detect_text(image_bytes=b"fake-image-bytes")
        call_args = self.mock_textract.detect_document_text.call_args
        self.assertIn("Bytes", call_args.kwargs["Document"])

    def test_detect_text_raises_without_input(self):
        """Test detect_text raises ValueError when no input is given."""
        with self.assertRaises(ValueError):
            self.extractor.detect_text()

    def test_analyze_expense_parses_vendor_and_total(self):
        """Test expense analysis extracts vendor name and total."""
        self.mock_textract.analyze_expense.return_value = MOCK_EXPENSE_RESPONSE
        result = self.extractor.analyze_expense(s3_key="test/receipt.jpg")
        docs = result.get("documents", [])
        self.assertEqual(len(docs), 1)
        summary = docs[0]["summary_fields"]
        self.assertEqual(summary.get("VENDOR_NAME"), "Acme Corp")
        self.assertEqual(summary.get("TOTAL"), "$500.00")

    def test_analyze_expense_parses_line_items(self):
        """Test expense analysis extracts line items."""
        self.mock_textract.analyze_expense.return_value = MOCK_EXPENSE_RESPONSE
        result = self.extractor.analyze_expense(s3_key="test/receipt.jpg")
        line_items = result["documents"][0]["line_items"]
        self.assertEqual(len(line_items), 1)
        self.assertEqual(line_items[0].get("ITEM"), "Widget A")
        self.assertEqual(line_items[0].get("QUANTITY"), "2")

    def test_analyze_id_extracts_fields(self):
        """Test ID document analysis extracts name and DOB."""
        self.mock_textract.analyze_id.return_value = MOCK_ID_RESPONSE
        fields = self.extractor.analyze_id(s3_key="test/license.png")
        self.assertEqual(fields.get("FIRST_NAME"), "John")
        self.assertEqual(fields.get("LAST_NAME"), "Doe")
        self.assertEqual(fields.get("DATE_OF_BIRTH"), "01/01/1990")


class TestTextractQueriesExtractor(unittest.TestCase):

    def setUp(self):
        with patch("boto3.client") as mock_client:
            self.mock_textract = MagicMock()
            mock_client.return_value = self.mock_textract
            self.qe = TextractQueriesExtractor(region="us-east-1", bucket="test-bucket")
            self.qe.textract = self.mock_textract

    def test_query_document_returns_answers(self):
        """Test query_document extracts answers keyed by alias."""
        self.mock_textract.analyze_document.return_value = MOCK_QUERY_RESPONSE
        queries = [{"Text": "What is the invoice number?", "Alias": "INV_NO"}]
        answers = self.qe.query_document(queries=queries, s3_key="test/invoice.pdf")
        self.assertIn("INV_NO", answers)
        self.assertEqual(answers["INV_NO"], "INV-001")

    def test_query_document_raises_without_input(self):
        """Test query_document raises ValueError without document input."""
        with self.assertRaises(ValueError):
            self.qe.query_document(queries=[{"Text": "Question?"}])


if __name__ == "__main__":
    unittest.main()
