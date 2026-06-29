"""
run_pipeline.py — CLI entrypoint for the IDP pipeline
======================================================

Usage:
    python scripts/run_pipeline.py \\
        --document path/to/document.pdf \\
        --config  config/config.yaml \\
        [--questions "What is the invoice number?" "What is the total?"] \\
        [--no-genai] \\
        [--output results.json]
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict

# Add project root to path so `src` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.idp_pipeline import IDPPipeline

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the IDP pipeline on one or more documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF with GenAI Q&A
  python scripts/run_pipeline.py --document invoice.pdf

  # Process with custom questions
  python scripts/run_pipeline.py \\
      --document contract.pdf \\
      --questions "What is the contract start date?" "Who are the parties?"

  # Skip GenAI phase (faster, no Bedrock cost)
  python scripts/run_pipeline.py --document receipt.png --no-genai

  # Save output to JSON
  python scripts/run_pipeline.py --document form.pdf --output results.json
        """,
    )
    parser.add_argument(
        "--document", "-d",
        required=True,
        help="Path to the document to process (PDF, PNG, JPEG, TIFF)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="Path to config/config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--questions", "-q",
        nargs="*",
        default=None,
        help="Custom GenAI questions to ask about the document",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Textract natural-language queries, e.g. 'What is the invoice number?'",
    )
    parser.add_argument(
        "--no-genai",
        action="store_true",
        default=False,
        help="Disable Bedrock GenAI Q&A phase",
    )
    parser.add_argument(
        "--enable-a2i",
        action="store_true",
        default=False,
        help="Enable Amazon A2I human review for low-confidence extractions",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save pipeline results to this JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def format_result_summary(result) -> str:
    """Return a human-readable summary of the IDPResult."""
    lines = [
        "",
        "=" * 70,
        "  IDP PIPELINE RESULTS",
        "=" * 70,
        f"  Document       : {result.document_path}",
        f"  S3 URI         : {result.s3_uri}",
        f"  Document Type  : {result.document_type or 'N/A'}",
        f"  Confidence     : {result.classification_confidence:.2%}" if result.classification_confidence else "  Confidence     : N/A",
        "-" * 70,
        f"  Raw text (first 300 chars):",
        f"  {result.raw_text[:300]}..." if len(result.raw_text) > 300 else f"  {result.raw_text}",
        "-" * 70,
        f"  Tables extracted   : {len(result.tables)}",
        f"  KV pairs extracted : {len(result.kv_pairs)}",
        f"  Entities detected  : {len(result.entities)}",
        f"  PII entities found : {len(result.pii_entities)}",
    ]

    if result.kv_pairs:
        lines.append("-" * 70)
        lines.append("  Key-Value Pairs:")
        for k, v in list(result.kv_pairs.items())[:10]:
            lines.append(f"    {k:30s} : {v}")
        if len(result.kv_pairs) > 10:
            lines.append(f"    ... and {len(result.kv_pairs) - 10} more")

    if result.query_answers:
        lines.append("-" * 70)
        lines.append("  Textract Query Answers:")
        for alias, answer in result.query_answers.items():
            lines.append(f"    {alias:30s} : {answer}")

    if result.genai_answers:
        lines.append("-" * 70)
        lines.append("  GenAI Q&A Answers:")
        for item in result.genai_answers:
            q = item.get("question", "")
            a = item.get("answer", "")
            lines.append(f"    Q: {q}")
            lines.append(f"    A: {a[:200]}...")
            lines.append("")

    if result.human_review_triggered:
        lines.append("-" * 70)
        lines.append(f"  ⚠  Human review triggered — Loop ARN: {result.human_review_arn}")

    if result.errors:
        lines.append("-" * 70)
        lines.append("  Errors / Warnings:")
        for err in result.errors:
            lines.append(f"    ⚠  {err}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build queries list for Textract QUERIES feature
    textract_queries = None
    if args.queries:
        textract_queries = [
            {"Text": q, "Alias": f"Q{i+1}"}
            for i, q in enumerate(args.queries)
        ]

    logger.info("Loading pipeline from %s", args.config)
    pipeline = IDPPipeline.from_config(args.config)
    pipeline.enable_genai = not args.no_genai
    pipeline.enable_a2i = args.enable_a2i

    logger.info("Processing document: %s", args.document)
    result = pipeline.process_document(
        document_path=args.document,
        queries=textract_queries,
        genai_questions=args.questions,
    )

    # Print summary
    print(format_result_summary(result))

    # Optionally save to JSON
    if args.output:
        # Convert dataclass to dict — handle non-serialisable objects
        def default_serialiser(obj):
            try:
                return str(obj)
            except Exception:
                return "<not serialisable>"

        result_dict = asdict(result)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, default=default_serialiser)
        logger.info("Results saved to %s", args.output)
        print(f"\nResults saved to: {args.output}")

    return 0 if not result.errors else 1


if __name__ == "__main__":
    sys.exit(main())
