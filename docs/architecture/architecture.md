# IDP Architecture Description

## Overview

The Intelligent Document Processing (IDP) pipeline combines six AWS AI service
phases to transform raw documents into structured, actionable data.

```
Documents (PDF/PNG/JPEG/TIFF)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 1 – Data Capture                                                  │
│  Amazon S3 (document storage) + Amazon SQS (event queue)                │
│  + AWS Lambda (trigger handler)                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 2 – Classification                                                │
│  Amazon Textract (DetectDocumentText for training data)                  │
│  → Amazon Comprehend Custom Classifier (train + deploy endpoint)         │
│  → Document type: invoice / bank-statement / receipt / etc.             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 3 – Extraction                                                    │
│  Amazon Textract:                                                        │
│    - DetectDocumentText  (unstructured raw text)                         │
│    - AnalyzeDocument TABLES  (structured tabular data)                   │
│    - AnalyzeDocument FORMS   (semi-structured key-value pairs)           │
│    - AnalyzeExpense           (invoices & receipts)                      │
│    - AnalyzeID                (driver's licences, passports)             │
│    - AnalyzeDocument QUERIES  (natural-language questions)               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 4 – Enrichment                                                    │
│  Amazon Comprehend:                                                      │
│    - DetectEntities         (built-in NER: PERSON, LOCATION, DATE, ...)  │
│    - DetectPiiEntities      (NAME, EMAIL, SSN, PHONE, ...)               │
│    - Custom Entity Recognizer (SAVINGS_AC, CHECKING_AC, etc.)           │
│  Amazon Comprehend Medical:                                              │
│    - DetectEntitiesV2       (MEDICATION, MEDICAL_CONDITION, ANATOMY)    │
│    - DetectPHI              (Protected Health Information)               │
│    - InferICD10CM / InferRxNorm / InferSNOMEDCT                          │
│  PII Redaction: Textract bounding boxes + Pillow image masking           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 5 – Generative AI Q&A  (Bedrock + LangChain)                      │
│  Amazon Textract PDF Loader (LangChain)                                  │
│  → Text splitting (RecursiveCharacterTextSplitter)                       │
│  → Amazon Bedrock Titan Embeddings (vector generation)                   │
│  → FAISS vector store (in-memory)                                        │
│  → LangChain RetrievalQA chain                                           │
│  → Amazon Bedrock Claude 3 (LLM Q&A / summarisation / extraction)       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 6 – Human Review & Validation  (A2I)                              │
│  Confidence score < threshold → Amazon A2I human review loop             │
│  Amazon SageMaker Ground Truth workforce                                 │
│  Human-corrected output → Amazon S3                                      │
│  → Downstream systems (databases, CRM, ERP)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## AWS Services Used

| Service | Purpose |
|---|---|
| Amazon S3 | Document storage, training data, output artefacts |
| Amazon SQS | Event-driven pipeline trigger queue |
| AWS Lambda | Serverless trigger handler |
| Amazon Textract | OCR, table/form/expense/ID extraction, NL queries |
| Amazon Comprehend | Document classification, NER, PII detection |
| Amazon Comprehend Medical | Medical NER, PHI, ICD-10, RxNorm, SNOMED CT |
| Amazon Bedrock (Titan) | Text embeddings for RAG |
| Amazon Bedrock (Claude 3) | LLM for Q&A, summarisation, structured extraction |
| LangChain | RAG orchestration layer |
| FAISS | In-memory vector similarity search |
| Amazon A2I | Human-in-the-loop review for low-confidence extractions |
| Amazon SageMaker | A2I workforce and human review UI |

## Data Flow

1. Documents are uploaded to **Amazon S3** (any format: PDF, PNG, JPEG, TIFF).
2. An SQS event triggers **AWS Lambda** which starts the pipeline.
3. **Amazon Textract** extracts raw text for Comprehend classification training.
4. **Amazon Comprehend** classifies the document type using a custom trained model.
5. **Amazon Textract** performs deep extraction (tables, forms, queries, expense, ID).
6. **Amazon Comprehend** enriches extracted text with named entities and detects PII.
7. PII entities are redacted from the document image using bounding-box geometry.
8. **Amazon Bedrock + LangChain** enables RAG-based Q&A over the document.
9. If extraction confidence is low, **Amazon A2I** routes the document to a human reviewer.
10. Validated output is stored in S3 and forwarded to downstream systems.

## References

- [AWS blog: IDP Part 1](https://aws.amazon.com/blogs/machine-learning/part-1-intelligent-document-processing-with-aws-ai-services/)
- [AWS blog: IDP Part 2](https://aws.amazon.com/blogs/machine-learning/part-2-intelligent-document-processing-with-aws-ai-services/)
- [AWS blog: IDP with Textract, Bedrock & LangChain](https://aws.amazon.com/blogs/machine-learning/intelligent-document-processing-with-amazon-textract-amazon-bedrock-and-langchain/)
