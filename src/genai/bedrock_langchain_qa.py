"""
Generative AI Phase – IDP with Amazon Textract, Amazon Bedrock & LangChain
==========================================================================
Implements Retrieval-Augmented Generation (RAG) Q&A over documents by:

  1. Loading documents from S3 or local disk using LangChain's
     AmazonTextractPDFLoader (which calls Textract under the hood).
  2. Splitting text into chunks and generating vector embeddings using
     Amazon Titan Embeddings via Amazon Bedrock.
  3. Storing embeddings in an in-memory FAISS vector store.
  4. Answering natural-language questions using a Bedrock LLM
     (default: Anthropic Claude 3 Sonnet) with retrieved context (RAG).

Reference:
  https://aws.amazon.com/blogs/machine-learning/intelligent-document-processing-with-amazon-textract-amazon-bedrock-and-langchain/
"""

import logging
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)


class BedrockDocumentQA:
    """
    End-to-end RAG pipeline: Textract → LangChain → Bedrock embeddings
    → FAISS vector store → Bedrock LLM Q&A.

    Args:
        region (str): AWS region.
        bucket (str): S3 bucket containing documents.
        bedrock_model_id (str): Bedrock LLM model ID.
        embedding_model_id (str): Bedrock embedding model ID.
        max_tokens (int): Max tokens for LLM response.
        temperature (float): LLM temperature (0 = deterministic).
    """

    def __init__(
        self,
        region: str = "us-east-1",
        bucket: str = "",
        bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v1",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        self.region = region
        self.bucket = bucket
        self.bedrock_model_id = bedrock_model_id
        self.embedding_model_id = embedding_model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._bedrock_runtime = boto3.client(
            "bedrock-runtime", region_name=region
        )
        self._vector_store: Optional[Any] = None  # FAISS vector store
        self._qa_chain: Optional[Any] = None

    # ------------------------------------------------------------------
    # Step 1 – Load documents
    # ------------------------------------------------------------------

    def load_document_from_s3(self, s3_key: str) -> List[Any]:
        """
        Load a document from S3 using LangChain AmazonTextractPDFLoader.

        The loader calls Amazon Textract internally to extract text and
        returns LangChain Document objects.

        Args:
            s3_key: S3 object key of the document.

        Returns:
            List of LangChain Document objects (one per page).
        """
        try:
            from langchain_community.document_loaders import AmazonTextractPDFLoader  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required: pip install langchain-community"
            ) from exc

        s3_uri = f"s3://{self.bucket}/{s3_key}"
        logger.info("Loading document from %s via Textract", s3_uri)
        loader = AmazonTextractPDFLoader(
            file_path=s3_uri,
            region_name=self.region,
        )
        documents = loader.load()
        logger.info("Loaded %d page(s) from %s", len(documents), s3_uri)
        return documents

    def load_document_from_path(self, file_path: str) -> List[Any]:
        """
        Load a local PDF document using LangChain AmazonTextractPDFLoader.

        Args:
            file_path: Local path to a PDF file.

        Returns:
            List of LangChain Document objects.
        """
        try:
            from langchain_community.document_loaders import AmazonTextractPDFLoader  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required: pip install langchain-community"
            ) from exc

        logger.info("Loading local document: %s", file_path)
        loader = AmazonTextractPDFLoader(
            file_path=file_path,
            region_name=self.region,
        )
        documents = loader.load()
        logger.info("Loaded %d page(s) from %s", len(documents), file_path)
        return documents

    def load_text_as_documents(self, text: str, source: str = "document") -> List[Any]:
        """
        Wrap plain text as LangChain Document objects (useful when text
        has already been extracted by Textract).

        Args:
            text: Full document text.
            source: Metadata source label.

        Returns:
            List containing a single LangChain Document.
        """
        try:
            from langchain_core.documents import Document  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("langchain-core is required: pip install langchain-core") from exc

        return [Document(page_content=text, metadata={"source": source})]

    # ------------------------------------------------------------------
    # Step 2 – Split, embed, and build vector store
    # ------------------------------------------------------------------

    def build_vector_store(
        self,
        documents: List[Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """
        Split documents into chunks, generate Bedrock Titan embeddings,
        and store them in a FAISS index.

        Args:
            documents: LangChain Document objects.
            chunk_size: Max characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            FAISS vector store object.
        """
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: PLC0415
            from langchain_aws import BedrockEmbeddings  # noqa: PLC0415
            from langchain_community.vectorstores import FAISS  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain, langchain-aws, langchain-community, and faiss-cpu "
                "are required."
            ) from exc

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks", len(documents), len(chunks))

        # Create Bedrock Titan embedding model
        embeddings = BedrockEmbeddings(
            client=self._bedrock_runtime,
            model_id=self.embedding_model_id,
        )

        # Build FAISS vector store from document chunks
        self._vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS vector store built with %d chunks", len(chunks))
        return self._vector_store

    # ------------------------------------------------------------------
    # Step 3 – Build RAG Q&A chain
    # ------------------------------------------------------------------

    def build_qa_chain(
        self,
        vector_store: Optional[Any] = None,
        k: int = 4,
        chain_type: str = "stuff",
    ) -> Any:
        """
        Build a LangChain RetrievalQA chain using the Bedrock LLM and
        the FAISS vector store as the retriever.

        Args:
            vector_store: FAISS vector store (uses self._vector_store if None).
            k: Number of documents to retrieve per query.
            chain_type: LangChain chain type ("stuff", "map_reduce", "refine").

        Returns:
            LangChain RetrievalQA chain.
        """
        try:
            from langchain.chains import RetrievalQA  # noqa: PLC0415
            from langchain_aws import ChatBedrock  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain and langchain-aws are required."
            ) from exc

        vs = vector_store or self._vector_store
        if vs is None:
            raise ValueError("Build or provide a vector store first.")

        llm = ChatBedrock(
            client=self._bedrock_runtime,
            model_id=self.bedrock_model_id,
            model_kwargs={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        retriever = vs.as_retriever(search_kwargs={"k": k})

        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
        )
        logger.info(
            "RAG Q&A chain built (model=%s, k=%d, chain=%s)",
            self.bedrock_model_id,
            k,
            chain_type,
        )
        return self._qa_chain

    # ------------------------------------------------------------------
    # Step 4 – Answer questions
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        qa_chain: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Answer a natural-language question using the RAG chain.

        Args:
            question: The question to ask about the document.
            qa_chain: LangChain QA chain (uses self._qa_chain if None).

        Returns:
            Dict with keys:
              - "answer": The LLM's answer string.
              - "source_documents": List of retrieved LangChain Documents.
        """
        chain = qa_chain or self._qa_chain
        if chain is None:
            raise ValueError("Build the QA chain first via build_qa_chain().")

        logger.info("Answering question: %s", question)
        result = chain.invoke({"query": question})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])
        logger.info("Answer: %s... (%d source chunks)", answer[:100], len(sources))
        return {"answer": answer, "source_documents": sources}

    def answer_questions_batch(
        self,
        questions: List[str],
        qa_chain: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Answer a list of questions using the RAG chain.

        Args:
            questions: List of questions.
            qa_chain: LangChain QA chain.

        Returns:
            List of result dicts (same format as answer_question).
        """
        return [
            {"question": q, **self.answer_question(q, qa_chain)}
            for q in questions
        ]

    # ------------------------------------------------------------------
    # Convenience: one-shot pipeline
    # ------------------------------------------------------------------

    def process_and_ask(
        self,
        document_path: str,
        questions: List[str],
        use_s3: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        One-shot pipeline: load document → build vector store → answer questions.

        Args:
            document_path: Local file path or S3 key (if use_s3=True).
            questions: List of questions to answer.
            use_s3: If True, load from S3 using document_path as S3 key.
            chunk_size: Chunk size for text splitting.
            chunk_overlap: Overlap between chunks.
            k: Number of retrieval results per query.

        Returns:
            List of Q&A result dicts.
        """
        if use_s3:
            docs = self.load_document_from_s3(document_path)
        else:
            docs = self.load_document_from_path(document_path)

        self.build_vector_store(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.build_qa_chain(k=k)
        return self.answer_questions_batch(questions)

    # ------------------------------------------------------------------
    # Direct LLM summarisation (no retrieval)
    # ------------------------------------------------------------------

    def summarise_document(
        self,
        text: str,
        max_length_words: int = 200,
    ) -> str:
        """
        Summarise document text directly using the Bedrock LLM (no RAG).

        Args:
            text: Full document text.
            max_length_words: Approximate target summary word count.

        Returns:
            Summary string.
        """
        try:
            from langchain_aws import ChatBedrock  # noqa: PLC0415
            from langchain_core.messages import HumanMessage  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("langchain-aws is required.") from exc

        llm = ChatBedrock(
            client=self._bedrock_runtime,
            model_id=self.bedrock_model_id,
            model_kwargs={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        prompt = (
            f"Please provide a concise summary of the following document "
            f"in approximately {max_length_words} words.\n\nDocument:\n{text[:8000]}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content if hasattr(response, "content") else str(response)
        logger.info("Document summarised (%d chars → %d chars)", len(text), len(summary))
        return summary

    def extract_structured_data(
        self,
        text: str,
        schema_description: str,
    ) -> str:
        """
        Use the Bedrock LLM to extract structured data from document text
        based on a natural-language schema description.

        Args:
            text: Document text.
            schema_description: Description of what to extract, e.g.
                "Extract invoice number, vendor name, total amount, and due date
                 as a JSON object."

        Returns:
            LLM response string (typically JSON).
        """
        try:
            from langchain_aws import ChatBedrock  # noqa: PLC0415
            from langchain_core.messages import HumanMessage  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("langchain-aws is required.") from exc

        llm = ChatBedrock(
            client=self._bedrock_runtime,
            model_id=self.bedrock_model_id,
            model_kwargs={
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            },
        )

        prompt = (
            f"{schema_description}\n\n"
            f"Document text:\n{text[:8000]}\n\n"
            "Return ONLY valid JSON. Do not include explanation."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content if hasattr(response, "content") else str(response)
        logger.info("Structured extraction complete")
        return result
