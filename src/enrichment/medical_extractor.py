"""
Phase 4 – Document Enrichment: Amazon Comprehend Medical
==========================================================
Extract medical entities (medications, diagnoses, anatomy, etc.) and
detect PHI from clinical / medical documents.

Services used:
  - Amazon Comprehend Medical — DetectEntitiesV2, DetectPHI,
    InferICD10CM, InferRxNorm, InferSNOMEDCT
"""

import logging
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)


class MedicalExtractor:
    """
    Extracts medical information from clinical text using
    Amazon Comprehend Medical.

    Args:
        region (str): AWS region.
    """

    def __init__(self, region: str = "us-east-1") -> None:
        self.region = region
        self.comprehend_medical = boto3.client(
            "comprehendmedical", region_name=region
        )

    # ------------------------------------------------------------------
    # General medical entity detection
    # ------------------------------------------------------------------

    def detect_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect medical entities using ComprehendMedical DetectEntitiesV2.

        Detected categories include:
          - MEDICATION (drug names, dosages, routes, etc.)
          - MEDICAL_CONDITION (diagnoses, symptoms, signs)
          - ANATOMY (body parts, systems)
          - TEST_TREATMENT_PROCEDURE (tests, procedures)
          - PROTECTED_HEALTH_INFORMATION (PHI)

        Args:
            text: Clinical text to analyse (max 20 000 UTF-8 characters).

        Returns:
            List of entity dicts containing Category, Type, Text, Score,
            Traits, and Attributes.
        """
        response = self.comprehend_medical.detect_entities_v2(Text=text[:20000])
        entities = response.get("Entities", [])
        logger.info("Detected %d medical entities", len(entities))
        return entities

    def get_entities_by_category(
        self,
        text: str,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect medical entities grouped by category.

        Args:
            text: Clinical text.
            categories: Filter to specific categories. None = all.

        Returns:
            Dict: {category → list of entity dicts}.
        """
        entities = self.detect_medical_entities(text)
        allowed = set(
            categories
            or [
                "MEDICATION",
                "MEDICAL_CONDITION",
                "ANATOMY",
                "TEST_TREATMENT_PROCEDURE",
                "PROTECTED_HEALTH_INFORMATION",
            ]
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in allowed}
        for entity in entities:
            cat = entity.get("Category", "")
            if cat in allowed:
                grouped[cat].append(entity)
        return grouped

    # ------------------------------------------------------------------
    # PHI detection
    # ------------------------------------------------------------------

    def detect_phi(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect Protected Health Information (PHI) in clinical text.

        PHI types include: NAME, ADDRESS, AGE, DATE, PHONE, FAX,
        EMAIL, ID, URL, SSN, ACCOUNT, CERTIFICATE, LICENSE, VEHICLE,
        DEVICE, BIOID, and more.

        Args:
            text: Clinical text.

        Returns:
            List of PHI entity dicts.
        """
        response = self.comprehend_medical.detect_phi(Text=text[:20000])
        phi_entities = response.get("Entities", [])
        logger.info("Detected %d PHI entities", len(phi_entities))
        return phi_entities

    # ------------------------------------------------------------------
    # ICD-10-CM inference
    # ------------------------------------------------------------------

    def infer_icd10cm(self, text: str) -> List[Dict[str, Any]]:
        """
        Link medical conditions in text to ICD-10-CM codes.

        Args:
            text: Clinical text.

        Returns:
            List of condition dicts with linked ICD-10-CM concepts.
        """
        response = self.comprehend_medical.infer_icd10_cm(Text=text[:10000])
        entities = response.get("Entities", [])
        logger.info("ICD-10-CM inferred %d entities", len(entities))
        return entities

    # ------------------------------------------------------------------
    # RxNorm inference
    # ------------------------------------------------------------------

    def infer_rxnorm(self, text: str) -> List[Dict[str, Any]]:
        """
        Link medication mentions in text to RxNorm codes.

        Args:
            text: Clinical text.

        Returns:
            List of medication dicts with linked RxNorm concepts.
        """
        response = self.comprehend_medical.infer_rx_norm(Text=text[:10000])
        entities = response.get("Entities", [])
        logger.info("RxNorm inferred %d entities", len(entities))
        return entities

    # ------------------------------------------------------------------
    # SNOMED CT inference
    # ------------------------------------------------------------------

    def infer_snomed_ct(self, text: str) -> List[Dict[str, Any]]:
        """
        Link medical entities in text to SNOMED CT concepts.

        Args:
            text: Clinical text.

        Returns:
            List of entity dicts with linked SNOMED CT concepts.
        """
        response = self.comprehend_medical.infer_snomedct(Text=text[:10000])
        entities = response.get("Entities", [])
        logger.info("SNOMED CT inferred %d entities", len(entities))
        return entities

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------

    def full_medical_analysis(self, text: str) -> Dict[str, Any]:
        """
        Run a complete medical analysis pipeline on clinical text.

        Returns:
            Dict with keys: entities, phi, icd10cm, rxnorm
        """
        return {
            "entities": self.detect_medical_entities(text),
            "phi": self.detect_phi(text),
            "icd10cm": self.infer_icd10cm(text),
            "rxnorm": self.infer_rxnorm(text),
        }
