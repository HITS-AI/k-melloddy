#!/usr/bin/env python3
"""
Manual (non-LLM) endpoint converter.

Provides a deterministic fallback for mapping K-MELLODDY endpoints to GIST endpoints
when LLM APIs are unavailable. Mapping uses a combination of curated synonyms and
string similarity heuristics.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_ENDPOINT_SYNONYMS: Dict[str, str] = {
    # Permeability assays
    "caco2": "Caco2",
    "caco 2": "Caco2",
    "caco-2": "Caco2",
    "pampa": "PAMPA_pH7.4(bc)",
    "pampa ph74": "PAMPA_pH7.4(bc)",
    "pampa ph7.4": "PAMPA_pH7.4(bc)",
    "pampa ph75": "PAMPA_pH7.4(bc)",
    "pampa apical": "PAMPA_pH7.4(mc)",
    "pampa basolateral": "PAMPA_pH7.4(bc)",
    "mdck": "Efflux_ratio",
    "skin permeability": "skin_permeability",
    "bia skin permeability": "skin_permeability",
    "hia": "HIA",
    "human intestinal absorption": "HIA",
    "bbb": "BBB_logbb(cls)",
    "blood brain barrier": "BBB_logbb(cls)",
    "brain penetration": "BBB_logbb(cls)",
    "vdss": "VDss",
    # Transporters and pumps
    "pgp inhibitor": "pgp_inhibitor",
    "pgp substrate": "pgp_substrate",
    "p-gp substrate": "pgp_substrate",
    "p-gp inhibitor": "pgp_inhibitor",
    "bcrp inhibitor": "BCRP",
    "oatp1b1": "OATP1B1_Inhibitor",
    "oatp1b3": "OATP1B3_Inhibitor",
    "oatp2b1": "OATP2B1_Inhibitor",
    "mate1": "MATE1_Inhibitor",
    "oct2": "OCT2_Inhibitor",
    # CYPs
    "cyp1a2": "CYP1A2_Inhibitor",
    "cyp2b6": "CYP2B6_Inhibitor",
    "cyp2c9": "CYP2C9_Inhibitor",
    "cyp2c19": "CYP2C19_Inhibitor",
    "cyp2d6": "CYP2D6_Inhibitor",
    "cyp3a4": "CYP3A4_Inhibitor",
    "cyp1a2 substrate": "CYP1A2_Substrate",
    "cyp2b6 substrate": "CYP2B6_Substrate",
    "cyp2c9 substrate": "CYP2C9_Substrate",
    "cyp2c19 substrate": "CYP2C19_Substrate",
    "cyp2d6 substrate": "CYP2D6_Substrate",
    "cyp3a4 substrate": "CYP3A4_Substrate",
    # Toxicity & safety
    "skin reaction": "Skin Reaction",
    "respiratory toxicity": "Respir_tox",
    "liver toxicity": "Liver_tox_hepato",
    "dili": "DILI",
    "herg": "hERG",
    "ames": "AMES",
    "carcinogenicity": "Carcinogen",
    "neurotoxicity": "Neuro_tox",
    "nephrotoxicity": "Nephro_tox",
    "mitochondrial toxicity": "Mito_tox",
    "hemolytic": "Hemolytic",
    "reproductive toxicity": "Reprotox",
    "eye irritation": "Eye_irritation",
    "eye corrosion": "Eye_corrosion",
    "skin irritation": "Skin Reaction",
    "micronucleus": "Micronucleus",
    "fdamdd": "FDAMDD(reg)",
    # Misc
    "plasma protein binding": "PPBR",
    "ppb": "PPBR",
    "clearance": "Clearance_total",
    "intrinsic clearance": "Clearance_Hepatocyte_AZ",
    "clp": "CLp(r)",
    "mrt": "MRT",
    "ugt": "UGT_substrate",
    "gr": "GR",
    "tr": "TR",
}


@dataclass
class ManualConversionConfig:
    """
    Configuration for ManualFormatConverter.

    Attributes:
        mapping_path: Optional path to CSV/JSON with additional endpoint synonyms.
        min_similarity: Minimum score (0-1) required to accept automatic similarity match.
        prefer_exact: If True, exact token overlap matches trump sequence scores.
    """

    mapping_path: Optional[str] = None
    min_similarity: float = 0.55
    prefer_exact: bool = True


class ManualFormatConverter:
    """Endpoint converter using deterministic heuristics."""

    def __init__(self, config: Optional[ManualConversionConfig] = None):
        self.config = config or ManualConversionConfig()
        self.gist_endpoints = self._load_gist_endpoints()
        self._normalized_gist = {
            endpoint: self._normalize(endpoint) for endpoint in self.gist_endpoints
        }
        self._gist_tokens = {endpoint: self._tokenize(norm)
                             for endpoint, norm in self._normalized_gist.items()}
        self.synonym_map = self._load_synonym_map()

    def _load_gist_endpoints(self) -> List[str]:
        gist_file = Path(__file__).resolve().parents[2] / "gist" / "gist_format.txt"
        if not gist_file.exists():
            raise FileNotFoundError(
                f"GIST endpoint file not found at expected path: {gist_file}"
            )
        with open(gist_file, "r", encoding="utf-8") as handler:
            first_line = handler.readline().strip()
            if not first_line:
                raise ValueError("GIST endpoint list is empty.")
            return [column for column in first_line.split("\t") if column]

    def _load_synonym_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for key, value in DEFAULT_ENDPOINT_SYNONYMS.items():
            mapping[self._normalize(key)] = value

        if self.config.mapping_path:
            path = Path(self.config.mapping_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Manual mapping file not found: {path}")
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as handler:
                    file_map = json.load(handler)
                for key, value in file_map.items():
                    mapping[self._normalize(key)] = value
            else:
                with open(path, "r", encoding="utf-8") as handler:
                    reader = csv.DictReader(handler)
                    if "source" not in reader.fieldnames or "target" not in reader.fieldnames:
                        raise ValueError(
                            f"Mapping CSV must contain 'source' and 'target' columns: {path}"
                        )
                    for row in reader:
                        mapping[self._normalize(row["source"])] = row["target"]
        return mapping

    @staticmethod
    def _normalize(value: Any) -> str:
        text = str(value).lower()
        text = text.replace("μ", "u").replace("µ", "u")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _tokenize(normalized: str) -> Tuple[str, ...]:
        if not normalized:
            return tuple()
        return tuple(token for token in normalized.split(" ") if token)

    @staticmethod
    def _sequence_score(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        # Lightweight sequence similarity (Dice coefficient on bigrams)
        def bigrams(text: str) -> List[str]:
            return [text[i:i + 2] for i in range(len(text) - 1)] or [text]

        bigrams_a = bigrams(a)
        bigrams_b = bigrams(b)
        overlap = len(set(bigrams_a) & set(bigrams_b))
        score = (2.0 * overlap) / (len(bigrams_a) + len(bigrams_b))
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _token_overlap(tokens_a: Tuple[str, ...], tokens_b: Tuple[str, ...]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    def _match_with_similarity(self, normalized: str, tokens: Tuple[str, ...]) -> Optional[str]:
        best_score = 0.0
        best_endpoint: Optional[str] = None

        for endpoint, gist_tokens in self._gist_tokens.items():
            seq_score = self._sequence_score(normalized, self._normalized_gist[endpoint])
            overlap_score = self._token_overlap(tokens, gist_tokens)
            combined = max(seq_score, overlap_score)

            # Encourage leading token match
            if tokens and gist_tokens and tokens[0] == gist_tokens[0]:
                combined += 0.1

            if combined > best_score:
                best_score = combined
                best_endpoint = endpoint

        if best_score >= self.config.min_similarity:
            logger.debug(
                "Similarity match: `%s` -> `%s` (score=%.2f)",
                normalized, best_endpoint, best_score
            )
            return best_endpoint

        return None

    def match_endpoint(self, endpoint: Any) -> str:
        normalized = self._normalize(endpoint)
        if not normalized:
            return str(endpoint)

        tokens = self._tokenize(normalized)

        # Direct synonym mapping
        if normalized in self.synonym_map:
            return self.synonym_map[normalized]

        # Partial keyword lookup (token-level)
        for synonym_key, mapped in self.synonym_map.items():
            if self.config.prefer_exact and synonym_key == normalized:
                return mapped
            if synonym_key in normalized:
                return mapped

        # Similarity fallback
        matched = self._match_with_similarity(normalized, tokens)
        if matched:
            return matched

        # Final fallback: return original endpoint
        return str(endpoint)

    def match_endpoints(self, df: pd.DataFrame, endpoint_column: str = "endpoint") -> Dict[str, str]:
        if endpoint_column not in df.columns:
            raise ValueError(f"Endpoint column '{endpoint_column}' not found in dataframe.")

        unique_endpoints = df[endpoint_column].dropna().unique()
        mapping: Dict[str, str] = {}
        for endpoint in unique_endpoints:
            mapping[str(endpoint)] = self.match_endpoint(endpoint)
        return mapping

    def convert_to_gist_format(self, df: pd.DataFrame) -> pd.DataFrame:
        converted_df = df.copy()

        endpoint_column: Optional[str] = None
        for candidate in ("Test", "endpoint"):
            if candidate in df.columns:
                endpoint_column = candidate
                break

        endpoint_mapping: Dict[str, str] = {}
        if endpoint_column:
            mapping_frame = df[[endpoint_column]].dropna().drop_duplicates()
            endpoint_mapping = self.match_endpoints(mapping_frame, endpoint_column=endpoint_column)

        rename_map = {
            "Chemical ID": "compound_id",
            "Chemical Name": "compound_name",
            "SMILES_Structure_Parent": "smiles",
            "Test": "original_endpoint",
            "Test_Type": "test_type",
            "Test_Subject": "test_subject",
            "Measurement_Type": "measurement_type",
            "Measurement_Value": "activity_value",
            "Measurement_Unit": "activity_unit",
        }

        converted_df = converted_df.rename(columns={k: v for k, v in rename_map.items() if k in converted_df.columns})

        source_column = "original_endpoint" if "original_endpoint" in converted_df.columns else endpoint_column
        if endpoint_mapping and source_column and source_column in converted_df.columns:
            converted_df["gist_endpoint"] = converted_df[source_column].map(endpoint_mapping)
            converted_df[source_column] = converted_df[source_column].map(endpoint_mapping)

        if "activity_value" in converted_df.columns:
            converted_df["activity_value"] = pd.to_numeric(converted_df["activity_value"], errors="coerce")

        return converted_df


__all__ = ["ManualConversionConfig", "ManualFormatConverter"]
