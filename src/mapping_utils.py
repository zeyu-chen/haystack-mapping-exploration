"""
Utility functions for BAS point name mapping.

This module provides comprehensive tools for analyzing BAS point names
and mapping them to standardized Project Haystack tags.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PointNameAnalyzer:
    """Analyze and extract features from BAS point names."""

    def __init__(self):
        """Initialize analyzer with comprehensive pattern dictionaries."""
        self.common_abbreviations = {
            "temp": ["T", "TEMP", "TMP", "TEMPERATURE"],
            "pressure": ["P", "PRESS", "PRESSURE", "PSI"],
            "flow": ["F", "FLOW", "FLW", "RATE", "CFM", "GPM"],
            "command": ["C", "CMD", "COMMAND", "CTRL", "CONTROL"],
            "status": ["S", "STAT", "STATUS", "STATE", "ENABLE"],
            "setpoint": ["SP", "SET", "SETPOINT", "TARGET", "DESIRED"],
            "air": ["AIR", "A"],
            "water": ["WATER", "W", "H2O", "CHW", "HW"],
            "supply": ["SUP", "SUPPLY", "S", "LEAVING"],
            "return": ["RET", "RETURN", "R", "ENTERING"],
            "mixed": ["MIX", "MIXED", "M"],
            "outside": ["OUT", "OUTSIDE", "OUTDOOR", "OA", "AMBIENT"],
            "zone": ["ZONE", "Z", "ROOM", "SPACE"],
            "fan": ["FAN", "F", "BLOWER"],
            "damper": ["DAMPER", "DPR", "DAMP", "ACTUATOR"],
            "valve": ["VALVE", "VLV", "V"],
            "sensor": ["SENSOR", "SNS", "S", "FEEDBACK", "FB"],
            "power": ["POWER", "PWR", "KW", "WATT", "KILOWATT"],
            "energy": ["ENERGY", "KWH", "WATT_HR", "KILOWATT_HR"],
            "humidity": ["HUMIDITY", "RH", "HUMID"],
            "position": ["POS", "POSITION", "PERCENT", "PCT"],
            "speed": ["SPEED", "RPM", "FREQ", "FREQUENCY", "HZ"],
            "alarm": ["ALARM", "ALM", "ALERT", "FAULT"],
            "level": ["LEVEL", "LVL", "HEIGHT"],
            "discharge": ["DISCHARGE", "DISCH", "DCH"],
            "entering": ["ENTERING", "ENT", "INLET"],
            "leaving": ["LEAVING", "LVG", "OUTLET"],
        }

        self.equipment_patterns = {
            "ahu": [r"AHU[-_]?\d*", r"AIR[-_]?HANDLER", r"AIRHANDLER", r"OAHU"],
            "vav": [r"VAV[-_]?\d*", r"VAV[-_]?BOX"],
            "fcu": [r"FCU[-_]?\d*", r"FAN[-_]?COIL"],
            "chiller": [r"CH[-_]?\d*", r"CHILLER", r"CHILL"],
            "boiler": [r"BOILER", r"BLR[-_]?\d*"],
            "pump": [r"PUMP[-_]?\d*", r"P[-_]?\d+"],
            "coolingTower": [r"CT[-_]?\d*", r"COOLING[-_]?TOWER"],
            "heatExchanger": [r"HX[-_]?\d*", r"HEAT[-_]?EXCHANGER"],
            "light": [r"LIGHT[-_]?\d*", r"LT[-_]?\d*"],
            "elevator": [r"ELEV[-_]?\d*", r"ELEVATOR"],
            "unitHeater": [r"UH[-_]?\d*", r"UNIT[-_]?HEATER"],
            "weather": [r"WEATHER", r"WS", r"WD", r"OAT", r"OAH", r"OAP"],
            "meter": [r"METER", r"MTR", r"B[-_]?\d+"],
            "thermostat": [r"TSTAT", r"THERMOSTAT", r"T[-_]?\d+"],
            "site": [r"SITE", r"BUILDING", r"B[-_]?\d*"],
            "schedule": [r"SCHED", r"SCHEDULE"],
        }

        self.point_type_indicators = {
            "sensor": ["SENSOR", "SNS", "FEEDBACK", "FB", "ACTUAL"],
            "cmd": ["CMD", "COMMAND", "CONTROL", "OUTPUT", "OUT"],
            "sp": ["SP", "SETPOINT", "SET", "TARGET", "DESIRED"],
        }

        # Pre-compile regex patterns for performance
        self._compiled_patterns = {}
        for equip_type, patterns in self.equipment_patterns.items():
            self._compiled_patterns[equip_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def extract_equipment_number(self, point_name: str) -> str | None:
        """Extract equipment number from point name."""
        # Look for patterns like AHU-1, VAV_101, etc.
        match = re.search(r"[-_](\d+)", point_name)
        return match.group(1) if match else None

    def identify_equipment_type(self, point_name: str) -> str | None:
        """Identify equipment type from point name using compiled patterns."""
        for equip_type, compiled_patterns in self._compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(point_name):
                    return equip_type
        return None

    def extract_concepts(self, point_name: str) -> list[str]:
        """Extract semantic concepts from point name."""
        point_upper = point_name.upper()
        found_concepts = []

        for concept, abbreviations in self.common_abbreviations.items():
            for abbrev in abbreviations:
                if abbrev in point_upper:
                    found_concepts.append(concept)
                    break

        return found_concepts

    def identify_point_type(self, point_name: str) -> str | None:
        """Identify point type (sensor, cmd, sp) from point name."""
        point_upper = point_name.upper()

        for point_type, indicators in self.point_type_indicators.items():
            for indicator in indicators:
                if indicator in point_upper:
                    return point_type

        # Default heuristics
        if any(word in point_upper for word in ["STATUS", "TEMP", "PRESSURE", "FLOW"]):
            return "sensor"
        elif any(word in point_upper for word in ["CMD", "COMMAND"]):
            return "cmd"
        elif any(word in point_upper for word in ["SP", "SET"]):
            return "sp"

        return "sensor"  # Default assumption

    def parse_point_name(self, point_name: str) -> dict:
        """Parse point name and extract all relevant information."""
        return {
            "original_name": point_name,
            "equipment_type": self.identify_equipment_type(point_name),
            "equipment_number": self.extract_equipment_number(point_name),
            "point_type": self.identify_point_type(point_name),
            "concepts": self.extract_concepts(point_name),
            "name_length": len(point_name),
            "has_numbers": bool(re.search(r"\d", point_name)),
            "has_underscores": "_" in point_name,
            "has_hyphens": "-" in point_name,
            "segments": re.split(r"[-_]", point_name),
            "uppercase_ratio": sum(1 for c in point_name if c.isupper())
            / len(point_name),
            "digit_ratio": sum(1 for c in point_name if c.isdigit()) / len(point_name),
        }


class HaystackTagMapper:
    """Map BAS point names to Haystack tags."""

    def __init__(self):
        """Initialize mapper with comprehensive tag templates."""
        self.analyzer = PointNameAnalyzer()
        self.tag_templates = self._load_tag_templates()
        self.confidence_weights = {
            "exact_match": 1.0,
            "equipment_match": 0.8,
            "concept_match": 0.6,
            "pattern_match": 0.4,
            "fallback": 0.2,
        }

    def _load_tag_templates(self) -> dict:
        """Load comprehensive tag templates for different equipment types."""
        return {
            "ahu": {
                "supply_air_temp": {
                    "tags": "ahu supply air temp sensor point",
                    "patterns": ["SAT", "SUPPLY.*TEMP", "SUPPLY.*AIR.*TEMP"],
                    "concepts": ["supply", "air", "temp"],
                },
                "return_air_temp": {
                    "tags": "ahu return air temp sensor point",
                    "patterns": ["RAT", "RETURN.*TEMP", "RETURN.*AIR.*TEMP"],
                    "concepts": ["return", "air", "temp"],
                },
                "mixed_air_temp": {
                    "tags": "ahu mixed air temp sensor point",
                    "patterns": ["MAT", "MIXED.*TEMP", "MIXED.*AIR.*TEMP"],
                    "concepts": ["mixed", "air", "temp"],
                },
                "supply_air_flow": {
                    "tags": "ahu supply air flow sensor point",
                    "patterns": ["SAF", "SUPPLY.*FLOW", "SUPPLY.*AIR.*FLOW"],
                    "concepts": ["supply", "air", "flow"],
                },
                "return_air_flow": {
                    "tags": "ahu return air flow sensor point",
                    "patterns": ["RAF", "RETURN.*FLOW", "RETURN.*AIR.*FLOW"],
                    "concepts": ["return", "air", "flow"],
                },
                "supply_air_pressure": {
                    "tags": "ahu supply air pressure sensor point",
                    "patterns": ["SAP", "SUPPLY.*PRESS", "SUPPLY.*AIR.*PRESS"],
                    "concepts": ["supply", "air", "pressure"],
                },
                "fan_status": {
                    "tags": "ahu fan run sensor point",
                    "patterns": [".*FAN.*STAT", ".*FAN.*STATUS", "SFS"],
                    "concepts": ["fan", "status"],
                },
                "fan_command": {
                    "tags": "ahu fan run cmd point",
                    "patterns": [".*FAN.*CMD", ".*FAN.*COMMAND", "SFC"],
                    "concepts": ["fan", "command"],
                },
                "cool_command": {
                    "tags": "ahu cool cmd point",
                    "patterns": ["CC", "COOL.*CMD", "COOLING.*CMD"],
                    "concepts": ["command"],
                },
                "heat_command": {
                    "tags": "ahu heat cmd point",
                    "patterns": ["HC", "HEAT.*CMD", "HEATING.*CMD"],
                    "concepts": ["command"],
                },
                "outside_damper": {
                    "tags": "ahu outside damper cmd point",
                    "patterns": ["OAD", "OUTSIDE.*DAMP", ".*DAMPER.*CMD"],
                    "concepts": ["outside", "damper", "command"],
                },
            },
            "vav": {
                "zone_temp": {
                    "tags": "vav zone air temp sensor point",
                    "patterns": ["ZT", "ZONE.*TEMP", "ROOM.*TEMP"],
                    "concepts": ["zone", "air", "temp"],
                },
                "zone_temp_setpoint": {
                    "tags": "vav zone air temp sp point",
                    "patterns": ["ZTS", "ZONE.*TEMP.*SP", "ZONE.*TEMP.*SET"],
                    "concepts": ["zone", "air", "temp", "setpoint"],
                },
                "damper_position": {
                    "tags": "vav discharge damper cmd point",
                    "patterns": ["DPR", "DAMPER.*POS", ".*DAMPER.*CMD"],
                    "concepts": ["damper", "command"],
                },
                "air_flow": {
                    "tags": "vav discharge air flow sensor point",
                    "patterns": ["AF", "AIR.*FLOW", ".*FLOW"],
                    "concepts": ["air", "flow"],
                },
                "air_flow_setpoint": {
                    "tags": "vav discharge air flow sp point",
                    "patterns": ["AFS", "AIR.*FLOW.*SP", ".*FLOW.*SET"],
                    "concepts": ["air", "flow", "setpoint"],
                },
                "reheat_command": {
                    "tags": "vav reheat valve cmd point",
                    "patterns": ["REHEAT.*CMD", "RH.*CMD", "HEAT.*CMD"],
                    "concepts": ["valve", "command"],
                },
            },
            "chiller": {
                "supply_temp": {
                    "tags": "chiller leaving chilled water temp sensor point",
                    "patterns": ["CHWS", "CHW.*SUP", "LEAVING.*TEMP"],
                    "concepts": ["supply", "water", "temp"],
                },
                "return_temp": {
                    "tags": "chiller entering chilled water temp sensor point",
                    "patterns": ["CHWR", "CHW.*RET", "ENTERING.*TEMP"],
                    "concepts": ["return", "water", "temp"],
                },
                "flow": {
                    "tags": "chiller chilled water flow sensor point",
                    "patterns": ["CHW.*FLOW", "FLOW"],
                    "concepts": ["water", "flow"],
                },
                "power": {
                    "tags": "chiller elec power sensor point",
                    "patterns": ["POWER", "KW", "KILOWATT"],
                    "concepts": ["power"],
                },
                "status": {
                    "tags": "chiller run sensor point",
                    "patterns": ["STATUS", "STAT", "RUN"],
                    "concepts": ["status"],
                },
                "command": {
                    "tags": "chiller run cmd point",
                    "patterns": ["CMD", "COMMAND", "ENABLE"],
                    "concepts": ["command"],
                },
            },
            "pump": {
                "status": {
                    "tags": "pump run sensor point",
                    "patterns": ["STATUS", "STAT", "RUN"],
                    "concepts": ["status"],
                },
                "command": {
                    "tags": "pump run cmd point",
                    "patterns": ["CMD", "COMMAND", "ENABLE"],
                    "concepts": ["command"],
                },
                "speed": {
                    "tags": "pump speed sensor point",
                    "patterns": ["SPEED", "RPM"],
                    "concepts": ["speed"],
                },
                "frequency": {
                    "tags": "pump freq sensor point",
                    "patterns": ["FREQ", "FREQUENCY", "HZ"],
                    "concepts": ["speed"],
                },
            },
            "weather": {
                "outside_air_temp": {
                    "tags": "outside air temp sensor point",
                    "patterns": ["OAT", "OUTSIDE.*TEMP", "AMBIENT.*TEMP"],
                    "concepts": ["outside", "air", "temp"],
                },
                "outside_air_humidity": {
                    "tags": "outside air humidity sensor point",
                    "patterns": ["OAH", "OUTSIDE.*HUM", "AMBIENT.*HUM"],
                    "concepts": ["outside", "air", "humidity"],
                },
                "outside_air_pressure": {
                    "tags": "outside air pressure sensor point",
                    "patterns": ["OAP", "OUTSIDE.*PRESS", "ATMOSPHERIC.*PRESS"],
                    "concepts": ["outside", "air", "pressure"],
                },
                "wind_speed": {
                    "tags": "wind speed sensor point",
                    "patterns": ["WS", "WIND.*SPEED"],
                    "concepts": ["speed"],
                },
                "wind_direction": {
                    "tags": "wind direction sensor point",
                    "patterns": ["WD", "WIND.*DIR"],
                    "concepts": [],
                },
            },
        }

    def map_to_haystack(
        self, point_name: str, confidence_threshold: float = 0.7
    ) -> dict:
        """Map point name to Haystack tags with confidence scoring."""
        parsed = self.analyzer.parse_point_name(point_name)

        result = {
            "original_name": point_name,
            "equipment_type": parsed["equipment_type"],
            "point_type": parsed["point_type"],
            "mapped_tags": "",
            "confidence": 0.0,
            "method": "unknown",
            "debug_info": {},
        }

        # Try pattern-based mapping first
        if parsed["equipment_type"] and parsed["equipment_type"] in self.tag_templates:
            best_match, confidence = self._pattern_based_mapping(
                point_name, parsed, parsed["equipment_type"]
            )

            if best_match and confidence >= confidence_threshold:
                result["mapped_tags"] = best_match
                result["confidence"] = confidence
                result["method"] = "pattern_based"
                result["debug_info"]["matched_patterns"] = True
                return result

        # Fallback to concept-based mapping
        concept_tags, concept_confidence = self._concept_based_mapping(parsed)
        result["mapped_tags"] = concept_tags
        result["confidence"] = concept_confidence
        result["method"] = "concept_based"
        result["debug_info"]["concepts_used"] = parsed["concepts"]

        return result

    def _pattern_based_mapping(
        self, point_name: str, parsed: dict, equipment_type: str
    ) -> tuple[str | None, float]:
        """Perform pattern-based mapping for specific equipment type."""
        point_upper = point_name.upper()
        equipment_templates = self.tag_templates[equipment_type]

        best_match = None
        best_score = 0.0

        for _template_name, template_data in equipment_templates.items():
            score = 0.0

            # Check pattern matches
            for pattern in template_data["patterns"]:
                if re.search(pattern, point_upper):
                    score += 0.5

            # Check concept matches
            template_concepts = template_data["concepts"]
            point_concepts = parsed["concepts"]

            if template_concepts and point_concepts:
                concept_overlap = len(
                    set(template_concepts).intersection(set(point_concepts))
                )
                concept_score = concept_overlap / len(template_concepts)
                score += concept_score * 0.5

            if score > best_score:
                best_score = score
                best_match = template_data["tags"]

        return best_match, min(best_score, 1.0)

    def _concept_based_mapping(self, parsed: dict) -> tuple[str, float]:
        """Fallback concept-based mapping when pattern matching fails."""
        tags = []
        confidence = self.confidence_weights["fallback"]

        # Add equipment type
        if parsed["equipment_type"]:
            tags.append(parsed["equipment_type"])
            confidence += 0.2

        # Add main concepts with priority
        concepts = parsed["concepts"]
        priority_concepts = ["temp", "pressure", "flow", "power", "energy"]

        for concept in priority_concepts:
            if concept in concepts:
                tags.append(concept)
                confidence += 0.1
                break

        # Add medium/substance
        if "air" in concepts:
            tags.append("air")
        elif "water" in concepts:
            tags.append("water")

        # Add location/direction
        if "supply" in concepts:
            tags.append("supply")
        elif "return" in concepts:
            tags.append("return")
        elif "outside" in concepts:
            tags.append("outside")
        elif "zone" in concepts:
            tags.append("zone")

        # Add point type
        if parsed["point_type"]:
            if parsed["point_type"] == "cmd":
                tags.append("cmd point")
            elif parsed["point_type"] == "sp":
                tags.append("sp point")
            else:
                tags.append("sensor point")
            confidence += 0.1

        return " ".join(tags) if tags else "unknown point", min(confidence, 1.0)

    def batch_map(
        self, point_names: list[str], confidence_threshold: float = 0.7
    ) -> list[dict]:
        """Map multiple point names efficiently."""
        return [
            self.map_to_haystack(name, confidence_threshold) for name in point_names
        ]


class MappingEvaluator:
    """Evaluate mapping performance with multiple metrics."""

    def __init__(self):
        """Initialize evaluator with TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), stop_words="english"
        )

    def semantic_similarity(self, predicted: str, actual: str) -> float:
        """Calculate semantic similarity between predicted and actual tags."""
        try:
            # Handle empty strings
            if not predicted.strip() or not actual.strip():
                return 0.0

            # Vectorize the tag strings
            vectors = self.vectorizer.fit_transform([predicted, actual])

            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(similarity)
        except Exception:
            # Fallback to simple string matching
            return self._simple_similarity(predicted, actual)

    def _simple_similarity(self, predicted: str, actual: str) -> float:
        """Simple word overlap similarity as fallback."""
        if not predicted.strip() and not actual.strip():
            return 1.0
        if not predicted.strip() or not actual.strip():
            return 0.0

        pred_words = set(predicted.lower().split())
        actual_words = set(actual.lower().split())

        intersection = pred_words.intersection(actual_words)
        union = pred_words.union(actual_words)

        return len(intersection) / len(union) if union else 0.0

    def _extract_core_tags(self, tag_string: str) -> str:
        """Extract core Haystack tags from full tag string."""
        # Remove dis: prefix and quoted description if present
        # Example: 'dis:"AHU-1 Supply Air Temp" ahu supply air temp sensor point'
        # -> 'ahu supply air temp sensor point'

        if "dis:" in tag_string:
            # Find the end of the dis: section (after closing quote)
            parts = tag_string.split('" ', 1)
            if len(parts) == 2:
                return parts[1].strip()
            else:
                # Fallback: remove everything up to first space after dis:
                parts = tag_string.split(" ", 1)
                if len(parts) == 2:
                    return parts[1].strip()

        return tag_string.strip()

    def evaluate_mappings(
        self, predictions: list[str], actuals: list[str]
    ) -> dict[str, float | list[float]]:
        """Evaluate mapping performance with comprehensive metrics."""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        similarities = []
        exact_matches = 0
        partial_matches = 0

        for pred, actual in zip(predictions, actuals, strict=False):
            # Extract core tags from actual (remove dis: prefix and description)
            clean_actual = self._extract_core_tags(actual)

            similarity = self.semantic_similarity(pred, clean_actual)
            similarities.append(similarity)

            if pred.lower().strip() == clean_actual.lower().strip():
                exact_matches += 1
            elif similarity > 0.5:  # Threshold for partial match
                partial_matches += 1

        return {
            "exact_match_accuracy": exact_matches / len(predictions),
            "partial_match_accuracy": (exact_matches + partial_matches)
            / len(predictions),
            "average_similarity": float(np.mean(similarities)),
            "median_similarity": float(np.median(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "std_similarity": float(np.std(similarities)),
            "similarities": similarities,
            "total_predictions": len(predictions),
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
        }

    def confusion_matrix_analysis(
        self, predictions: list[str], actuals: list[str]
    ) -> dict:
        """Analyze common prediction errors."""
        error_patterns = {}
        equipment_errors = {}

        for pred, actual in zip(predictions, actuals, strict=False):
            if pred.lower() != actual.lower():
                # Extract equipment types for error analysis
                pred_equip = self._extract_equipment_type(pred)
                actual_equip = self._extract_equipment_type(actual)

                if pred_equip != actual_equip:
                    key = f"{actual_equip} -> {pred_equip}"
                    equipment_errors[key] = equipment_errors.get(key, 0) + 1

                # Track specific error patterns
                error_key = f"'{actual}' predicted as '{pred}'"
                error_patterns[error_key] = error_patterns.get(error_key, 0) + 1

        return {
            "equipment_type_errors": dict(
                sorted(equipment_errors.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "common_errors": dict(
                sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

    def _extract_equipment_type(self, tag_string: str) -> str:
        """Extract equipment type from tag string."""
        common_equipment = ["ahu", "vav", "chiller", "pump", "boiler", "fcu"]
        for equip in common_equipment:
            if equip in tag_string.lower():
                return equip
        return "unknown"


def load_haystack_definitions(file_path: str) -> pd.DataFrame:
    """Load Haystack definitions from JSON file."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Warning: File {file_path} not found")
            return pd.DataFrame()

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract relevant information
        definitions = []
        if "rows" in data:
            for row in data["rows"]:
                if "def" in row:
                    definitions.append(
                        {
                            "def": row.get("def", ""),
                            "dis": row.get("dis", ""),
                            "doc": row.get("doc", ""),
                            "is": row.get("is", ""),
                            "lib": row.get("lib", ""),
                        }
                    )

        return pd.DataFrame(definitions)
    except Exception as e:
        print(f"Error loading Haystack definitions: {e}")
        return pd.DataFrame()


def create_sample_mappings(n_samples: int = 100) -> pd.DataFrame:
    """Create additional sample mappings for testing (placeholder)."""
    # This would typically load from a larger database
    # For now, return empty DataFrame
    return pd.DataFrame()


def export_mapping_results(
    results: list[dict], output_file: str, format_type: str = "csv"
) -> None:
    """Export mapping results to file."""
    df = pd.DataFrame(results)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type.lower() == "json" or output_path.suffix == ".json":
        df.to_json(output_file, orient="records", indent=2)
    elif format_type.lower() == "csv" or output_path.suffix == ".csv":
        df.to_csv(output_file, index=False)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def validate_mapping_quality(results: list[dict]) -> dict[str, int | float]:
    """Validate overall mapping quality."""
    total_mappings = len(results)
    high_confidence = sum(1 for r in results if r["confidence"] > 0.8)
    medium_confidence = sum(1 for r in results if 0.5 <= r["confidence"] <= 0.8)
    low_confidence = sum(1 for r in results if r["confidence"] < 0.5)

    equipment_coverage = len(
        {r["equipment_type"] for r in results if r["equipment_type"]}
    )

    return {
        "total_mappings": total_mappings,
        "high_confidence_count": high_confidence,
        "medium_confidence_count": medium_confidence,
        "low_confidence_count": low_confidence,
        "high_confidence_ratio": high_confidence / total_mappings,
        "equipment_types_identified": equipment_coverage,
        "average_confidence": sum(r["confidence"] for r in results) / total_mappings,
    }
