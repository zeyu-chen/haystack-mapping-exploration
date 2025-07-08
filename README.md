# Haystack Mapping Exploration

Automated mapping tool for Building Automation System (BAS) point names to Project Haystack standard tags. Provides CLI tools for analyzing, mapping, and evaluating BAS point name patterns.

## Project Structure

```
haystack-mapping-exploration/
├── data/                          # Data files
│   ├── sample_bas_mappings.csv    # Sample BAS point mappings (55 examples)
│   ├── defs.json                  # Haystack standard definitions
│   └── units.txt                  # Units definition
├── src/                           # Source code
│   ├── mapping_utils.py           # Core mapping utilities
│   └── cli_mapper.py              # Command line interface
├── results/                       # Output results
├── ruff.toml                      # Code formatting config
└── pyproject.toml                 # Project configuration
```

## Core Features

- **Smart Mapping**: Pattern matching + concept reasoning hybrid approach
- **Multiple Modes**: Single point mapping, batch processing, interactive analysis
- **Evaluation**: Exact match, semantic similarity, error analysis
- **Equipment Coverage**: 12+ equipment types (AHU, VAV, Chiller, etc.)

## Quick Start

### Installation
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project directory
cd haystack-mapping-exploration
```

### Basic Usage
```bash
# Single point mapping
uv run python src/cli_mapper.py --point-name "AHU-1_SAT"

# Batch processing
uv run python src/cli_mapper.py -i data/sample_bas_mappings.csv --evaluate

# Interactive mode
uv run python src/cli_mapper.py --interactive
```

## Performance Metrics

### Current Results (55 sample points)
- **Exact Match Accuracy**: 44.6%
- **Average Semantic Similarity**: 63.2%
- **Equipment Type Coverage**: 12 different equipment types
- **High Confidence Mappings**: 64%

### Main Challenges
- **Concept Matching Errors**: 87.1% of errors from concept matching
- **Insufficient Equipment Patterns**: FCU, Boiler lack dedicated patterns
- **Point Type Confusion**: SP vs Sensor identification errors
- **Abbreviation Ambiguity**: Multiple meanings for WS/WD abbreviations

## Data Format

### Input CSV
```csv
original_point_name,standard_tags,equipment_type,point_type
AHU-1_SAT,dis:"AHU-1 Supply Air Temp" ahu supply air temp sensor point,ahu,sensor
```

### Output Result
```json
{
  "original_name": "AHU-1_SAT",
  "equipment_type": "ahu", 
  "mapped_tags": "ahu supply air temp sensor point",
  "confidence": 1.0,
  "method": "pattern_based"
}
```

## Mapping Approaches

### 1. Pattern Matching (Primary)
- Pre-compiled regex patterns
- Equipment-specific template matching
- 95%+ accuracy for recognized equipment

### 2. Concept Reasoning (Fallback)
- 20+ semantic concept extraction
- Abbreviation dictionary matching
- Hierarchical tag construction

## Improvement Plan

### Short-term Goals
1. **Expand Pattern Library**: Add dedicated patterns for FCU, Boiler
2. **Improve Point Type ID**: Enhance SP vs Sensor recognition
3. **Optimize Concept Extraction**: Resolve WS/WD ambiguity issues

### Expected Impact
- **Pattern Matching**: 4 errors → 0 errors (+7.1%)
- **Concept Matching**: 27 errors → 10 errors (+30.4%)
- **Overall Improvement**: 44.6% → 75%+ exact match accuracy

## Haystack Standard Integration

### Current Status
- **Standardization**: 95% (38/40 tags comply with Haystack standard)
- **Database Size**: 719 official standard definitions
- **Relevant Tags**: 119 HVAC-related tags available

### Integration Value
1. **Eliminate Ambiguity**: Standardized tag definitions
2. **Semantic Reasoning**: Smart mapping based on tag relationships
3. **Interoperability**: Compatible with Haystack ecosystem
4. **Auto-expansion**: Reduce hard-coding, support new equipment types

## Development Tools

```bash
# Code formatting
uv run ruff format src/

# Code linting
uv run ruff check src/ --fix

# Run tests
uv run python test_examples.py
```

## Tech Stack

- **Python**: 3.11+
- **Core Libraries**: pandas, scikit-learn, numpy
- **Tools**: uv (package management), ruff (code quality)
- **Standard**: Project Haystack

---

**Version**: 0.1.0 | **Updated**: 2025-07-08 | **Python**: 3.11+