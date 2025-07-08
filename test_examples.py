#!/usr/bin/env python3
"""
Test examples for the Haystack mapping CLI tools.

This script demonstrates various usage patterns of the optimized CLI tools.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and display the results."""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            cwd=Path(__file__).parent
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    """Run test examples."""
    print("Haystack Mapping CLI Tool Test Examples")
    print("="*60)
    
    # Test 1: Single point mapping
    run_command([
        "uv", "run", "python", "src/cli_mapper.py",
        "--point-name", "AHU-2_SAT",
        "--verbose"
    ], "Single Point Mapping (AHU-2_SAT)")
    
    # Test 2: Single point mapping with JSON output
    run_command([
        "uv", "run", "python", "src/cli_mapper.py", 
        "--point-name", "VAV-101_ZT",
        "--format", "json"
    ], "Single Point JSON Output")
    
    # Test 3: Batch processing with evaluation
    run_command([
        "uv", "run", "python", "src/cli_mapper.py",
        "--input-file", "data/sample_bas_mappings.csv",
        "--evaluate",
        "--output-file", "results/test_mapping_results.csv"
    ], "Batch Processing with Evaluation")
    
    # Test 4: Pattern analysis
    run_command([
        "uv", "run", "python", "src/cli_mapper.py",
        "--analyze", "data/sample_bas_mappings.csv"
    ], "Pattern Analysis")
    
    # Test 5: Different confidence threshold
    run_command([
        "uv", "run", "python", "src/cli_mapper.py",
        "--point-name", "UNKNOWN_POINT_123",
        "--confidence", "0.3",
        "--verbose"
    ], "Low Confidence Mapping")
    
    # Test 6: Multiple test points
    test_points = [
        "CH-1_POWER",
        "PUMP-2_STATUS", 
        "FCU-301_SPEED",
        "OAT",
        "LIGHT-1_CMD"
    ]
    
    for point in test_points:
        run_command([
            "uv", "run", "python", "src/cli_mapper.py",
            "--point-name", point,
            "--format", "table"
        ], f"Test Point: {point}")
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()