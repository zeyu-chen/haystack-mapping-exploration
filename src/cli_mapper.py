#!/usr/bin/env python3
"""
Command line interface for BAS point name mapping.

This tool provides various ways to map BAS point names to Haystack tags:
- Single point mapping
- Batch processing from CSV files
- Interactive mode for exploration
- Performance evaluation and reporting
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mapping_utils import (
    HaystackTagMapper,
    MappingEvaluator,
    PointNameAnalyzer,
)


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description="Map BAS point names to Haystack tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s -p "AHU-1_SAT"
  %(prog)s -i data.csv -o results.csv
  %(prog)s -i data.csv --evaluate
  %(prog)s --interactive
  %(prog)s --analyze data.csv
        """,
    )
    parser.add_argument(
        "--point-name",
        "-p",
        type=str,
        help="Single point name to map",
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        help="CSV file with point names to map",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        help="Output file for results (JSON or CSV)",
    )
    parser.add_argument(
        "--evaluate",
        "-e",
        action="store_true",
        help="Evaluate mapping performance (requires actual tags)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.7,
        help="Confidence threshold for mapping (default: 0.7)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mapping session",
    )
    parser.add_argument(
        "--analyze",
        "-a",
        type=str,
        help="Analyze patterns in CSV file without mapping",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "table"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing large files (default: 100)",
    )

    args = parser.parse_args()

    # Initialize mapper
    try:
        mapper = HaystackTagMapper()
        if args.verbose:
            print("Initialized HaystackTagMapper")
    except Exception as e:
        print(f"Error initializing mapper: {e}")
        sys.exit(1)

    # Handle different operation modes
    if args.point_name:
        handle_single_point(mapper, args)
    elif args.input_file:
        handle_batch_processing(mapper, args)
    elif args.analyze:
        handle_analysis(args)
    elif args.interactive:
        handle_interactive_mode(mapper, args)
    else:
        print("Error: Please provide an operation mode")
        parser.print_help()
        sys.exit(1)


def handle_single_point(mapper: HaystackTagMapper, args: argparse.Namespace) -> None:
    """Handle single point mapping."""
    try:
        result = mapper.map_to_haystack(args.point_name, args.confidence)

        if args.format == "json":
            print(json.dumps(result, indent=2))
        elif args.format == "csv":
            df = pd.DataFrame([result])
            print(df.to_csv(index=False))
        else:
            print_mapping_result(result, args.verbose)

    except Exception as e:
        print(f"Error mapping point '{args.point_name}': {e}")
        sys.exit(1)


def handle_batch_processing(
    mapper: HaystackTagMapper, args: argparse.Namespace
) -> None:
    """Handle batch processing from CSV file."""
    try:
        if args.verbose:
            print(f"Loading data from {args.input_file}")

        df = pd.read_csv(args.input_file)

        # Validate required columns
        if "original_point_name" not in df.columns:
            print("Error: Input file must have 'original_point_name' column")
            sys.exit(1)

        if args.verbose:
            print(f"Processing {len(df)} point names...")

        # Process in batches for large files
        results = []
        for i in range(0, len(df), args.batch_size):
            batch = df.iloc[i : i + args.batch_size]

            if args.verbose:
                print(
                    f"Processing batch {i // args.batch_size + 1}/{(len(df) - 1) // args.batch_size + 1}"
                )

            batch_results = []
            for _, row in batch.iterrows():
                point_name = row["original_point_name"]
                result = mapper.map_to_haystack(point_name, args.confidence)
                batch_results.append(result)

            results.extend(batch_results)

        results_df = pd.DataFrame(results)

        # Evaluate if requested and possible
        if args.evaluate and "standard_tags" in df.columns:
            evaluator = MappingEvaluator()
            predictions = results_df["mapped_tags"].tolist()
            actuals = df["standard_tags"].tolist()

            evaluation = evaluator.evaluate_mappings(predictions, actuals)
            print_evaluation_results(evaluation)

        # Save or display results
        if args.output_file:
            save_results(results_df, args.output_file, args.format)
            print(f"Results saved to {args.output_file}")
        else:
            display_results(results_df, args.format)

    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


def handle_analysis(args: argparse.Namespace) -> None:
    """Handle pattern analysis without mapping."""
    try:
        df = pd.read_csv(args.analyze)

        if "original_point_name" not in df.columns:
            print("Error: Input file must have 'original_point_name' column")
            sys.exit(1)

        analyzer = PointNameAnalyzer()

        print("\n" + "=" * 60)
        print("POINT NAME PATTERN ANALYSIS")
        print("=" * 60)

        # Analyze patterns
        equipment_types = []
        concepts_list = []

        for point_name in df["original_point_name"]:
            parsed = analyzer.parse_point_name(point_name)
            equipment_types.append(parsed["equipment_type"])
            concepts_list.append(parsed["concepts"])

        # Equipment type distribution
        print("\nEquipment Types:")
        equipment_counts = pd.Series(equipment_types).value_counts()
        print(equipment_counts)

        # Concept frequency
        print("\nConcept Frequency:")
        all_concepts = [concept for concepts in concepts_list for concept in concepts]
        concept_counts = pd.Series(all_concepts).value_counts()
        print(concept_counts.head(10))

        # Name statistics
        print("\nName Statistics:")
        name_lengths = df["original_point_name"].str.len()
        print(f"Average length: {name_lengths.mean():.1f}")
        print(f"Min length: {name_lengths.min()}")
        print(f"Max length: {name_lengths.max()}")

        # Common patterns
        print("\nCommon Patterns:")
        patterns = df["original_point_name"].str.contains("_").sum()
        print(f"Names with underscores: {patterns}/{len(df)}")

        patterns = df["original_point_name"].str.contains("-").sum()
        print(f"Names with hyphens: {patterns}/{len(df)}")

        print("=" * 60)

    except Exception as e:
        print(f"Error analyzing file: {e}")
        sys.exit(1)


def handle_interactive_mode(
    mapper: HaystackTagMapper, args: argparse.Namespace
) -> None:
    """Handle interactive mapping mode."""
    print("\nInteractive BAS Point Mapping Mode")
    print("Type 'quit' or 'exit' to end session")
    print("Type 'help' for available commands")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nEnter point name (or command): ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "help":
                print_interactive_help()
            elif user_input.lower().startswith("confidence "):
                try:
                    new_confidence = float(user_input.split()[1])
                    args.confidence = new_confidence
                    print(f"Confidence threshold set to {new_confidence}")
                except (IndexError, ValueError):
                    print("Usage: confidence <value>")
            elif user_input:
                result = mapper.map_to_haystack(user_input, args.confidence)
                print_mapping_result(result, verbose=True)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def print_interactive_help() -> None:
    """Print help for interactive mode."""
    print("\nAvailable commands:")
    print("  help                 - Show this help")
    print("  confidence <value>   - Set confidence threshold")
    print("  quit/exit/q         - Exit interactive mode")
    print("  <point_name>        - Map a point name")


def print_mapping_result(result: dict, verbose: bool = False) -> None:
    """Print mapping result for a single point."""
    print(f"\nOriginal Name: {result['original_name']}")
    print(f"Equipment Type: {result['equipment_type']}")
    print(f"Mapped Tags: {result['mapped_tags']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Method: {result['method']}")

    if verbose and "debug_info" in result:
        print(f"Debug Info: {result['debug_info']}")


def print_evaluation_results(evaluation: dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Exact Match Accuracy: {evaluation['exact_match_accuracy']:.3f}")
    print(f"Average Similarity: {evaluation['average_similarity']:.3f}")
    print(f"Median Similarity: {evaluation['median_similarity']:.3f}")
    print(f"Min Similarity: {evaluation['min_similarity']:.3f}")
    print(f"Max Similarity: {evaluation['max_similarity']:.3f}")
    print("=" * 50)


def save_results(results_df: pd.DataFrame, output_file: str, format_type: str) -> None:
    """Save results to file."""
    output_path = Path(output_file)

    if format_type == "json" or output_path.suffix == ".json":
        results_df.to_json(output_file, orient="records", indent=2)
    else:
        results_df.to_csv(output_file, index=False)


def display_results(results_df: pd.DataFrame, format_type: str) -> None:
    """Display results to console."""
    print("\nMapping Results:")
    print("-" * 50)

    if format_type == "json":
        print(results_df.to_json(orient="records", indent=2))
    elif format_type == "csv":
        print(results_df.to_csv(index=False))
    else:
        # Table format with truncation for readability
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 50)
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
