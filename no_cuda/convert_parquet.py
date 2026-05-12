"""
Convert parquet files to CSV for use with the FedGATSage preprocessing pipeline.
Handles large files by reading/writing in chunks to avoid OOM kills.

Usage:
    # Convert a single file
    python convert_parquet.py --input ../data/CIC-ToN-IoT-V2.parquet

    # Convert with custom output path
    python convert_parquet.py --input ../data/CIC-ToN-IoT-V2.parquet --output ../data/my_dataset.csv

    # Convert both Kaggle datasets at once
    python convert_parquet.py --all

    # Adjust chunk size for very large files (rows per chunk)
    python convert_parquet.py --input ../data/CIC-ToN-IoT-V2.parquet --chunk_size 50000
"""

import os
import sys
import argparse
import pandas as pd
import pyarrow.parquet as pq


def convert_parquet_to_csv(input_path: str, output_path: str = None,
                           chunk_size: int = 100000) -> str:
    """
    Convert a parquet file to CSV using chunked I/O to handle large files.
    
    Args:
        input_path: Path to .parquet file
        output_path: Path to output .csv (default: same name as input)
        chunk_size: Number of rows to process per chunk
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.csv'

    # Get total row count and schema info without loading full file
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow
    
    input_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Input:  {input_path} ({input_mb:.1f} MB, {total_rows:,} rows)")
    print(f"Output: {output_path}")
    print(f"Processing in chunks of {chunk_size:,} rows...")

    first_chunk = True
    total_written = 0

    for chunk_num, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        df_chunk = batch.to_pandas()
        rows_in_chunk = len(df_chunk)

        # First chunk: write with header; subsequent chunks: append without header
        df_chunk.to_csv(
            output_path,
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk
        )

        total_written += rows_in_chunk
        pct = (total_written / total_rows) * 100
        print(f"  Chunk {chunk_num + 1}: {total_written:,}/{total_rows:,} rows ({pct:.0f}%)")

        first_chunk = False

    output_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! {total_written:,} rows → {output_path} ({output_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet datasets to CSV for FedGATSage"
    )
    parser.add_argument(
        '--input', '-i', type=str,
        help='Path to input .parquet file'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Path to output .csv file (default: same name as input)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Convert both CIC-ToN-IoT-V2.parquet and NF-ToN-IoT.parquet from ../data/'
    )
    parser.add_argument(
        '--chunk_size', type=int, default=100000,
        help='Rows per chunk (default: 100000, lower if still OOM)'
    )

    args = parser.parse_args()

    if args.all:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        datasets = ['CIC-ToN-IoT-V2.parquet', 'NF-ToN-IoT.parquet']
        for name in datasets:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                try:
                    convert_parquet_to_csv(path, chunk_size=args.chunk_size)
                    print()
                except Exception as e:
                    print(f"Error converting {name}: {e}", file=sys.stderr)
                    print()
            else:
                print(f"Skipping {name} (not found at {path})\n")
    elif args.input:
        try:
            convert_parquet_to_csv(args.input, args.output, args.chunk_size)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
