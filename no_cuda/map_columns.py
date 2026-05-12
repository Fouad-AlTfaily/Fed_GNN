"""
Map Kaggle parquet column names to the format expected by the FedGATSage codebase.

The original paper used CICFlowMeter CSV exports, but the Kaggle parquet files
use different naming conventions:
  - NF-ToN-IoT: NFStream format (no IPs, different column names)
  - CIC-ToN-IoT: CICFlowMeter-derived but may have variations

Usage:
    # Map NF-ToN-IoT (generates synthetic IPs since dataset has none)
    python map_columns.py --input ../data/NF-ToN-IoT.parquet --dataset nf_ton_iot

    # Map CIC-ToN-IoT
    python map_columns.py --input ../data/CIC-ToN-IoT-V2.parquet --dataset cic_ton_iot

    # Map both
    python map_columns.py --all
"""

import os
import sys
import argparse
import hashlib
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# ============================================================
# Column mappings for each dataset
# ============================================================

# NF-ToN-IoT: NFStream → CICFlowMeter (what the code expects)
NF_TON_IOT_MAP = {
    'L4_SRC_PORT': 'Src Port',
    'L4_DST_PORT': 'Dst Port',
    'PROTOCOL': 'Protocol',
    'L7_PROTO': 'L7 Protocol',
    'IN_BYTES': 'TotLen Fwd Pkts',
    'OUT_BYTES': 'TotLen Bwd Pkts',
    'IN_PKTS': 'Tot Fwd Pkts',
    'OUT_PKTS': 'Tot Bwd Pkts',
    'TCP_FLAGS': 'TCP Flags Raw',  # will be split into SYN/RST/ACK counts
    'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
    'Label': 'Label',
    'Attack': 'Attack',
}

# CIC-ToN-IoT: Known variations to normalize
CIC_TON_IOT_MAP = {
    # These are common CICFlowMeter variations; add more as discovered
    'src_ip': 'Src IP',
    'dst_ip': 'Dst IP',
    'source_ip': 'Src IP',
    'destination_ip': 'Dst IP',
    'src port': 'Src Port',
    'dst port': 'Dst Port',
    'source port': 'Src Port',
    'destination port': 'Dst Port',
    'protocol': 'Protocol',
    'flow duration': 'Flow Duration',
    'total fwd packets': 'Tot Fwd Pkts',
    'total bwd packets': 'Tot Bwd Pkts',
    'total length of fwd packets': 'TotLen Fwd Pkts',
    'total length of bwd packets': 'TotLen Bwd Pkts',
    'fwd packet length max': 'Fwd Pkt Len Max',
    'bwd packet length max': 'Bwd Pkt Len Max',
    'flow iat mean': 'Flow IAT Mean',
    'flow iat std': 'Flow IAT Std',
    'flow packets/s': 'Flow Pkts/s',
    'syn flag count': 'SYN Flag Cnt',
    'rst flag count': 'RST Flag Cnt',
    'ack flag count': 'ACK Flag Cnt',
    'label': 'Label',
    'attack': 'Attack',
    'Label': 'Label',
    'Attack': 'Attack',
}


def _make_synthetic_ip(row, prefix=10, role='src') -> str:
    """
    Generate a synthetic IP for flows that lack real IPs.
    Uses port + protocol to create consistent pseudo-identifiers,
    preserving graph structure for community detection.
    """
    port = row.get('Src Port', row.get('L4_SRC_PORT', 0))
    if role == 'dst':
        port = row.get('Dst Port', row.get('L4_DST_PORT', 0))
    proto = row.get('Protocol', row.get('PROTOCOL', 0))
    
    # Hash port+protocol to stable octet values
    key = f"{port}_{proto}_{role}"
    h = hashlib.md5(key.encode()).hexdigest()
    octet2 = int(h[0:2], 16) % 256
    octet3 = int(h[2:4], 16) % 256
    octet4 = int(h[4:6], 16) % 256
    
    return f"{prefix}.{octet2}.{octet3}.{octet4}"


def split_tcp_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract SYN, RST, ACK flag counts from a single TCP_FLAGS integer column.
    TCP flags bitfield: FIN=0x01, SYN=0x02, RST=0x04, PSH=0x08, ACK=0x10, URG=0x20
    """
    flag_col = None
    for candidate in ['TCP Flags Raw', 'TCP_FLAGS', 'tcp_flags']:
        if candidate in df.columns:
            flag_col = candidate
            break
    
    if flag_col is None:
        return df
    
    flags = df[flag_col].fillna(0).astype(int)
    df['SYN Flag Cnt'] = ((flags & 0x02) != 0).astype(int)
    df['RST Flag Cnt'] = ((flags & 0x04) != 0).astype(int)
    df['ACK Flag Cnt'] = ((flags & 0x10) != 0).astype(int)
    df.drop(columns=[flag_col], inplace=True)
    
    print(f"  Extracted SYN/RST/ACK flags from TCP_FLAGS bitfield")
    return df


def compute_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features that the code expects but the dataset lacks."""
    
    # Flow IAT Mean / Std — use small defaults if missing
    if 'Flow IAT Mean' not in df.columns:
        if 'Flow Duration' in df.columns and 'Tot Fwd Pkts' in df.columns:
            # Rough estimate: duration / number of packets
            pkts = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
            df['Flow IAT Mean'] = df['Flow Duration'] / (pkts + 1)
        else:
            df['Flow IAT Mean'] = 1000.0
        print(f"  Computed missing: Flow IAT Mean")
    
    if 'Flow IAT Std' not in df.columns:
        df['Flow IAT Std'] = df['Flow IAT Mean'] * 0.5
        print(f"  Computed missing: Flow IAT Std")
    
    # Flow Pkts/s
    if 'Flow Pkts/s' not in df.columns:
        if 'Flow Duration' in df.columns:
            pkts = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
            duration_sec = df['Flow Duration'] / 1_000_000  # microseconds to seconds
            df['Flow Pkts/s'] = pkts / (duration_sec + 0.001)
        else:
            df['Flow Pkts/s'] = 10.0
        print(f"  Computed missing: Flow Pkts/s")
    
    return df


def map_nf_ton_iot(df: pd.DataFrame) -> pd.DataFrame:
    """Map NF-ToN-IoT (NFStream) columns to expected names."""
    print("  Applying NF-ToN-IoT column mapping...")
    
    # 1. Rename columns using the mapping
    df = df.rename(columns=NF_TON_IOT_MAP)
    
    # 2. Generate synthetic IPs (NF-ToN-IoT has no real IPs)
    if 'Src IP' not in df.columns:
        df['Src IP'] = df.apply(lambda r: _make_synthetic_ip(r, prefix=10, role='src'), axis=1)
        print(f"  Generated synthetic Src IP from port+protocol")
    if 'Dst IP' not in df.columns:
        df['Dst IP'] = df.apply(lambda r: _make_synthetic_ip(r, prefix=192, role='dst'), axis=1)
        print(f"  Generated synthetic Dst IP from port+protocol")
    
    # 3. Split TCP flags
    df = split_tcp_flags(df)
    
    # 4. Compute missing derived features
    df = compute_missing_features(df)
    
    return df


def map_cic_ton_iot(df: pd.DataFrame) -> pd.DataFrame:
    """Map CIC-ToN-IoT columns to expected names."""
    print("  Applying CIC-ToN-IoT column mapping...")
    
    # Normalize column names (case-insensitive matching)
    rename = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in CIC_TON_IOT_MAP:
            target = CIC_TON_IOT_MAP[col_lower]
            if col != target:
                rename[col] = target
    
    if rename:
        print(f"  Renamed {len(rename)} columns: {rename}")
        df = df.rename(columns=rename)
    else:
        print(f"  No column renaming needed")
    
    # Check if IPs exist
    if 'Src IP' not in df.columns:
        print(f"  WARNING: No 'Src IP' column found — generating synthetic IPs")
        df['Src IP'] = df.apply(lambda r: _make_synthetic_ip(r, prefix=10, role='src'), axis=1)
    if 'Dst IP' not in df.columns:
        print(f"  WARNING: No 'Dst IP' column found — generating synthetic IPs")
        df['Dst IP'] = df.apply(lambda r: _make_synthetic_ip(r, prefix=192, role='dst'), axis=1)
    
    # Compute missing features if needed
    df = compute_missing_features(df)
    
    return df


def map_columns(input_path: str, output_path: str = None, 
                dataset: str = 'cic_ton_iot') -> str:
    """
    Read a parquet file, map columns, and save as CSV.
    Handles large files with chunked processing.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{base}_mapped.csv")
    
    input_mb = os.path.getsize(input_path) / (1024 * 1024)
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    print(f"Mapping: {os.path.basename(input_path)} ({input_mb:.1f} MB, {total_rows:,} rows)")
    print(f"Dataset: {dataset}")
    print(f"Output:  {output_path}")
    
    mapper = map_nf_ton_iot if dataset == 'nf_ton_iot' else map_cic_ton_iot
    
    first_chunk = True
    total_written = 0
    chunk_size = 100000
    
    for chunk_num, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        df_chunk = batch.to_pandas()
        
        # Apply column mapping
        df_chunk = mapper(df_chunk)
        
        # Write CSV
        df_chunk.to_csv(
            output_path,
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk
        )
        
        total_written += len(df_chunk)
        pct = (total_written / total_rows) * 100
        print(f"  Chunk {chunk_num + 1}: {total_written:,}/{total_rows:,} rows ({pct:.0f}%)")
        first_chunk = False
    
    output_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! → {output_path} ({output_mb:.1f} MB)\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Map Kaggle parquet columns to FedGATSage expected format"
    )
    parser.add_argument('--input', '-i', type=str, help='Input .parquet file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output .csv file')
    parser.add_argument('--dataset', type=str, choices=['nf_ton_iot', 'cic_ton_iot'],
                        default='cic_ton_iot', help='Dataset type')
    parser.add_argument('--all', action='store_true',
                        help='Map both NF-ToN-IoT and CIC-ToN-IoT from ../data/')
    
    args = parser.parse_args()
    
    if args.all:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        configs = [
            ('NF-ToN-IoT.parquet', 'nf_ton_iot'),
            ('CIC-ToN-IoT-V2.parquet', 'cic_ton_iot'),
        ]
        for filename, ds_type in configs:
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                try:
                    map_columns(path, dataset=ds_type)
                except Exception as e:
                    print(f"Error: {e}\n")
            else:
                print(f"Skipping {filename} (not found)\n")
    elif args.input:
        try:
            map_columns(args.input, args.output, args.dataset)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
