"""
OBD-II Electrical Dataset Generation Script
Executes the synthetic data generator and saves training and testing datasets.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


from src.generator.synthetic_generator import (
    OBDSyntheticDataGenerator,
    SimulationPattern,
    PATTERN_NAMES,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OBD-II electrical telemetry dataset."
    )
    parser.add_argument(
        "--sessions-per-pattern",
        type=int,
        default=50,
        help="Number of simulated vehicle sessions per diagnostic pattern (default: 50).",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=60.0,
        help="Duration in seconds for each vehicle trip session (default: 60s).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=5.0,
        help="OBD-II sampling rate in Hz (default: 5.0 Hz).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save generated CSV files (default: data/raw).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚗 SMART OBD-II ELECTRICAL DIAGNOSTIC SYSTEM - DATA GENERATOR ⚡")
    print("=" * 70)
    print(f"📊 Sampling Rate: {args.sampling_rate} Hz (dt = {1.0/args.sampling_rate:.2f}s)")
    print(f"⏱️ Session Duration: {args.duration_sec} seconds")
    print(f"🔄 Sessions per Class: {args.sessions_per_pattern} (Total: {args.sessions_per_pattern * 4} sessions)")
    print(f"📁 Output Directory: {out_path}")
    print("-" * 70)

    generator = OBDSyntheticDataGenerator(
        sampling_rate_hz=args.sampling_rate,
        random_seed=args.seed,
    )

    print("🚀 Generating full synthetic dataset...")
    df = generator.generate_full_dataset(
        sessions_per_pattern=args.sessions_per_pattern,
        session_duration_sec=args.duration_sec,
    )

    raw_csv_path = out_path / "obd_electrical_telemetry_raw.csv"
    df.to_csv(raw_csv_path, index=False)

    print(f"✅ Full dataset successfully generated and saved to:")
    print(f"   -> {raw_csv_path}")
    print(f"   -> Total Rows: {len(df):,} records")
    print(f"   -> Total Columns: {len(df.columns)}")
    print("-" * 70)

    # Print summary statistics per class
    print("\n📈 Dataset Distribution by Class:")
    for pat in SimulationPattern:
        subset = df[df["label"] == pat.value]
        sessions_cnt = subset["session_id"].nunique()
        records_cnt = len(subset)
        min_v = subset["battery_voltage_v"].min()
        max_v = subset["battery_voltage_v"].max()
        mean_v = subset["battery_voltage_v"].mean()
        
        ar_name = PATTERN_NAMES[pat]["ar"]
        en_name = PATTERN_NAMES[pat]["en"]
        
        print(f"\n🏷️  Class {pat.value}: {en_name}")
        print(f"    الاسم بالعربية: {ar_name}")
        print(f"    الجلسات (Sessions): {sessions_cnt} | السجلات (Rows): {records_cnt:,}")
        print(f"    نطاق الجهد (Voltage): Min = {min_v:.2f}V | Avg = {mean_v:.2f}V | Max = {max_v:.2f}V")

    # Generate a small sample test file for quick testing & API verification
    sample_csv_path = out_path / "sample_test_telemetry.csv"
    sample_df = df.groupby("label", as_index=False).head(50).reset_index(drop=True)
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"\n📦 Generated lightweight sample file for rapid testing: {sample_csv_path}")
    print("=" * 70)



if __name__ == "__main__":
    main()
