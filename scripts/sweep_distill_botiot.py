#!/usr/bin/env python3
"""
COLIDE – BoT-IoT Distillation Sweep (Alpha + Temperature)
Runs multiple distillation configurations sequentially.
"""

import subprocess
import sys

EXPERIMENTS = [
    (0.3, 1.0, "a0.3_T1.0"),
    (0.7, 1.0, "a0.7_T1.0"),
    (0.9, 1.0, "a0.9_T1.0"),
    (0.5, 3.0, "a0.5_T3.0"),
    (0.7, 5.0, "a0.7_T5.0"),
]

def run_exp(alpha, temp, suffix):
    print(f"\n{'='*70}")
    print(f"SWEEP: alpha={alpha}, temperature={temp}, suffix={suffix}")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    subprocess.run([
        "python", "-u", "scripts/train_distill.py",
        "--dataset", "botiot",
        "--alpha", str(alpha),
        "--temperature", str(temp),
        "--epochs", "50",
        "--patience", "10",
        "--suffix", suffix,
    ], check=True)

if __name__ == "__main__":
    for alpha, temp, suffix in EXPERIMENTS:
        run_exp(alpha, temp, suffix)
    print("\n✅ Sweep complete.")
    