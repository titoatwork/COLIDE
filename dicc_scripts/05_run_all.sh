#!/bin/bash
# =============================================================================
# COLIDE DICC Master Script - Submit all jobs
# =============================================================================
echo "=== Submitting COLIDE DICC Benchmark Jobs ==="

# First run setup (only needed once)
# bash 01_setup.sh

echo "Submitting V100 benchmark..."
sbatch 02_benchmark_v100.sh
echo ""

echo "Submitting A100 benchmark..."
sbatch 03_benchmark_a100.sh
echo ""

echo "Submitting Nsight profiling..."
sbatch 04_nsight_profile.sh
echo ""

echo "All jobs submitted. Check status with: squeue -u $USER"
