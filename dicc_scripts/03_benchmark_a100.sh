#!/bin/bash
#SBATCH --job-name=colide_a100
#SBATCH --output=benchmarks/results/dicc_a100_%j.log
#SBATCH --error=benchmarks/results/dicc_a100_%j.err
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

echo "=== COLIDE A100 Benchmark ==="
echo "Node: $(hostname)"
nvidia-smi
date

cd /scr/$USER/colide
conda activate colide
export PYTHONPATH=.

# HW+job-id tag: every output file from this run gets this suffix so that
# submitting this same script again (e.g. the required second-day drift-check
# run) or a concurrent V100 job on shared storage can never silently
# overwrite this run's results -- see HANDOFF.md "Session 4" for why this
# matters (this exact overwrite class of bug already bit the RTX 3050 local
# runs once).
HW=a100
RUNTAG="${HW}_${SLURM_JOB_ID}"

echo ""
echo "=== CUDA Kernel Benchmarks (A100), single-run sanity check ==="
./inference/kernels/a100/fused_block1
echo "---"
./inference/kernels/a100/fused_block2
echo "---"
./inference/kernels/a100/fused_block3
echo "---"
./inference/kernels/a100/fused_block3_fp16
echo "---"
./inference/kernels/a100/fused_block4

echo ""
echo "=== CUDA Kernel Statistical Benchmark (A100), n=20 trials -- THIS is the ==="
echo "=== number that should go in the paper, not the single-run numbers above ==="
python scripts/benchmark_cuda_kernels_stats.py --kernels-dir inference/kernels/a100 --suffix "" --tag "$RUNTAG"

echo ""
echo "=== Python Benchmarks ==="
python scripts/benchmark_pipeline.py
for f in benchmarks/results/pipeline_benchmark*.json; do
    [ -f "$f" ] && cp -v "$f" "benchmarks/results/$(basename "${f%.json}")_${RUNTAG}.json"
done
echo "---"
python scripts/benchmark_batch.py
echo "---"
python scripts/benchmark_ort.py
echo "---"
python scripts/benchmark_energy.py
[ -f benchmarks/results/energy_efficiency.json ] && cp -v benchmarks/results/energy_efficiency.json "benchmarks/results/energy_efficiency_${RUNTAG}.json"
echo "---"
python scripts/benchmark_stats.py
[ -f benchmarks/results/statistical_confidence.json ] && cp -v benchmarks/results/statistical_confidence.json "benchmarks/results/statistical_confidence_${RUNTAG}.json"

echo ""
echo "=== A100 Benchmark Complete (tag: ${RUNTAG}) ==="
echo "=== Pull back every benchmarks/results/*${RUNTAG}* file, plus this log, before running the day-2 resubmission ==="
date
