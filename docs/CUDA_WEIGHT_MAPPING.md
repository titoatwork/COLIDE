# CUDA Weight Mapping (Block 3)

Short note on how custom BiLSTM kernels map weights relative to PyTorch LSTM.
A full end-to-end weight audit against production checkpoints is **ongoing**.

## Gate order

PyTorch `nn.LSTM` packs gates as **i, f, g, o** (input, forget, cell, output).
Custom CUDA kernels use the same order:

| Slice in `weight_ih` / `weight_hh` | Gate |
|------------------------------------|------|
| `[0 * H : 1 * H]`                  | i    |
| `[1 * H : 2 * H]`                  | f    |
| `[2 * H : 3 * H]`                  | g    |
| `[3 * H : 4 * H]`                  | o    |

Biases `bias_ih` and `bias_hh` follow the same layout and are added per gate.

## Matrix layout (optimized kernels)

- Host / PyTorch-style `W_hh`: shape `(4H, H)`, row-major, gate-major.
- Optimized FP32 path transposes to `W_hh_t` with shape `(H, 4H)` for coalesced
  hidden-to-hidden accumulation across threads.
- Optimized FP16 path repacks transposed `W_hh` into half2 pairs
  `(i,f)` and `(g,o)` per output unit for `__hfma2`.

## Bidirectional alignment

- Forward: process `pos = t`, store at `pos`.
- Reverse: process `pos = seq_len - 1 - t`, store at original `pos` (not
  recurrence index `t`), matching PyTorch BiLSTM sequence alignment.
- Layer-1 combine concatenates aligned `fw | rev` channels at each time index
  before layer 2.

## Block-3 output contract (current harness)

| Aspect | Current harness | Preferred for full V3 |
|--------|-----------------|------------------------|
| Shape  | `(128,)` last-timestep extract | full sequence `[seq, 128]` (or batched) |
| Index  | `seq_len - 1` on **aligned** sequences | full sequence → attention + LN + mean pool |
| Semantics | Matches PyTorch `output[:, -1, :]` **after** reverse position alignment | Required for attention path |

Honest status:

- **Full sequence** is what V3 attention consumes and is the long-term parity target.
- **Current extract is last-timestep** (`extract_last_timestep_kernel` at
  `seq_len-1`) for parity with the existing CUDA / PyTorch Block-3 harness.
- Do not claim last-timestep equals reverse recurrence final state `h_n` without
  qualification: after alignment, reverse at index `seq_len-1` is the reverse
  hidden at the last **sequence** position (PyTorch `output[:, -1, rev:]`), not
  the reverse direction’s state after processing position 0.

## Real-weight parity harness

Run the champion real-weight Block-3 gate (does **not** retrain or modify
weights):

```bash
PYTHONPATH=. python scripts/parity_block3_cuda_pt.py
# dry-run / help:
PYTHONPATH=. python scripts/parity_block3_cuda_pt.py --dry-run
PYTHONPATH=. python scripts/parity_block3_cuda_pt.py --help
```

| Item | Path |
|------|------|
| Harness | `scripts/parity_block3_cuda_pt.py` |
| Gate JSON | `benchmarks/results/block3_parity_gate.json` |
| Champion | `config/paths.py` → `CHAMPION_PATH` / `model/best_model_botiot_twostage.pth` |
| MD5 | `80a90f7cc210276300eaa90173a5a385` |

What it checks:

1. **PyTorch BiLSTM** (matching ops: `bilstm1` → `bilstm2`, eval mode) on a
   fixed-seed batch with **champion** weights.
2. **CUDA-contract CPU reference** (mirrors `cpu_lstm_forward` in
   `fused_block3.cu`: gate order i,f,g,o; reverse store at original `pos`;
   last-timestep extract at `seq_len-1` on the aligned sequence).
3. Optional **CUDA binary self-check** (`inference/kernels/fused_block3`,
   `fused_block3_fp16`) when binaries exist — those use *synthetic* RNG
   weights inside the binary (not champion inject).

Gate metadata:

- `kernel_status: "code_fixed_awaiting_rebench"` until post_fix rebench.
- `use_in_manuscript: false` until rebench + full gate green.
- `valid: false` while real-weight GPU inject path is still pending.

## Open audit items

- Wire real champion weights into CUDA binaries (or a harness launcher) for
  direct GPU real-weight parity (beyond PT vs CPU-ref + synthetic self-check).
- Confirm `W_hh` transpose / half2 repack bit-exactness for production dumps
  on GPU path.
- Sequence-level GPU comparison vs PyTorch BiLSTM stack after rebench.
- `compute-sanitizer` racecheck/synccheck on fixed kernels (DICC or local).
