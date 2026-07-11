#!/usr/bin/env python3
"""Write a tiny executable fake kernel that prints parseable latency lines."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from textwrap import dedent


TEMPLATES = {
    "fused_block1": dedent(
        """\
        #!/usr/bin/env bash
        echo "✅ FP32 validation PASSED"
        echo "⏱️  Fused kernel (FP32) time: {latency} µs"
        exit {exit_code}
        """
    ),
    "fused_block2": dedent(
        """\
        #!/usr/bin/env bash
        echo "✅ FP32 validation PASSED"
        echo "⏱️  Fused kernel (FP32) time: {latency} µs"
        exit {exit_code}
        """
    ),
    "fused_block3": dedent(
        """\
        #!/usr/bin/env bash
        echo "✅ FP32 validation PASSED"
        echo "✅ CUDA Graph validation PASSED"
        echo "Without CUDA Graphs: {latency_no} µs"
        echo "With CUDA Graphs: {latency} µs"
        exit {exit_code}
        """
    ),
    "fused_block3_fp16": dedent(
        """\
        #!/usr/bin/env bash
        echo "✅ FP16 half2 validation PASSED"
        echo "Block3 FP16 half2: {latency} µs"
        exit {exit_code}
        """
    ),
    "fused_block3_naive": dedent(
        """\
        #!/usr/bin/env bash
        echo "✅ FP32 validation PASSED"
        echo "⏱️  Block3 (BiLSTM) time: {latency} µs"
        exit {exit_code}
        """
    ),
    "fused_block4": dedent(
        """\
        #!/usr/bin/env bash
        echo "✅ FP32 validation PASSED"
        echo "⏱️  Block4 (Dense) time: {latency} µs"
        exit {exit_code}
        """
    ),
    "fused_pipeline": dedent(
        """\
        #!/usr/bin/env bash
        echo "✅ FP32 validation PASSED"
        echo "Blocks 1+2+4 chained (no BiLSTM): {latency} µs"
        exit {exit_code}
        """
    ),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", required=True, choices=sorted(TEMPLATES))
    p.add_argument("--latency", type=float, default=100.0)
    p.add_argument("--latency-no", type=float, default=110.0)
    p.add_argument("--exit-code", type=int, default=0)
    p.add_argument("--fail-validation", action="store_true")
    p.add_argument("--omit-metric", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = TEMPLATES[args.name].format(
        latency=args.latency,
        latency_no=args.latency_no,
        exit_code=args.exit_code,
    )
    if args.fail_validation:
        body = body.replace("PASSED", "FAILED")
    if args.omit_metric:
        # Drop lines containing time/latency patterns
        body = "\n".join(
            ln for ln in body.splitlines()
            if "time" not in ln.lower()
            and "Graphs" not in ln
            and "half2" not in ln
            and "chained" not in ln
        ) + "\nexit {0}\n".format(args.exit_code)

    path = out_dir / args.name
    path.write_text(body)
    os.chmod(path, 0o755)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
