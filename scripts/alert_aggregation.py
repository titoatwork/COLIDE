"""
COLIDE — Alert Aggregation Module
Batches structurally identical IDS alerts before LLM dispatch
to prevent ring-buffer overflow during high-volume attacks (e.g., DDoS).

Design:
  - Alerts are grouped by (attack_type, source_ip) over a configurable window
  - At the end of each window, a single batched prompt is dispatched to the LLM
  - This reduces LLM calls from potentially thousands/sec to one per window per group
  - The detection pipeline is never blocked (async dispatch preserved)

Usage:
  aggregator = AlertAggregator(window_seconds=10)
  aggregator.add_alert(attack_type="DDoS", source_ip="192.168.1.5", flow_data={...})
  # Periodically call flush() or let the background thread handle it
  batched_alerts = aggregator.flush()
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


@dataclass
class AlertGroup:
    """A group of structurally identical alerts."""
    attack_type: str
    source_ip: str
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    sample_flows: list = field(default_factory=list)
    max_samples: int = 5  # Keep up to 5 representative flows

    def add(self, flow_data: dict):
        now = time.time()
        if self.count == 0:
            self.first_seen = now
        self.last_seen = now
        self.count += 1
        if len(self.sample_flows) < self.max_samples:
            self.sample_flows.append(flow_data)


class AlertAggregator:
    """
    Aggregates IDS alerts over a time window before dispatching to LLM.
    
    During a DDoS attack generating 25,000+ flows/sec, this reduces
    LLM dispatch from thousands of individual calls to one batched
    prompt per (attack_type, source_ip) pair every `window_seconds`.
    
    Args:
        window_seconds: Aggregation window duration (default: 10s)
        on_flush: Callback function receiving list of AlertGroups
        max_groups: Maximum number of tracked groups (memory safety)
    """

    def __init__(self, window_seconds: float = 10.0,
                 on_flush: Optional[Callable] = None,
                 max_groups: int = 1000):
        self.window_seconds = window_seconds
        self.on_flush = on_flush
        self.max_groups = max_groups
        self._groups: Dict[tuple, AlertGroup] = defaultdict(AlertGroup)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def add_alert(self, attack_type: str, source_ip: str,
                  flow_data: Optional[dict] = None):
        """Add an alert to the aggregation buffer. Thread-safe."""
        key = (attack_type, source_ip)
        with self._lock:
            if key not in self._groups:
                if len(self._groups) >= self.max_groups:
                    return  # Drop to prevent memory exhaustion
                self._groups[key] = AlertGroup(
                    attack_type=attack_type, source_ip=source_ip)
            self._groups[key].add(flow_data or {})

    def flush(self) -> List[AlertGroup]:
        """Flush all aggregated alerts and return them as a list."""
        with self._lock:
            groups = list(self._groups.values())
            self._groups.clear()
        return [g for g in groups if g.count > 0]

    def format_prompt(self, groups: List[AlertGroup]) -> str:
        """Format aggregated alerts into a single LLM prompt."""
        if not groups:
            return ""

        lines = ["The following attack activity was detected in the "
                 f"last {self.window_seconds} seconds:\n"]

        for g in sorted(groups, key=lambda x: x.count, reverse=True):
            duration = g.last_seen - g.first_seen
            lines.append(
                f"- {g.attack_type} from {g.source_ip}: "
                f"{g.count} flows over {duration:.1f}s"
            )
            if g.sample_flows:
                lines.append(f"  Sample flow: {g.sample_flows[0]}")

        lines.append(
            "\nProvide a concise security analysis of this activity, "
            "including likely attack intent, severity assessment, "
            "and recommended mitigation actions."
        )
        return "\n".join(lines)

    def start_background(self):
        """Start a background thread that flushes periodically."""
        self._running = True
        self._thread = threading.Thread(target=self._background_loop,
                                         daemon=True)
        self._thread.start()

    def stop_background(self):
        """Stop the background flush thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.window_seconds + 1)

    def _background_loop(self):
        """Periodically flush and dispatch to LLM callback."""
        while self._running:
            time.sleep(self.window_seconds)
            groups = self.flush()
            if groups and self.on_flush:
                prompt = self.format_prompt(groups)
                self.on_flush(prompt, groups)


# ============================================================
# Demo / test
# ============================================================
if __name__ == "__main__":
    def mock_llm_dispatch(prompt, groups):
        total = sum(g.count for g in groups)
        print(f"\n[LLM DISPATCH] {len(groups)} groups, "
              f"{total} total alerts aggregated")
        print(f"Prompt length: {len(prompt)} chars")
        print(prompt[:500])
        print("...")

    # Simulate DDoS: 25,000 alerts in 10 seconds
    agg = AlertAggregator(window_seconds=5, on_flush=mock_llm_dispatch)
    agg.start_background()

    print("Simulating DDoS attack: 25,000 flows over 5 seconds...")
    start = time.time()
    for i in range(25000):
        agg.add_alert(
            attack_type="DDoS",
            source_ip=f"192.168.1.{i % 10}",
            flow_data={"dst_port": 80, "bytes": 1024, "flow_id": i}
        )
    elapsed = time.time() - start
    print(f"Ingested 25,000 alerts in {elapsed*1000:.1f} ms")
    print(f"Rate: {25000/elapsed:,.0f} alerts/sec")
    print(f"Waiting for flush window...")

    time.sleep(6)  # Wait for flush
    agg.stop_background()
    print("\nDone. Without aggregation: 25,000 LLM calls.")
    print("With aggregation: 10 LLM calls (one per source IP).")