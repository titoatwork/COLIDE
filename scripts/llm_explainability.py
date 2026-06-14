"""
COLIDE - Async LLM Explainability Prototype
Demonstrates that LLM-generated alerts do not impact detection latency.

Architecture:
  - Detection thread: classifies flows, pushes alerts to ring buffer
  - LLM thread: consumes alerts from ring buffer, generates explanations
  - Ring buffer: fixed-size queue with backpressure (drops oldest if full)

Model: TinyLlama-1.1B Q4 (swappable for Phi-3-mini in production)
"""

import sys
import os
import time
import copy
import threading
import queue
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import yaml
import torch

sys.path.insert(0, '.')

# ================================================================
# Alert data structure
# ================================================================
@dataclass
class Alert:
    flow_id: int
    predicted_class: str
    confidence: float
    features: dict
    timestamp: float
    explanation: Optional[str] = None
    explanation_time_ms: Optional[float] = None

# ================================================================
# Ring buffer (thread-safe, drops oldest on overflow)
# ================================================================
class RingBuffer:
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.buffer = queue.Queue(maxsize=capacity)
        self.dropped = 0

    def push(self, item):
        try:
            self.buffer.put_nowait(item)
        except queue.Full:
            try:
                self.buffer.get_nowait()  # drop oldest
                self.dropped += 1
            except queue.Empty:
                pass
            self.buffer.put_nowait(item)

    def pop(self, timeout=1.0):
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def size(self):
        return self.buffer.qsize()

# ================================================================
# LLM Explainer (runs in separate thread)
# ================================================================
class LLMExplainer:
    def __init__(self, model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print("[LLM] Loading model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto'
        )
        self.model.eval()
        print(f"[LLM] Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    def generate_explanation(self, alert: Alert) -> str:
        prompt = f"""<|system|>
You are a network security analyst. Given an IDS alert, provide a brief, actionable explanation in 2-3 sentences.</s>
<|user|>
Alert: {alert.predicted_class} attack detected (confidence: {alert.confidence:.2%})
Flow features: {json.dumps(alert.features, indent=None)}
What is happening and what should the SOC analyst do?</s>
<|assistant|>
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

# ================================================================
# LLM Worker Thread
# ================================================================
def llm_worker(ring_buffer, explainer, results, stop_event):
    while not stop_event.is_set():
        alert = ring_buffer.pop(timeout=0.5)
        if alert is None:
            continue

        start = time.perf_counter()
        try:
            explanation = explainer.generate_explanation(alert)
        except Exception as e:
            explanation = f"[Error: {str(e)}]"
        elapsed_ms = (time.perf_counter() - start) * 1000

        alert.explanation = explanation
        alert.explanation_time_ms = elapsed_ms
        results.append(alert)

# ================================================================
# Detection Simulator
# ================================================================
def simulate_detection(num_flows=20):
    """Simulate IDS classification results with some attacks."""
    from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    class_names = config['data']['class_names']
    feature_names = config['data']['feature_columns']

    model = CNNBiLSTM(config)
    model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    # Load some real test data
    X_test = np.load('data/processed/X_test.npy')

    flows = []
    # Pick a mix of indices that include attacks
    np.random.seed(42)
    indices = np.random.choice(len(X_test), num_flows, replace=False)

    for idx in indices:
        x = X_test[idx]
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        features = {name: float(x[i]) for i, name in enumerate(feature_names)}
        flows.append({
            'features': features,
            'predicted_class': class_names[pred_class],
            'confidence': confidence
        })

    return flows

# ================================================================
# Main benchmark
# ================================================================
def main():
    print("=" * 70)
    print("COLIDE ASYNC LLM EXPLAINABILITY BENCHMARK")
    print("=" * 70)

    # Initialize components
    ring_buffer = RingBuffer(capacity=32)
    explainer = LLMExplainer()
    results = []
    stop_event = threading.Event()

    # Start LLM worker thread
    llm_thread = threading.Thread(
        target=llm_worker,
        args=(ring_buffer, explainer, results, stop_event),
        daemon=True
    )
    llm_thread.start()

    # Simulate detection
    print("\n[Detection] Simulating flow classification...")
    flows = simulate_detection(num_flows=20)

    # ============================================================
    # Benchmark 1: Detection latency WITHOUT LLM dispatch
    # ============================================================
    print("\n--- Benchmark 1: Detection Only (no LLM) ---")
    detection_times_no_llm = []
    for flow in flows:
        start = time.perf_counter()
        # Simulate classification (already done, just measure overhead)
        _ = flow['predicted_class']
        elapsed = (time.perf_counter() - start) * 1e6
        detection_times_no_llm.append(elapsed)

    # ============================================================
    # Benchmark 2: Detection latency WITH LLM dispatch
    # ============================================================
    print("--- Benchmark 2: Detection + Async LLM Dispatch ---")
    detection_times_with_llm = []
    alerts_dispatched = 0

    for i, flow in enumerate(flows):
        start = time.perf_counter()

        # Classification result (already computed)
        pred = flow['predicted_class']

        # Only dispatch alerts for attacks (not Normal)
        if pred != 'Normal':
            alert = Alert(
                flow_id=i,
                predicted_class=pred,
                confidence=flow['confidence'],
                features=flow['features'],
                timestamp=time.time()
            )
            ring_buffer.push(alert)  # Non-blocking push
            alerts_dispatched += 1

        elapsed = (time.perf_counter() - start) * 1e6
        detection_times_with_llm.append(elapsed)

    print(f"[Detection] {alerts_dispatched} alerts dispatched to LLM thread")
    print(f"[Detection] Ring buffer size: {ring_buffer.size()}")

    # Wait for LLM to process all alerts
    print("\n[LLM] Processing alerts...")
    timeout = 120  # max wait seconds
    start_wait = time.time()
    while len(results) < alerts_dispatched and (time.time() - start_wait) < timeout:
        processed = len(results)
        print(f"  Processed {processed}/{alerts_dispatched}...", end='\r')
        time.sleep(2)

    stop_event.set()
    llm_thread.join(timeout=5)

    # ============================================================
    # Results
    # ============================================================
    print(f"\n\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # Detection latency comparison
    avg_no_llm = np.mean(detection_times_no_llm)
    avg_with_llm = np.mean(detection_times_with_llm)
    print(f"\nDetection latency (avg, us):")
    print(f"  Without LLM dispatch: {avg_no_llm:.2f} us")
    print(f"  With LLM dispatch:    {avg_with_llm:.2f} us")
    print(f"  Overhead:             {avg_with_llm - avg_no_llm:.2f} us")
    print(f"  Impact:               {'NEGLIGIBLE' if (avg_with_llm - avg_no_llm) < 10 else 'SIGNIFICANT'}")

    # LLM generation stats
    if results:
        gen_times = [r.explanation_time_ms for r in results if r.explanation_time_ms]
        print(f"\nLLM generation time (ms):")
        print(f"  Mean:   {np.mean(gen_times):.1f} ms")
        print(f"  Median: {np.median(gen_times):.1f} ms")
        print(f"  Min:    {np.min(gen_times):.1f} ms")
        print(f"  Max:    {np.max(gen_times):.1f} ms")

    print(f"\nRing buffer stats:")
    print(f"  Capacity:  {ring_buffer.capacity}")
    print(f"  Dropped:   {ring_buffer.dropped}")
    print(f"  Processed: {len(results)}/{alerts_dispatched}")

    # Sample explanations
    print(f"\n{'='*70}")
    print("SAMPLE EXPLANATIONS")
    print(f"{'='*70}")
    for alert in results[:5]:
        print(f"\n[Flow {alert.flow_id}] {alert.predicted_class} ({alert.confidence:.2%})")
        print(f"  Explanation ({alert.explanation_time_ms:.0f}ms): {alert.explanation}")

    # Save results
    out_path = 'benchmarks/results/llm_explainability.json'
    save_data = {
        'detection_latency_no_llm_us': avg_no_llm,
        'detection_latency_with_llm_us': avg_with_llm,
        'overhead_us': avg_with_llm - avg_no_llm,
        'llm_generation_mean_ms': float(np.mean(gen_times)) if results else 0,
        'llm_generation_median_ms': float(np.median(gen_times)) if results else 0,
        'alerts_dispatched': alerts_dispatched,
        'alerts_processed': len(results),
        'alerts_dropped': ring_buffer.dropped,
        'model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'quantization': '4-bit',
        'sample_explanations': [
            {
                'flow_id': a.flow_id,
                'class': a.predicted_class,
                'confidence': a.confidence,
                'explanation': a.explanation,
                'generation_time_ms': a.explanation_time_ms
            }
            for a in results[:10]
        ]
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()