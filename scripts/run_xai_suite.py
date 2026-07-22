#!/usr/bin/env python3
"""
WP7 / tracker J* — Explainability suite beyond dispatch (val-only; test sealed).

Evaluates the production champion (CAD-CBA-v1 detector) with:
  J2  Faithfulness via feature occlusion (Δ predicted-class prob)
  J3  Consistency of occlusion ranks across two independent val draws
  J4  Latency: dispatch vs generation (from existing llm_explainability.json)
  J5  Analyst usefulness: structured-evidence rubric (automatic proxy)
  J6  Hallucination rate on stored TinyLlama samples (feature/class contradiction)
  J7  Agreement: LLM text mentions vs top occlusion features
  J8  Baselines: occlusion importance vs attention-pool proxy vs rule template
  J9  Structured evidence → template explanation (no free-form LLM required)

Does NOT require shap/lime packages (not installed). Implements transparent
tabular occlusion + attention diagnostic + rule templates.

Never overwrites the production champion. Writes under benchmarks/results/xai/.

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_xai_suite.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention  # noqa: E402
from scripts.protocol.botiot import FEATURE_COLUMNS, load_botiot, load_config  # noqa: E402
from scripts.protocol.result_schema import git_sha, make_result_envelope  # noqa: E402

CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
CHAMPION_MD5_EXPECTED = "80a90f7cc210276300eaa90173a5a385"
LLM_JSON = PROJECT_ROOT / "benchmarks" / "results" / "llm_explainability.json"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "xai"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model(device: torch.device, dropout: float | None = None, att_drop: float | None = None):
    cfg = load_config()
    # Prefer HPO dropouts for package path when present
    hpo_path = PROJECT_ROOT / "config" / "hpo_best.yaml"
    if hpo_path.is_file():
        hpo = yaml.safe_load(hpo_path.read_text()).get("hpo", {})
        bp = hpo.get("best_params") or {}
        if dropout is None and "dropout_rate" in bp:
            cfg["model"]["dropout_rate"] = float(bp["dropout_rate"])
        if att_drop is None and "attention_dropout" in bp:
            cfg["model"]["attention_dropout"] = float(bp["attention_dropout"])
    if dropout is not None:
        cfg["model"]["dropout_rate"] = float(dropout)
    if att_drop is not None:
        cfg["model"]["attention_dropout"] = float(att_drop)

    model = CNNBiLSTMAttention(cfg).to(device)
    state = torch.load(CHAMPION, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def _predict_proba(model, X: np.ndarray, device: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch].astype(np.float32)).to(device)
        logits = model(xb)
        outs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def occlusion_importance(
    model,
    X: np.ndarray,
    device: torch.device,
    feature_names: list[str],
    n_samples: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Mean drop in predicted-class probability when feature j is set to train-mean."""
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xs = X[idx].astype(np.float32)
    base = _predict_proba(model, Xs, device)
    pred = base.argmax(axis=1)
    base_p = base[np.arange(n), pred]
    means = Xs.mean(axis=0)
    n_feat = Xs.shape[1]
    drops = np.zeros(n_feat, dtype=np.float64)
    for j in range(n_feat):
        Xo = Xs.copy()
        Xo[:, j] = means[j]
        po = _predict_proba(model, Xo, device)
        p_j = po[np.arange(n), pred]
        drops[j] = float(np.mean(base_p - p_j))
    order = np.argsort(-drops)
    ranked = [
        {
            "rank": int(r + 1),
            "feature": feature_names[int(j)],
            "index": int(j),
            "mean_prob_drop": float(drops[j]),
        }
        for r, j in enumerate(order)
    ]
    return {
        "n_samples": int(n),
        "seed": int(seed),
        "method": "mean-replace occlusion; Δ P(y_hat)",
        "per_feature_drop": {feature_names[j]: float(drops[j]) for j in range(n_feat)},
        "ranked": ranked,
        "top3": [ranked[i]["feature"] for i in range(min(3, len(ranked)))],
        "mean_top1_drop": float(drops[order[0]]) if len(order) else 0.0,
        "sum_positive_drops": float(np.clip(drops, 0, None).sum()),
    }


@torch.no_grad()
def attention_temporal_proxy(
    model,
    X: np.ndarray,
    device: torch.device,
    n_samples: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Capture multi-head attention average weights (seq→seq) on a val subsample.
    Maps poorly to raw tabular features (attention is post-CNN temporal), so this
    is a *diagnostic baseline*, not a feature-level SHAP substitute.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xs = torch.from_numpy(X[idx].astype(np.float32)).to(device)

    weights_list: list[np.ndarray] = []

    def _hook(_mod, _inp, out):
        # out = (attn_out, attn_weights) when need_weights=True
        if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
            w = out[1].detach().float().cpu().numpy()
            weights_list.append(w)

    # Force need_weights by temporarily wrapping attention call via forward patch
    orig_forward = model.forward

    def forward_with_weights(x):
        x = model.input_projection(x)
        x = x.view(x.size(0), model.reshape_channels, model.reshape_length)
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.conv2(x)
        x = model.bn2(x)
        x = model.relu(x)
        x = model.pool(x)
        x = model.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = model.bilstm1(x)
        x = model.dropout(x)
        x, _ = model.bilstm2(x)
        x = model.dropout(x)
        attn_out, attn_w = model.attention(x, x, x, need_weights=True, average_attn_weights=True)
        weights_list.append(attn_w.detach().float().cpu().numpy())
        x = model.attention_norm(x + attn_out)
        x = torch.mean(x, dim=1)
        x = model.fc1(x)
        x = model.relu(x)
        x = model.dropout(x)
        return model.fc2(x)

    model.forward = forward_with_weights  # type: ignore[method-assign]
    try:
        _ = model(Xs)
    finally:
        model.forward = orig_forward  # type: ignore[method-assign]

    if not weights_list:
        return {"ok": False, "note": "no attention weights captured"}
    W = np.concatenate(weights_list, axis=0)  # (N, L, L) or (N, H, L, L)
    if W.ndim == 4:
        W = W.mean(axis=1)
    # Mean attention received by each temporal position
    recv = W.mean(axis=(0, 1))  # (L,)
    return {
        "ok": True,
        "n_samples": int(n),
        "seq_len": int(recv.shape[0]),
        "mean_attention_received": [float(v) for v in recv],
        "entropy_mean": float(
            (-np.clip(W, 1e-12, 1) * np.log(np.clip(W, 1e-12, 1))).sum(axis=-1).mean()
        ),
        "note": (
            "Attention is over CNN–BiLSTM temporal axis, not raw features. "
            "Useful as internal diagnostic; not interchangeable with feature occlusion."
        ),
    }


def rule_template_explanation(
    x: np.ndarray,
    pred_cls: str,
    conf: float,
    feature_names: list[str],
    top_feats: list[str],
    per_feat_drop: dict[str, float],
) -> str:
    parts = [
        f"Predicted class: {pred_cls} (confidence {conf:.3f}).",
        "Top supporting features by occlusion drop:",
    ]
    for f in top_feats[:3]:
        j = feature_names.index(f)
        parts.append(f"  - {f}={float(x[j]):.4g} (mean ΔP≈{per_feat_drop.get(f, 0.0):.4f})")
    parts.append(
        "Recommended analyst action: verify flow against known baselines for "
        f"{pred_cls}; cross-check the listed features before escalation."
    )
    return "\n".join(parts)


def spearman_rank_corr(a: list[str], b: list[str]) -> float:
    """Spearman correlation on ranks for a common feature list."""
    feats = list(dict.fromkeys(a + b))
    if len(feats) < 2:
        return 1.0
    ra = {f: i for i, f in enumerate(a)}
    rb = {f: i for i, f in enumerate(b)}
    # Missing → last rank
    xa = np.array([ra.get(f, len(a)) for f in feats], dtype=np.float64)
    xb = np.array([rb.get(f, len(b)) for f in feats], dtype=np.float64)
    xa = xa - xa.mean()
    xb = xb - xb.mean()
    denom = float(np.sqrt((xa**2).sum() * (xb**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((xa * xb).sum() / denom)


def evaluate_llm_samples(
    llm_path: Path,
    feature_names: list[str],
    global_top_feats: list[str],
) -> dict[str, Any]:
    if not llm_path.is_file():
        return {"ok": False, "note": f"missing {llm_path}"}
    doc = json.loads(llm_path.read_text())
    samples = doc.get("sample_explanations") or []
    # Feature mention detection:
    # Short English words that are also feature names (min/max/mean/seq) are
    # too ambiguous alone — only count them when nearby IDS/context cues appear,
    # OR when the token is a distinctive feature id (underscore / long name).
    ambiguous = {"min", "max", "mean", "seq"}
    class_names = ["DDoS", "DoS", "Normal", "Reconnaissance", "Theft"]

    def features_mentioned_in(text: str) -> list[str]:
        found: list[str] = []
        low = text.lower()
        for f in feature_names:
            fl = f.lower()
            if fl in ambiguous:
                # require feature-like context
                if re.search(
                    rf"\b{re.escape(fl)}\b.{{0,40}}(feature|rate|connection|flow|value|field)",
                    low,
                ) or re.search(
                    rf"(feature|rate|connection|flow|field).{{0,40}}\b{re.escape(fl)}\b",
                    low,
                ):
                    found.append(f)
            else:
                if re.search(re.escape(f), text, re.I):
                    found.append(f)
        return found

    n = len(samples)
    n_class_ok = 0
    n_feat_mention = 0
    n_hallucinate = 0
    n_generic = 0
    per = []
    for s in samples:
        text = str(s.get("explanation") or "")
        cls = str(s.get("class") or "")
        conf = float(s.get("confidence") or 0.0)
        class_ok = cls.lower() in text.lower() if cls else False
        if class_ok:
            n_class_ok += 1
        mentioned = features_mentioned_in(text)
        if mentioned:
            n_feat_mention += 1
        # Hallucination proxies:
        # 1) invents a different attack class strongly
        other = [c for c in class_names if c.lower() != cls.lower() and c.lower() in text.lower()]
        # 2) numeric claims that don't match confidence string (loose)
        # 3) pure generic SOC advice with zero feature tokens
        halluc = False
        if other and not class_ok:
            halluc = True
        if not mentioned and len(text) > 40:
            # generic without evidence — mark soft hallucination of specificity
            n_generic += 1
        # Contradicts high confidence with uncertainty language is OK; inventing features not in set
        invented = re.findall(r"\b([A-Za-z_]{3,})\b", text)
        # Count unknown feature-like tokens (heuristic)
        known = set(f.lower() for f in feature_names) | {
            "dos",
            "ddos",
            "normal",
            "theft",
            "reconnaissance",
            "ids",
            "soc",
            "attack",
            "flow",
            "ip",
            "port",
            "confidence",
            "connections",
            "network",
            "security",
            "analyst",
            "traffic",
            "source",
            "destination",
            "severity",
            "rate",
            "sequence",
            "standard",
            "deviation",
            "mean",
            "min",
            "max",
            "seq",
            "srate",
            "drate",
            "state",
            "stddev",
        }
        # If text claims a feature-like identifier not in known set and looks technical
        for tok in invented:
            if tok.lower() not in known and any(ch.isupper() for ch in tok) and "_" in tok:
                halluc = True
                break
        if halluc:
            n_hallucinate += 1
        agree_top = len(set(mentioned) & set(global_top_feats[:3])) > 0
        per.append(
            {
                "flow_id": s.get("flow_id"),
                "class": cls,
                "confidence": conf,
                "class_mentioned": class_ok,
                "features_mentioned": mentioned,
                "mentions_global_top3": agree_top,
                "generation_time_ms": s.get("generation_time_ms"),
                "hallucination_flag": halluc,
                "generic_no_features": (not mentioned and len(text) > 40),
            }
        )

    return {
        "ok": True,
        "n_samples": n,
        "class_mention_rate": float(n_class_ok / n) if n else 0.0,
        "any_feature_mention_rate": float(n_feat_mention / n) if n else 0.0,
        "hallucination_flag_rate": float(n_hallucinate / n) if n else 0.0,
        "generic_no_feature_rate": float(n_generic / n) if n else 0.0,
        "top3_feature_agreement_rate": float(
            sum(1 for p in per if p["mentions_global_top3"]) / n
        )
        if n
        else 0.0,
        "llm_generation_mean_ms": doc.get("llm_generation_mean_ms"),
        "llm_generation_median_ms": doc.get("llm_generation_median_ms"),
        "dispatch_overhead_p99_us": doc.get("overhead_p99_us"),
        "dispatch_p99_us": doc.get("dispatch_p99_us"),
        "model": doc.get("model"),
        "quantization": doc.get("quantization"),
        "per_sample": per,
        "note": (
            "Hallucination flags are automatic heuristics on stored TinyLlama samples "
            "(n=6) — not a large human study. Treat as bounded evidence for J6/J7."
        ),
    }


def structured_usefulness_score(
    template: str,
    pred_cls: str,
    top_feats: list[str],
) -> dict[str, Any]:
    """Automatic proxy for J5: does structured evidence include class, conf, top feats, action?"""
    checks = {
        "has_class": pred_cls.lower() in template.lower(),
        "has_confidence": "confidence" in template.lower(),
        "has_top_feature": any(f in template for f in top_feats[:3]),
        "has_action": "analyst" in template.lower() or "verify" in template.lower(),
        "has_numeric_feature_value": bool(re.search(r"=\s*-?\d", template)),
    }
    score = float(sum(checks.values()) / len(checks))
    return {"checks": checks, "score": score}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-occlusion", type=int, default=256)
    ap.add_argument("--n-attention", type=int, default=128)
    ap.add_argument("--n-examples", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    champ_md5 = _md5(CHAMPION)
    if champ_md5 != CHAMPION_MD5_EXPECTED:
        print(
            f"ERROR: champion md5 {champ_md5} != expected {CHAMPION_MD5_EXPECTED}",
            file=sys.stderr,
        )
        return 2

    print("[xai] loading protocol val (stage_b_ft)…")
    bundle = load_botiot(stage="stage_b_ft", seed=args.seed)
    class_names = bundle.class_names
    feature_names = list(FEATURE_COLUMNS)

    print("[xai] loading champion…")
    model, cfg = _load_model(device)

    # --- Occlusion faithfulness (two seeds for consistency) ---
    print("[xai] occlusion importance (seed A)…")
    occ_a = occlusion_importance(
        model, bundle.X_val, device, feature_names, n_samples=args.n_occlusion, seed=args.seed
    )
    print("[xai] occlusion importance (seed B)…")
    occ_b = occlusion_importance(
        model,
        bundle.X_val,
        device,
        feature_names,
        n_samples=args.n_occlusion,
        seed=args.seed + 7,
    )
    rank_corr = spearman_rank_corr(
        [r["feature"] for r in occ_a["ranked"]],
        [r["feature"] for r in occ_b["ranked"]],
    )

    # --- Attention diagnostic ---
    print("[xai] attention temporal proxy…")
    attn = attention_temporal_proxy(
        model, bundle.X_val, device, n_samples=args.n_attention, seed=args.seed
    )

    # --- Example structured explanations ---
    rng = np.random.default_rng(args.seed)
    n_ex = min(args.n_examples, len(bundle.X_val))
    ex_idx = rng.choice(len(bundle.X_val), size=n_ex, replace=False)
    Xex = bundle.X_val[ex_idx]
    probs = _predict_proba(model, Xex, device)
    preds = probs.argmax(axis=1)
    examples = []
    usefulness_scores = []
    for i in range(n_ex):
        pred_cls = class_names[int(preds[i])]
        conf = float(probs[i, preds[i]])
        top = occ_a["top3"]
        tmpl = rule_template_explanation(
            Xex[i], pred_cls, conf, feature_names, top, occ_a["per_feature_drop"]
        )
        use = structured_usefulness_score(tmpl, pred_cls, top)
        usefulness_scores.append(use["score"])
        examples.append(
            {
                "val_index": int(ex_idx[i]),
                "true_class": class_names[int(bundle.y_val[ex_idx[i]])],
                "pred_class": pred_cls,
                "confidence": conf,
                "template_explanation": tmpl,
                "usefulness": use,
            }
        )

    # --- LLM sample rubric (existing) ---
    print("[xai] LLM sample rubric…")
    llm_eval = evaluate_llm_samples(LLM_JSON, feature_names, occ_a["top3"])

    # --- Faithfulness scalar: fraction of samples where top-1 occlusion feature
    #     flip (set to mean of *other class* or extreme) changes prediction more
    #     than a random feature. Use relative drop ranking integrity.
    faithfulness_proxy = {
        "metric": "occlusion_positive_mass_on_top3",
        "definition": "sum of positive mean-prob-drops of top-3 / sum of all positive drops",
        "value": float(
            sum(max(0.0, occ_a["per_feature_drop"][f]) for f in occ_a["top3"])
            / max(occ_a["sum_positive_drops"], 1e-12)
        ),
        "mean_top1_drop": occ_a["mean_top1_drop"],
        "note": (
            "Higher mass on top-3 ⇒ more concentrated feature attributions. "
            "Not a ground-truth causal faithfulness certificate."
        ),
    }

    mean_useful = float(np.mean(usefulness_scores)) if usefulness_scores else 0.0

    # --- Decision for paper claims (J10) ---
    # Criteria for keeping full "explainable" branding:
    #  - structured evidence usefulness ≥ 0.75
    #  - occlusion consistency Spearman ≥ 0.7
    #  - LLM feature-mention rate ≥ 0.5 OR we drop LLM quality claims
    llm_feat_rate = float(llm_eval.get("any_feature_mention_rate") or 0.0)
    # Bounded claim (structured evidence + occlusion) vs full LLM-XAI branding.
    # Full title-level "explainable" needs strong LLM evidence agreement.
    structured_ok = (
        mean_useful >= 0.75
        and rank_corr >= 0.7
        and faithfulness_proxy["value"] >= 0.35
    )
    llm_ok = llm_feat_rate >= 0.5 and float(llm_eval.get("top3_feature_agreement_rate") or 0) >= 0.3

    if structured_ok and llm_ok:
        decision = "INCORPORATED"
        decision_note = (
            "XAI suite supports a *bounded* explainability contribution: structured "
            "evidence templates + consistent occlusion ranks + non-trivial LLM feature "
            "agreement. Still not a full human faithfulness study; do not claim SHAP/LIME "
            "parity (packages not used)."
        )
        j10 = "KEEP_BOUNDED_XAI_CLAIM"
    elif structured_ok:
        decision = "RUN_DOCUMENTED"
        decision_note = (
            "Structured evidence + occlusion suite is solid "
            f"(usefulness={mean_useful:.3f}, rank_corr={rank_corr:.3f}, "
            f"faith_top3_mass={faithfulness_proxy['value']:.3f}), but free-form TinyLlama "
            f"samples are weak on feature agreement (mention_rate={llm_feat_rate:.3f}, "
            f"top3_agree={llm_eval.get('top3_feature_agreement_rate')}). "
            "Paper path: keep dispatch-only 16.60 µs + structured-evidence template; "
            "do NOT brand title/abstract as full LLM-explainable IDS. J10=drop full claim."
        )
        j10 = "DROP_FULL_EXPLAINABLE_CLAIM_KEEP_STRUCTURED"
    else:
        decision = "RUN_DOCUMENTED"
        decision_note = (
            "XAI suite completed. Full 'explainable IDS' title claim NOT supported: "
            f"usefulness={mean_useful:.3f}, rank_corr={rank_corr:.3f}, "
            f"faith_top3_mass={faithfulness_proxy['value']:.3f}, "
            f"llm_feature_mention={llm_feat_rate:.3f}. "
            "Paper should keep dispatch-only micro-result (16.60 µs); drop explainable branding."
        )
        j10 = "DROP_FULL_EXPLAINABLE_CLAIM"

    latency_block = {
        "dispatch_overhead_p99_us": llm_eval.get("dispatch_overhead_p99_us"),
        "dispatch_p99_us": llm_eval.get("dispatch_p99_us"),
        "llm_generation_mean_ms": llm_eval.get("llm_generation_mean_ms"),
        "llm_generation_median_ms": llm_eval.get("llm_generation_median_ms"),
        "note": "Dispatch ≪ generation; never conflate µs dispatch with ms generation.",
        "source": str(LLM_JSON.relative_to(PROJECT_ROOT)),
    }

    summary: dict[str, Any] = {
        "experiment_id": "wp7_xai_suite",
        "tracker": ["J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10"],
        "work_package": "WP7",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": args.seed,
        "allow_test": False,
        "test_sealed": True,
        "checkpoint": str(CHAMPION.relative_to(PROJECT_ROOT)),
        "champion_md5": champ_md5,
        "champion_unchanged": True,
        "feature_names": feature_names,
        "class_names": class_names,
        "occlusion_seed_a": occ_a,
        "occlusion_seed_b": {
            "top3": occ_b["top3"],
            "ranked": occ_b["ranked"],
            "mean_top1_drop": occ_b["mean_top1_drop"],
        },
        "consistency_spearman_rank": rank_corr,
        "attention_proxy": attn,
        "faithfulness_proxy": faithfulness_proxy,
        "structured_examples": examples,
        "structured_usefulness_mean": mean_useful,
        "llm_sample_eval": llm_eval,
        "latency": latency_block,
        "baselines_compared": [
            "feature_occlusion",
            "attention_temporal_proxy",
            "rule_template_structured_evidence",
            "tinyllama_freeform_samples_existing",
        ],
        "shap_lime": {
            "installed": False,
            "note": "shap/lime not in env; occlusion + attention + rules used instead (documented).",
        },
        "decision": decision,
        "j10_path": j10,
        "decision_note": decision_note,
        "wall_sec": float(time.time() - t0),
        "git_sha": git_sha(PROJECT_ROOT),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_val": int(len(bundle.y_val)),
    }

    # Per-metric tracker flips (notes for PROF_FEEDBACK_TRACKER)
    summary["tracker_status"] = {
        "J2": {
            "status": "RUN_DOCUMENTED",
            "note": f"occlusion faithfulness proxy top3_mass={faithfulness_proxy['value']:.4f}",
        },
        "J3": {
            "status": "RUN_DOCUMENTED",
            "note": f"Spearman rank consistency={rank_corr:.4f} across two val draws",
        },
        "J4": {
            "status": "DONE",
            "note": (
                f"dispatch p99 overhead {latency_block['dispatch_overhead_p99_us']} µs; "
                f"generation mean {latency_block['llm_generation_mean_ms']} ms"
            ),
        },
        "J5": {
            "status": "RUN_DOCUMENTED",
            "note": f"structured usefulness mean={mean_useful:.3f} (automatic rubric, n={n_ex})",
        },
        "J6": {
            "status": "RUN_DOCUMENTED",
            "note": (
                f"hallucination_flag_rate={llm_eval.get('hallucination_flag_rate')} "
                f"generic_no_feature_rate={llm_eval.get('generic_no_feature_rate')} n={llm_eval.get('n_samples')}"
            ),
        },
        "J7": {
            "status": "RUN_DOCUMENTED",
            "note": (
                f"llm any_feature_mention={llm_eval.get('any_feature_mention_rate')}; "
                f"top3_agreement={llm_eval.get('top3_feature_agreement_rate')}"
            ),
        },
        "J8": {
            "status": "RUN_DOCUMENTED",
            "note": "occlusion vs attention proxy vs rule template (no shap/lime package)",
        },
        "J9": {
            "status": "DONE",
            "note": "structured evidence template includes class, conf, top feats, action",
        },
        "J10": {
            "status": "DONE" if j10.startswith("DROP") else "DONE",
            "note": j10 + " — " + decision_note[:200],
        },
    }

    out_summary = OUT_DIR / "summary.json"
    out_examples = OUT_DIR / "structured_examples.json"
    out_table = OUT_DIR / "table.md"

    out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    out_examples.write_text(json.dumps(examples, indent=2) + "\n")

    md = []
    md.append("# WP7 XAI suite (BoT-IoT val)\n")
    md.append(f"- Champion md5: `{champ_md5}`\n")
    md.append(f"- Decision: **{decision}** / J10 path: **{j10}**\n")
    md.append(f"- Occlusion top-3: {', '.join(occ_a['top3'])}\n")
    md.append(f"- Rank consistency (Spearman): **{rank_corr:.4f}**\n")
    md.append(f"- Faithfulness top-3 mass: **{faithfulness_proxy['value']:.4f}**\n")
    md.append(f"- Structured usefulness mean: **{mean_useful:.3f}**\n")
    md.append(
        f"- LLM feature mention rate: **{llm_feat_rate:.3f}** "
        f"(n={llm_eval.get('n_samples')})\n"
    )
    md.append(
        f"- Dispatch p99 overhead: **{latency_block['dispatch_overhead_p99_us']} µs**; "
        f"generation mean **{latency_block['llm_generation_mean_ms']} ms**\n"
    )
    md.append("\n## Occlusion ranking\n\n| Rank | Feature | Mean ΔP |\n|------|---------|--------|\n")
    for r in occ_a["ranked"]:
        md.append(f"| {r['rank']} | `{r['feature']}` | {r['mean_prob_drop']:.6f} |\n")
    md.append(f"\n## Decision note\n\n{decision_note}\n")
    out_table.write_text("".join(md))

    print(json.dumps(
        {
            "decision": decision,
            "j10_path": j10,
            "rank_corr": rank_corr,
            "faith_top3_mass": faithfulness_proxy["value"],
            "usefulness_mean": mean_useful,
            "llm_feature_mention": llm_feat_rate,
            "top3": occ_a["top3"],
            "wall_sec": summary["wall_sec"],
            "out": str(out_summary),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
