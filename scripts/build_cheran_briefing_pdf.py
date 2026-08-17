#!/usr/bin/env python3
"""Build Cheran first-share briefing PDF (readable, figure-backed, no invented numbers)."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "manuscript" / "figures"
# Output lands in the gitignored private/ tree: this briefing names a collaborator
# and assigns roles, so it is emailed as an attachment rather than published on the
# public repo. The generator stays tracked; only the rendered document is private.
OUT = ROOT / "private" / "correspondence" / "CHERAN_COLIDE_Briefing.pdf"

NAVY = colors.HexColor("#152A45")
TEAL = colors.HexColor("#2A5F6E")
GOLD = colors.HexColor("#B8954A")
CREAM = colors.HexColor("#F7F4EE")
INK = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#555555")
OK = colors.HexColor("#1E5C38")
WARN = colors.HexColor("#8A3B16")
LINE = colors.HexColor("#C9C3B6")
W, H = letter


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("CoverKicker", fontName="Times-Bold", fontSize=9, leading=12,
                         textColor=GOLD, tracking=1.2, spaceAfter=10))
    s.add(ParagraphStyle("CoverTitle", fontName="Times-Bold", fontSize=22, leading=26,
                         textColor=NAVY, spaceAfter=8))
    s.add(ParagraphStyle("CoverSub", fontName="Times-Italic", fontSize=12, leading=16,
                         textColor=TEAL, spaceAfter=10))
    s.add(ParagraphStyle("H1", fontName="Times-Bold", fontSize=13.5, leading=17,
                         textColor=NAVY, spaceBefore=12, spaceAfter=7))
    s.add(ParagraphStyle("H2", fontName="Times-Bold", fontSize=11.5, leading=14.5,
                         textColor=TEAL, spaceBefore=9, spaceAfter=5))
    s.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=10.4, leading=14.4,
                         textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6))
    s.add(ParagraphStyle("Item", fontName="Times-Roman", fontSize=10.4, leading=14.2,
                         textColor=INK, leftIndent=12, spaceAfter=2.5))
    s.add(ParagraphStyle("Cap", fontName="Times-Italic", fontSize=8.6, leading=11.4,
                         textColor=MUTED, alignment=TA_LEFT, spaceBefore=3, spaceAfter=10))
    s.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8.7, leading=11.4, textColor=INK))
    s.add(ParagraphStyle("CellW", fontName="Times-Bold", fontSize=8.7, leading=11.4, textColor=colors.white))
    s.add(ParagraphStyle("Quote", fontName="Times-Italic", fontSize=10.6, leading=14.6,
                         textColor=NAVY, leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=8))
    s.add(ParagraphStyle("Foot", fontName="Times-Roman", fontSize=8, leading=10.5, textColor=MUTED))
    return s


S = styles()


def P(text, name="Body"):
    return Paragraph(text, S[name])


def fig(name, width=6.5 * inch, caption=""):
    path = FIG / name
    if not path.exists():
        return [P(f"[Figure missing: {name}]", "Cap")]
    ir = ImageReader(str(path))
    iw, ih = ir.getSize()
    h = width * ih / iw
    if h > 2.85 * inch:
        h = 2.85 * inch
        width = h * iw / ih
    img = Image(str(path), width=width, height=h)
    img.hAlign = "CENTER"
    bits = [img]
    if caption:
        bits.append(P(caption, "Cap"))
    return [KeepTogether(bits)]


def grid(rows, widths, header=True):
    data = []
    for i, row in enumerate(rows):
        sty = S["CellW"] if (header and i == 0) else S["Cell"]
        data.append([Paragraph(str(c), sty) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM]),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(cmds))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, H - 0.40 * inch, W, 0.40 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Bold", 8.5)
    canvas.drawString(0.7 * inch, H - 0.25 * inch, "COLIDE  ·  Briefing for Cheranrach Mahandren")
    canvas.setFont("Times-Roman", 8)
    canvas.drawRightString(W - 0.7 * inch, H - 0.25 * inch, "First materials share  ·  17 August 2026")
    canvas.setFillColor(GOLD)
    canvas.rect(0, 0, W, 0.26 * inch, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(0.7 * inch, 0.09 * inch, "Confidential working brief  ·  github.com/titoatwork/COLIDE")
    canvas.drawRightString(W - 0.7 * inch, 0.09 * inch, f"Page {doc.page}")
    canvas.restoreState()


def story():
    x = []
    # COVER
    x += [
        P("CONFIDENTIAL — MANUSCRIPT COORDINATION", "CoverKicker"),
        P("COLIDE / CAD-CBA-v1", "CoverTitle"),
        P("A briefing for Cheranrach Mahandren, who will lead the paper writing", "CoverSub"),
        P(
            "From Ibteshamul Haque, with Prof. Por as supervisor. "
            "This is the first document I am sharing with you. It explains the project, "
            "what is frozen, which numbers you may use, how we should work together, "
            "and what not to claim. It is <b>not</b> the finished paper."
        ),
        HRFlowable(width="100%", thickness=1.4, color=GOLD, spaceAfter=10),
        P(
            "<b>Please read this PDF first</b> (~15–20 minutes). Then pull "
            "<font face='Courier'>github.com/titoatwork/COLIDE</font> (<b>master</b>) and open "
            "<font face='Courier'>docs/CHERAN_MANUSCRIPT_HANDOFF.md</font>. "
            "Do <b>not</b> use <font face='Courier'>CAD_CBA_v1_MANUSCRIPT.pdf</font> dated 22 July — it is stale."
        ),
        P("Working relationship", "H2"),
        grid(
            [
                ["Role", "Responsibility"],
                ["You (Cheran)", "Lead structure, narrative, venue style, related work, unifying the draft into a paper."],
                ["Ibteshamul", "Experiments, CUDA/DICC, artifact lookup, number checks, claim hygiene. 24–48 h fact-check."],
                ["Prof. Por", "Supervisor. High-level updates only — not day-to-day writing."],
            ],
            [1.6 * inch, 5.15 * inch],
        ),
        Spacer(1, 8),
        P(
            "Pre-manuscript evidence is <b>closed</b>. The remaining work is writing — not a new training campaign, "
            "and not inventing a post-fix server Block-3 rebench."
        ),
        P("1. What the project is", "H1"),
        P(
            "<b>COLIDE</b> is a multi-objective IoT intrusion-detection systems project. "
            "The method package is <b>CAD-CBA-v1</b>: a CNN–BiLSTM–Attention student, "
            "trained with focal loss and ensemble knowledge distillation from tree teachers "
            "(RF + XGBoost + LightGBM), under a sealed BoT-IoT protocol. "
            "Alongside detection we measure efficiency (latency, energy, memory) and "
            "custom CUDA kernels for the CNN–BiLSTM blocks only (<b>Option A</b> — not a full-model CUDA clone)."
        ),
        P(
            "The scientific story is <b>honest accuracy–efficiency</b>, not “we beat every baseline on F1.” "
            "Protocol LightGBM still leads pure validation F1. Trees remain strong on tabular flows. "
            "Our contribution is a frozen, protocol-fair package plus operation-matched CUDA and "
            "multi-GPU measurement — with explicit non-claims."
        ),
        P("Working title (locked for now)", "H2"),
        P(
            "CAD-CBA: A Class-Aware Distilled CNN–BiLSTM for Multi-Objective IoT Intrusion Detection "
            "with Operation-Matched CUDA Acceleration",
            "Quote",
        ),
        P(
            "<b>Target venue:</b> <i>Future Generation Computer Systems</i> (Elsevier), to be confirmed with "
            "Prof. Por. If that holds, the template is the Elsevier <font face='Courier'>elsarticle</font> "
            "LaTeX class or the Elsevier Word template — worth deciding early, because it shapes how we "
            "should set up the shared draft (§7.4). Authors and affiliations are for Prof. Por to finalise."
        ),
        P("2. Current state (17 August 2026)", "H1"),
        grid(
            [
                ["Layer", "State"],
                ["Pre-manuscript evidence pack", "CLOSED — numbers, claims, tables, figures, handoff frozen."],
                ["Champion model", "Frozen. md5 80a90f7cc210276300eaa90173a5a385. Do not retrain."],
                ["ToN-IoT leakage issue", "Closed. New leakage-safe run is the only active ToN result."],
                ["CUDA Block-3 correctness (laptop)", "Closed locally: race/align fix, sanitizers, real-weight parity."],
                ["CUDA Block-3 latency on DICC servers", "Historical pre-fix only (Option B). No post-fix campaign."],
                ["GitHub public repo", "Polished. Stale files kept and labeled, not deleted."],
                ["Manuscript Markdown", "Complete draft skeleton with synced numbers — see inventory below."],
                ["July 22 manuscript PDF", "STALE. Please ignore for writing."],
                ["Camera-ready paper", "Not started. That is your phase."],
            ],
            [2.55 * inch, 4.2 * inch],
        ),
        Spacer(1, 8),
        P("2.1 What already exists in the draft", "H2"),
        P(
            "So you know what you are inheriting: this is <b>not</b> a blank page. The working Markdown draft "
            "is roughly <b>5,150 words</b> across a complete skeleton — Sections 1–9 (Introduction, Related work, "
            "Method, Protocol, Results, Discussion, Threats to validity, Reproducibility, Conclusion) with "
            "<b>39 subsections</b>, plus four appendices. Every results subsection is already populated with "
            "numbers pulled from artifacts, and there are <b>9 figures</b> rendered on disk "
            "(7 of them with vector PDF versions, so they will scale cleanly in a venue template)."
        ),
        P(
            "What it needs is what you are best placed to give it: narrative, argument, related-work depth, "
            "venue style, and a voice that reads like one author rather than a lab notebook. "
            "The scaffolding and the evidence are done."
        ),
        P("3. The paper’s questions (locked answers)", "H1"),
        P(
            "These research questions are already answered by artifacts. Your job is to write them clearly, "
            "not to reopen the experiments unless something is scientifically wrong."
        ),
        grid(
            [
                ["RQ", "Locked answer"],
                ["Detection (BoT sealed test)", "Macro-F1 0.9780 ± 0.0033 (n=5). Near protocol RF; does not beat protocol LGBM."],
                ["Minority (Theft)", "Yes on sealed test: Theft F1 mean 1.0; min-class mean 0.9292."],
                ["Local efficiency (RTX 3050)", "Exploratory energy 0.920–0.943 mJ/flow; PT@256 ~24.2–25.7 µs; VRAM 322 MiB."],
                ["Cross-GPU / multi-day", "DICC tables exist. B3 server timing is pre-fix historical; PT wins B3 wall-clock."],
                ["Explainability", "Dispatch 16.60 µs p99 only. Not a free-form LLM-XAI paper."],
                ["Second dataset (ToN)", "Leakage-safe CNN 0.8075 vs RF 0.9626. Clean 0.9526 path is INVALID."],
            ],
            [2.15 * inch, 4.6 * inch],
        ),
        PageBreak(),
        P("4. Locked numbers you may put in the paper", "H1"),
        P("4.1 Principal detection — BoT-IoT", "H2"),
        grid(
            [
                ["Item", "Use this"],
                ["Principal test macro-F1", "0.9780 ± 0.0033 (sealed multi-seed, n=5)"],
                ["Champion", "model/best_model_botiot_twostage.pth"],
                ["MD5", "80a90f7cc210276300eaa90173a5a385"],
                ["Protocol LGBM val (pure F1 leader)", "0.9818 — we do not claim to beat this"],
                ["Protocol RF val", "0.9778"],
                ["Published RF 0.9864", "Different pipeline — dual-bar only, not our protocol"],
                ["Historical 0.9790", "Development/legacy only — never as principal"],
            ],
            [2.5 * inch, 4.25 * inch],
        ),
        P("4.2 ToN-IoT — secondary, leakage-safe", "H2"),
        grid(
            [
                ["Item", "Use this"],
                ["Protocol", "toniot_leakage_safe_v1 (13 features, split first, no SMOTE, no KD)"],
                ["CNN test macro-F1", "0.8075"],
                ["RF test macro-F1", "0.9626 (same split)"],
                ["Split", "Seed 42, 60/20/20 stratified random — not official temporal/host split"],
                ["Weak class", "mitm: F1 ≈ 0.111, precision ≈ 0.059, recall ≈ 0.91"],
                ["Never use as valid", "Clean CNN 0.9526 / RF 0.9851 / “+15.4%” (label leakage)"],
            ],
            [2.5 * inch, 4.25 * inch],
        ),
        P("4.3 Systems / CUDA — wording is as important as the number", "H2"),
        grid(
            [
                ["Item", "How to write it"],
                ["Local B3 parity", "Closed. GPU vs PyTorch full sequence max |Δ| ≈ 3.43×10⁻⁶."],
                ["Local sanitizers", "racecheck / synccheck / initcheck / memcheck: 0 errors (RTX 3050)."],
                ["DICC B3 latency", "Historical pre-fix only. Example: V100S ~513 µs CUDA vs ~363 µs PT."],
                ["Custom CUDA vs full V3", "Forbidden as a model-level speedup. Option A = Blocks 1–4 only."],
                ["Framework tables", "Separate operator table vs full-model table. No cross-table speedups."],
                ["~25,899 flows/s", "Bulk batched throughput (batch 128). Not a streaming SLA."],
                ["Energy 0.920–0.943 mJ/flow", "Exploratory GPU-board energy (WP6b ranges)."],
                ["16.60 µs p99", "Alert-construction / queue dispatch only."],
                ["Native TensorRT equivalence", "Not checked — do not claim numerical parity."],
            ],
            [2.35 * inch, 4.4 * inch],
        ),
        P(
            "If a sentence would be stronger than this table, stop and ask me before it goes into the draft. "
            "I would rather weaken a claim than reopen an invalid number."
        ),
        P("5. Figures (from locked artifacts)", "H1"),
        P(
            "These are the current figures on disk. Captions below are honest. "
            "You may redraw for venue style; do not change the underlying numbers."
        ),
        P("5.1 Architecture (schematic)", "H2"),
    ]
    x += fig(
        "fig_architecture.png",
        caption="Figure 1. CAD-CBA-v1 schematic (Option A CUDA covers fused Blocks 1–4 only; "
        "attention / LayerNorm / residual / classifier remain in PyTorch). Source: docs/manuscript/figures/fig_architecture.png.",
    )
    x += [
        P("5.2 Detection dual bars (protocol honesty)", "H2"),
    ]
    x += fig(
        "fig_detection_dual_bars.png",
        caption="Figure 2. Dual-bar detection view: sealed CAD-CBA test versus protocol and published baselines. "
        "Do not collapse published RF 0.9864 into our protocol. Source: fig_detection_dual_bars.png.",
    )
    x += [PageBreak(), P("5.3 Sealed-test confusion (BoT, seed 42 representative)", "H2")]
    x += fig(
        "fig_confusion_matrix_b14_seed42.png",
        caption="Figure 3. Sealed BoT test confusion, seed 42 representative "
        "(seed-42 F1 ≈ 0.9787; multi-seed mean 0.9780±0.0033). Source: fig_confusion_matrix_b14_seed42.png.",
    )
    x += [P("5.4 Ablation ladder (package credit, not attention-alone)", "H2")]
    x += fig(
        "fig_ablation_ladder.png",
        caption="Figure 4. Ablation ladder. Attention+CE alone is a negative result versus CNN–BiLSTM; "
        "credit the frozen package. Source: fig_ablation_ladder.png.",
    )
    x += [P("5.5 Corrected ToN per-class (CNN)", "H2")]
    x += fig(
        "fig_toniot_corrected_cnn_per_class.png",
        caption="Figure 5. ToN leakage-safe CNN per-class test. Macro-F1 0.8075 vs RF 0.9626. "
        "mitm is weak (F1 ≈ 0.11). Do not replace this with the invalid “clean” 0.9526 table. "
        "Source: fig_toniot_corrected_cnn_per_class.png.",
    )
    x += [P("5.6 Local systems ranges (exploratory energy / latency)", "H2")]
    x += fig(
        "fig_wp6b_systems_ranges.png",
        caption="Figure 6. Local multi-session WP6b ranges on RTX 3050 (exploratory energy; not DICC). "
        "Source: fig_wp6b_systems_ranges.png.",
    )
    x += [
        PageBreak(),
        P("6. Hard non-claims (please keep these in the paper)", "H1"),
        P("• We do not claim pure-F1 supremacy over protocol LightGBM.", "Item"),
        P("• We do not claim full custom-CUDA versus full V3 PyTorch speedup or parity.", "Item"),
        P("• We do not claim DICC Block-3 latency as post-fix, or that CUDA B3 beats cuDNN on servers.", "Item"),
        P("• We do not claim production LLM explainability.", "Item"),
        P("• We do not cite ToN “clean” 0.9526 / 0.9851 / +15.4% as valid.", "Item"),
        P("• We do not call ~25,899 f/s a live streaming SLA, or energy a certified metrology result.", "Item"),
        P("• We do not treat the 32 latent steps as observed packet chronology (pseudo-sequence).", "Item"),
        P("7. How I suggest we work", "H1"),
        P("7.1 Suggested division", "H2"),
        grid(
            [
                ["You draft", "I support"],
                ["Title/abstract variants, intro, related work, discussion, conclusion", "Artifact lookup, CUDA/DICC wording"],
                ["Unify MD → Word or LaTeX for the venue", "Regenerate a table/figure from JSON if needed"],
                ["Related-work citations and flow", "Confirm any new quantitative sentence against JSON"],
                ["Flag any claim that feels too strong", "Reply with the allowed wording or a weaker sentence"],
            ],
            [3.4 * inch, 3.35 * inch],
        ),
        P("7.2 Source of truth (in this order)", "H2"),
        P("1. Gate JSON under <font face='Courier'>benchmarks/results/</font> (ToN, B3 parity, framework, sanitizer).", "Item"),
        P("2. <font face='Courier'>docs/RESULTS_INDEX.md</font> and <font face='Courier'>docs/CLAIM_MAP_PREWRITE.md</font>.", "Item"),
        P("3. <font face='Courier'>docs/manuscript/TABLES_FROM_ARTIFACTS.md</font> and <font face='Courier'>docs/manuscript/figures/</font>.", "Item"),
        P("4. Working draft <font face='Courier'>docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md</font> — rewrite freely; keep numbers.", "Item"),
        P("5. This briefing — orientation only.", "Item"),
        P(
            "Older emails, HANDOFF.md, STATUS_REPORT_DRAFT, and PROF_POR_* packs are historical. "
            "If they conflict with RESULTS_INDEX, ignore them."
        ),
        P("7.3 What to open after this PDF", "H2"),
        grid(
            [
                ["Order", "File"],
                ["1", "docs/CHERAN_MANUSCRIPT_HANDOFF.md"],
                ["2", "docs/CLAIM_MAP_PREWRITE.md"],
                ["3", "docs/RESULTS_INDEX.md"],
                ["4", "docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md  (working draft)"],
                ["5", "docs/manuscript/TABLES_FROM_ARTIFACTS.md"],
                ["6", "docs/B3_SERVER_LATENCY_DECISION.md  (Option B)"],
                ["7", "docs/KNOWN_LIMITATIONS.md"],
            ],
            [1.0 * inch, 5.75 * inch],
        ),
        P("7.4 One shared draft, please — not emailed versions", "H2"),
        P(
            "This is the one process request I would push for. Once you pick the format, let us keep the "
            "manuscript in <b>a single live document</b> that we both edit — Overleaf if we go LaTeX, "
            "Google Docs if we go Word. Email should carry coordination, never the manuscript itself."
        ),
        P(
            "The reason is specific to this project. The invalid ToN result we had to withdraw, and the stale "
            "0.9790 figure that lingered for weeks, both came from numbers being re-typed by hand as they moved "
            "between documents. Two copies of a draft in two inboxes is exactly how that happens again. "
            "With one live document, tables get copied once from "
            "<font face='Courier'>TABLES_FROM_ARTIFACTS.md</font> and never re-keyed."
        ),
        P("7.5 Claim checking is a service, not a supervision", "H2"),
        P(
            "To be clear about how I would like the claim rules to feel: they exist so that you never have to "
            "guess whether a number is safe. Every figure in this paper traces to a JSON artifact, and I can "
            "check any sentence against its artifact within <b>24–48 hours</b>. There is also an automated "
            "scan in the repo (<font face='Courier'>scripts/check_stale_claims.py</font>) that flags withdrawn "
            "numbers and forbidden phrasings — I am happy to run it over any draft you send and return it "
            "annotated. Flagging something as uncertain is always the right call and never a problem."
        ),
        P("8. Access", "H1"),
        P(
            "<b>Repository (preferred):</b> https://github.com/titoatwork/COLIDE — branch <b>master</b>. "
            "Public academic repo (Academic Research license, not MIT)."
        ),
        P(
            "<b>Zip pack</b> (attached with this PDF, ~2 MB): "
            "<font face='Courier'>COLIDE_Cheran_manuscript_pack_20260817.zip</font> — the working draft, all "
            "figures, the claim and results documents, and the key result JSON. Everything you need to start "
            "is in there, so the repository is optional depth rather than a prerequisite."
        ),
        P(
            "The repository now also carries the evidence files behind the headline numbers "
            "(<font face='Courier'>benchmarks/results/sealed_test/</font> and the CUDA and baseline statistics), "
            "so if you ever want to confirm a figure yourself rather than take my word for it, the artifact is there."
        ),
        P("9. Three questions, and a suggested first step", "H1"),
        P("Whenever you have read this, it would help me to know:", "Body"),
        P("1. <b>Format</b> — Elsevier <font face='Courier'>elsarticle</font> LaTeX, or the Word template?", "Item"),
        P("2. <b>Workspace</b> — does a single shared Overleaf or Google Doc work for you (§7.4)?", "Item"),
        P("3. <b>Timeline</b> — what does your availability look like over the next few weeks?", "Item"),
        Spacer(1, 6),
        P(
            "On the writing itself, rather than starting everywhere at once, I would suggest beginning with "
            "<b>Section 1 (Introduction)</b> and <b>Section 2 (Related work)</b>. Those are the two sections "
            "where an outside reader adds the most and where I have the least to offer, and they carry the "
            "fewest hard numbers — so you can build up context on the project without having to hold the "
            "whole claim map in your head on day one. The results prose will be much easier afterwards."
        ),
        P(
            "A short call whenever it suits you would also be welcome — thirty minutes would cover "
            "far more than another round of email.",
            "Body",
        ),
        P("10. A note on tone", "H1"),
        P(
            "This project went through a serious remediation: we withdrew an invalid ToN result, "
            "fixed Block-3 kernel bugs, and refused to invent a server rebench we did not run. "
            "The paper should sound like that — confident where the artifact is strong, "
            "and explicitly modest where it is not. That honesty is part of the contribution."
        ),
        P(
            "Thank you for taking the manuscript lead. I will stay available for numbers, CUDA language, "
            "and DICC tables whenever you need a check.",
            "Quote",
        ),
        HRFlowable(width="100%", thickness=0.8, color=NAVY, spaceAfter=8),
        P(
            "Ibteshamul Haque  ·  FCSIT, Universiti Malaya  ·  COLIDE under Prof. Por  ·  17 August 2026. "
            "Authority chain: JSON → RESULTS_INDEX / CLAIM_MAP → manuscript MD → this briefing.",
            "Foot",
        ),
    ]
    return x


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.68 * inch,
        rightMargin=0.68 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.42 * inch,
        title="COLIDE briefing for Cheranrach Mahandren",
        author="Ibteshamul Haque",
        subject="First materials share — project, state, numbers, figures, working rules",
    )
    doc.build(story(), onFirstPage=header_footer, onLaterPages=header_footer)
    print("wrote", OUT, OUT.stat().st_size)
    dl = Path("/mnt/c/Users/Ibteshamul Haque/Downloads/COLIDE_Cheran_Briefing.pdf")
    try:
        dl.write_bytes(OUT.read_bytes())
        print("copied", dl, dl.stat().st_size)
    except Exception as e:
        print("Downloads copy failed:", e)


if __name__ == "__main__":
    main()
