# WP8 ToN-IoT final method (CAD-CBA-v1 mapped)

- Val macro-F1: **0.8080** (KD selected)
- Test macro-F1: **0.8110**
- RF same-split test: **0.9393**
- Ensemble teacher val: **0.9618**
- Features: **13** (historical clean 26)
- Decision: **RUN_DOCUMENTED**

CAD-CBA-v1 on ToN processed (13-feat): val macro-F1=0.8080, test macro-F1=0.8110; same-split RF test=0.9393. KD val best=0.8080; FT did not improve — kept KD. Not weight-transfer from BoT. Feature set differs from historical 26-feat clean (0.9526). Neural lags RF on this 13-feat ToN protocol — honest multi-dataset gap for systems paper.
