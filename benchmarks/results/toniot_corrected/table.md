# ToN-IoT corrected leakage-safe results

- Protocol: `toniot_leakage_safe_v1`
- Features (13): `['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'src_ip_bytes', 'dst_ip_bytes', 'src_port', 'dst_port', 'missed_bytes', 'proto', 'service', 'conn_state']`
- Feature SHA-256: `838239eea277712ed719a17ea5f451eebbea368fa673a0676820741b438ecb61`
- Split seed: **42** (60/20/20 stratified, train-only preprocess)
- RF test macro-F1: **0.9626** (val 0.9626)
- CNN test macro-F1: **0.8075** (val 0.8066)
- valid: true | use_in_manuscript: true | source_dirty: false
- git_sha: `fd08f36925762978c2ca73b63c477e95a9fbc86f`
- checkpoint_sha256: `1974ac3ed299d8c5f785e733607e5ed73ff197348a5d04eb231be5e3fea68dcd`
- SMOTE: false | KD: false
- numeric_missing: fixed_zero_imputation | scaler: MinMaxScaler train-only

## RF per-class F1 (test)

- `backdoor`: F1=0.9999  P=1.0000  R=0.9997  support=3742
- `ddos`: F1=0.9812  P=0.9842  R=0.9782  support=3999
- `dos`: F1=0.9848  P=0.9968  R=0.9732  support=3799
- `injection`: F1=0.9719  P=0.9883  R=0.9559  support=3993
- `mitm`: F1=0.7490  P=0.6547  R=0.8750  support=208
- `normal`: F1=0.9941  P=0.9938  R=0.9943  support=8408
- `password`: F1=0.9942  P=0.9990  R=0.9894  support=3972
- `ransomware`: F1=0.9992  P=0.9983  R=1.0000  support=2947
- `scanning`: F1=0.9893  P=0.9820  R=0.9968  support=4000
- `xss`: F1=0.9631  P=0.9410  R=0.9861  support=3027

## CNN per-class F1 (test)

- `backdoor`: F1=0.9880  P=0.9770  R=0.9992  support=3742
- `ddos`: F1=0.7783  P=0.9744  R=0.6479  support=3999
- `dos`: F1=0.9532  P=0.9991  R=0.9113  support=3799
- `injection`: F1=0.7577  P=0.7287  R=0.7891  support=3993
- `mitm`: F1=0.1114  P=0.0593  R=0.9087  support=208
- `normal`: F1=0.9228  P=0.9877  R=0.8660  support=8408
- `password`: F1=0.7666  P=0.8125  R=0.7256  support=3972
- `ransomware`: F1=0.9783  P=0.9709  R=0.9857  support=2947
- `scanning`: F1=0.9524  P=0.9848  R=0.9220  support=4000
- `xss`: F1=0.8666  P=0.8737  R=0.8596  support=3027

## Rare-class note

CNN min per-class F1 is **0.1114** (classes with F1 < 0.2: `mitm`). Low F1 with high recall and low precision indicates rare-class overprediction under class-weighted CE; do not retune against this already-observed test set. RF remains substantially stronger and more balanced on this leakage-safe split. The `mitm` class is among the weak set; treat rare-class CNN scores as exploratory and prefer RF for balanced per-class behavior on this protocol.
