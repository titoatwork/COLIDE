# ToN-IoT corrected leakage-safe results

- Protocol: `toniot_leakage_safe_v1`
- Features (13): `['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'src_ip_bytes', 'dst_ip_bytes', 'src_port', 'dst_port', 'missed_bytes', 'proto', 'service', 'conn_state']`
- Feature SHA-256: `838239eea277712ed719a17ea5f451eebbea368fa673a0676820741b438ecb61`
- Split seed: **42** (60/20/20 stratified, train-only preprocess)
- RF test macro-F1: **0.9626** (val 0.9626)
- CNN test macro-F1: **0.8075** (val 0.8066)
- valid: true | use_in_manuscript: true
- SMOTE: false | KD: false
