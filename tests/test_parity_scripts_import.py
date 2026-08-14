"""Smoke imports for Phase 3 / Phase 7 parity harnesses."""

from __future__ import annotations


def test_export_block3_weights_import():
    import scripts.export_block3_weights as mod

    assert hasattr(mod, "main")
    assert callable(mod.main)
    assert hasattr(mod, "CUDA_WEIGHT_FILES")
    assert hasattr(mod, "export")
    assert "w_ih1_f" in mod.CUDA_WEIGHT_FILES
    assert "w_ih2_r" in mod.CUDA_WEIGHT_FILES


def test_parity_block3_cuda_pt_import():
    import scripts.parity_block3_cuda_pt as mod

    assert hasattr(mod, "main")
    assert callable(mod.main)
    assert hasattr(mod, "cpu_block3_pipeline")
    assert hasattr(mod, "extract_bilstm_weights")
    assert hasattr(mod, "pytorch_block3")
    assert hasattr(mod, "pytorch_v3_suffix")
    assert hasattr(mod, "run_parity")
    assert mod.FP32_TOL_PT_CPU > 0
    assert mod.FP32_TOL_GPU_PT > 0


def test_parity_framework_backends_import():
    import scripts.parity_framework_backends as mod

    assert hasattr(mod, "main")
    assert callable(mod.main)
    assert hasattr(mod, "run_parity")
    assert hasattr(mod, "find_onnx")
    assert mod.FP32_TOL_EAGER > 0
    assert mod.FP32_TOL_ORT > 0
