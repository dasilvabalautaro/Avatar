from pathlib import Path

import pytest

from avatar_face.infrastructure.feasibility.onnx_quantizer import OnnxStaticInt8Quantizer


def test_quantizer_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ONNX"):
        OnnxStaticInt8Quantizer().quantize(tmp_path / "missing.onnx", tmp_path, overwrite=False)
