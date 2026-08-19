from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from avatar_face.domain.attributes import ATTRIBUTE_CARDINALITIES  # noqa: E402
from avatar_face.infrastructure.training.student_unet import (  # noqa: E402
    StudentUNet,
    StudentUNetConfig,
)


def test_default_budget_is_within_adr_0010() -> None:
    model = StudentUNet(StudentUNetConfig())
    parameters = model.parameter_count()
    assert 20_000_000 <= parameters <= 70_000_000


def test_forward_contract_small_config() -> None:
    config = StudentUNetConfig(
        image_size=64,
        base_channels=32,
        channel_multipliers=(1, 2),
        attention_resolutions=(32,),
        condition_dim=64,
        attribute_embedding_dim=8,
    )
    model = StudentUNet(config)
    batch = 2
    x = torch.randn(batch, 3, 64, 64)
    ratio = torch.rand(batch)
    attributes = torch.stack(
        [
            torch.tensor([c - 1 for c in ATTRIBUTE_CARDINALITIES]),
            torch.zeros(len(ATTRIBUTE_CARDINALITIES), dtype=torch.long),
        ]
    )
    output = model(x, ratio, attributes)
    assert output.shape == (batch, 3, 64, 64)
    assert torch.isfinite(output).all()


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        StudentUNetConfig(image_size=100, channel_multipliers=(1, 2, 3, 4))
