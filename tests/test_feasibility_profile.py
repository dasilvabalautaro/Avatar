import pytest

from avatar_face.domain.feasibility import (
    FEASIBILITY_PROFILES,
    FeasibilityProfile,
    get_feasibility_profile,
)


def test_profiles_generate_256_pixels_and_increase_in_size() -> None:
    micro = FEASIBILITY_PROFILES["micro"]
    bridge = FEASIBILITY_PROFILES["bridge"]
    bridge_slim = FEASIBILITY_PROFILES["bridge-slim"]
    target = FEASIBILITY_PROFILES["target"]
    stress = FEASIBILITY_PROFILES["stress"]

    assert micro.image_size == 256
    assert micro.estimated_parameters < bridge.estimated_parameters
    assert bridge_slim.estimated_parameters < bridge.estimated_parameters
    assert bridge.estimated_parameters < target.estimated_parameters
    assert target.estimated_parameters < stress.estimated_parameters
    assert target.int8_parameter_budget_bytes < 250 * 2**20


def test_profile_rejects_incompatible_attention_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        FeasibilityProfile(
            name="invalid",
            vocabulary_size=10,
            maximum_tokens=4,
            text_width=8,
            model_width=10,
            depth=1,
            attention_heads=3,
            latent_size=8,
            latent_channels=4,
            image_size=16,
            steps=1,
            decoder_channels=(8, 4),
        )


def test_unknown_profile_lists_available_profiles() -> None:
    with pytest.raises(ValueError, match="bridge, bridge-fast, bridge-slim, micro, stress, target"):
        get_feasibility_profile("unknown")
