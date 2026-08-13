import torch

from avatar_face.domain.feasibility import get_feasibility_profile
from avatar_face.infrastructure.feasibility.torch_model import (
    AvatarFaceFeasibilityModel,
    component_models,
    count_parameters,
    example_inputs,
)


def test_micro_model_produces_expected_image_shape() -> None:
    profile = get_feasibility_profile("micro")
    model = AvatarFaceFeasibilityModel(profile).eval()

    with torch.inference_mode():
        image = model(*example_inputs(profile))

    assert tuple(image.shape) == (1, 3, 256, 256)
    assert count_parameters(model) > 0
    assert float(image.min()) >= -1.0
    assert float(image.max()) <= 1.0


def test_bridge_components_cover_pipeline_contracts() -> None:
    profile = get_feasibility_profile("bridge")
    model = AvatarFaceFeasibilityModel(profile).eval()
    components = component_models(model)

    with torch.inference_mode():
        condition = components["encoder"][0](*components["encoder"][1])
        predicted = components["denoiser"][0](condition, components["denoiser"][1][1])
        image = components["decoder"][0](predicted)

    assert tuple(condition.shape) == (1, profile.model_width)
    assert tuple(predicted.shape) == (1, profile.latent_channels, 8, 8)
    assert tuple(image.shape) == (1, 3, 256, 256)
