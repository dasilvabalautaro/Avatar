import json
from pathlib import Path

import pytest

from avatar_face.domain.models import AvatarPrompt, InvalidPromptError
from avatar_face.infrastructure.dataset.procedural_generator import ProceduralAvatarDatasetGenerator

FIXTURES_PATH = Path("configs/regression-fixtures.json")


def test_regression_fixtures_match_the_frozen_smoke_dataset(tmp_path: Path) -> None:
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    dataset = fixtures["smoke_dataset"]
    generator = ProceduralAvatarDatasetGenerator(image_size=dataset["image_size"])
    result = generator.generate(
        str(tmp_path / "dataset-a"), dataset["samples"], dataset["seed"]
    )
    repeated = generator.generate(str(tmp_path / "dataset-b"), dataset["samples"], dataset["seed"])
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    samples = {sample["identifier"]: sample for sample in manifest["samples"]}

    assert fixtures["status"] == "frozen"
    # Los bytes PNG pueden variar entre builds de Pillow/libpng. La regresión
    # exige determinismo dentro del entorno y el preflight valida los hashes del
    # paquete transferido, que es la evidencia portable entre hosts.
    assert result.manifest_sha256 == repeated.manifest_sha256
    for expected in fixtures["caption_regression"]:
        actual = samples[expected["identifier"]]
        assert actual["split"] == expected["split"]
        assert actual["caption"] == expected["caption"]


def test_frozen_prompt_regression_inputs_are_valid() -> None:
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

    prompts = [
        AvatarPrompt(text=prompt["text"], seed=prompt["seed"], image_size=prompt["image_size"])
        for prompt in fixtures["prompt_regression"]
    ]

    assert [prompt.seed for prompt in prompts] == [42, 7, 20_260_813, 2**32 - 1]


def test_frozen_minor_prompts_are_rejected() -> None:
    """RF-09: todo prompt que sugiera un avatar de un menor debe rechazarse."""
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

    for prompt in fixtures["minor_prompt_rejection"]:
        with pytest.raises(InvalidPromptError, match="menor de edad"):
            AvatarPrompt(text=prompt["text"], seed=42, image_size=256)
