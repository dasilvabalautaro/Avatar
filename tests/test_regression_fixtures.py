import json
from pathlib import Path

from avatar_face.domain.models import AvatarPrompt
from avatar_face.infrastructure.dataset.procedural_generator import ProceduralAvatarDatasetGenerator

FIXTURES_PATH = Path("configs/regression-fixtures.json")


def test_regression_fixtures_match_the_frozen_smoke_dataset(tmp_path: Path) -> None:
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    dataset = fixtures["smoke_dataset"]
    result = ProceduralAvatarDatasetGenerator(image_size=dataset["image_size"]).generate(
        str(tmp_path / "dataset"), dataset["samples"], dataset["seed"]
    )
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    samples = {sample["identifier"]: sample for sample in manifest["samples"]}

    assert fixtures["status"] == "frozen"
    assert result.manifest_sha256 == dataset["manifest_sha256"]
    for expected in fixtures["caption_regression"]:
        actual = samples[expected["identifier"]]
        assert actual["split"] == expected["split"]
        assert actual["caption"] == expected["caption"]
        assert actual["sha256"] == expected["sha256"]


def test_frozen_prompt_regression_inputs_are_valid() -> None:
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

    prompts = [
        AvatarPrompt(text=prompt["text"], seed=prompt["seed"], image_size=prompt["image_size"])
        for prompt in fixtures["prompt_regression"]
    ]

    assert [prompt.seed for prompt in prompts] == [42, 7, 20_260_813, 2**32 - 1]
