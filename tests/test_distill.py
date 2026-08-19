from __future__ import annotations

import json
from pathlib import Path

import pytest

from avatar_face.domain.attributes import ATTRIBUTE_ORDER, parse_caption
from avatar_face.domain.distill import build_distill_captions, distill_captions_payload
from avatar_face.presentation.cli import main


def test_build_is_deterministic() -> None:
    first = build_distill_captions(64, 42)
    second = build_distill_captions(64, 42)
    assert first == second
    assert first != build_distill_captions(64, 43)


def test_pairs_are_balanced_and_parseable() -> None:
    pairs = build_distill_captions(64, 42)
    expressions = [pair.attributes.expression for pair in pairs]
    assert all(expressions.count(value) == 16 for value in set(expressions))
    for pair in pairs[:8]:
        assert parse_caption(pair.caption) == pair.attributes
        assert "of an adult" in pair.caption


def test_seeds_are_unique_and_bounded() -> None:
    pairs = build_distill_captions(256, 42)
    seeds = {pair.seed for pair in pairs}
    assert len(seeds) == len(pairs)
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_split_scheme_matches_training_dataset() -> None:
    pairs = build_distill_captions(30, 42)
    assert [pair.split for pair in pairs[:3]] == ["test", "validation", "train"]


def test_payload_schema() -> None:
    payload = distill_captions_payload(32, 42)
    assert payload["schema_version"] == 1
    assert payload["samples"] == 32
    pairs = payload["pairs"]
    assert isinstance(pairs, list)
    assert set(pairs[0]["attributes"]) == set(ATTRIBUTE_ORDER)
    splits = payload["splits"]
    assert isinstance(splits, dict)
    assert sum(splits.values()) == 32


def test_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        build_distill_captions(4, 42)


def test_cli_generates_captions_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "captions.json"
    assert main(["generate-distill-captions", "--output", str(output), "--samples", "32"]) == 0
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["samples"] == 32
    result = json.loads(capsys.readouterr().out)
    assert result["output"] == str(output)
    assert len(result["sha256"]) == 64
    # Sin --overwrite el comando no pisa un archivo existente.
    assert main(["generate-distill-captions", "--output", str(output), "--samples", "32"]) == 2
