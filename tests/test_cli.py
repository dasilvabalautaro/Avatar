from avatar_face.presentation.cli import main


def test_validate_prompt_command(capsys: object) -> None:
    exit_code = main(["validate-prompt", "avatar minimalista", "--seed", "9"])

    assert exit_code == 0


def test_validate_prompt_returns_two_for_invalid_prompt(capsys: object) -> None:
    exit_code = main(["validate-prompt", " "])

    assert exit_code == 2
