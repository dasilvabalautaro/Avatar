from avatar_face.presentation.cli import main


def test_validate_prompt_command(capsys: object) -> None:
    exit_code = main(["validate-prompt", "avatar minimalista", "--seed", "9"])

    assert exit_code == 0


def test_validate_prompt_returns_two_for_invalid_prompt(capsys: object) -> None:
    exit_code = main(["validate-prompt", " "])

    assert exit_code == 2


def test_validate_prompt_returns_two_for_minor_reference(capsys: object) -> None:
    exit_code = main(["validate-prompt", "avatar de un niño con gorra"])

    assert exit_code == 2


def test_describe_feasibility_command(capsys: object) -> None:
    exit_code = main(["describe-feasibility", "--profile", "target", "--json"])

    assert exit_code == 0


def test_benchmark_android_requires_serial() -> None:
    try:
        main(["benchmark-android"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("argparse debía rechazar un benchmark sin serial")
