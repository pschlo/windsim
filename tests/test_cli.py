from click.testing import CliRunner

from windsim.cli import cli


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "noise" in result.output
    assert "shadow" in result.output


def test_noise_help() -> None:
    result = CliRunner().invoke(cli, ["noise", "--help"])

    assert result.exit_code == 0
    assert "--root" in result.output
    assert "--project" in result.output
