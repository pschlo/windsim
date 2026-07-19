# Windsim

Wind turbine noise and shadow simulation framework.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Git](https://git-scm.com/downloads), then run a noise simulation against a data
repository:

```console
uvx git+https://github.com/pschlo/windsim.git noise --root path/to/repository
```

Append `--help` to see the available commands and options.

## Example

The included example data is available from a checkout:

```console
git clone https://github.com/pschlo/windsim.git
cd windsim
uv run windsim noise
```

## Development

```console
uv sync --locked
uv run pytest
uv build
```
