# Windsim

Wind turbine noise and shadow simulation framework.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
a noise simulation against a data repository:

```console
uvx https://github.com/pschlo/windsim/archive/refs/heads/main.zip noise --root path/to/repository
```

Append `--help` to see the available commands and options. This command follows
the latest code on the `main` branch.

## Example

Download and extract the repository's
[source archive](https://github.com/pschlo/windsim/archive/refs/heads/main.zip),
then open a terminal in the extracted directory and run:

```console
uv run windsim noise
```

## Development

```console
uv sync --locked
uv run pytest
uv build
```

Run `uv lock --refresh-package planner` when you want to update the locked
Planner revision from its rolling `main` branch.
