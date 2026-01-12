# windsim

Wind turbine simulation framework.

## Installation
This package is not currently uploaded to PyPI. Install as follows:

1. Find your release of choice [here](https://github.com/pschlo/windsim/releases)
2. Copy the link to `windsim-x.x.x.tar.gz`
3. Run `python -m pip install {link}`

You may also prepend a [direct reference](https://peps.python.org/pep-0440/#direct-references), which might be desirable for a `requirements.txt`.

## Building
The `.tar.gz` file in a release is the [source distribution](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist), which was created from the source code with `uv build --sdist`. [Built distributions](https://packaging.python.org/en/latest/glossary/#term-Built-Distribution) are not provided.

## Usage
Run `windsim noise` to start a noise simulation:
```
$ windsim noise --help
Usage: windsim noise [OPTIONS]

  Run a noise simulation.

Options:
  -r, --root DIRECTORY  Root directory of the data repository.  [default: ./repository]
  -p, --project TEXT    Project name.  [default: default]
  -c, --config FILE     Path to the config TOML file.  [default: ./config.toml]
  -h, --help            Show this message and exit.
```
An example config and data repository is provided in the `example` directory.
