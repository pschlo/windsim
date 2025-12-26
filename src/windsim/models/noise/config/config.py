from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import toml

from .exceptions import ConfigError
from .sections.root import ConfigData

SUFFIXES = {".toml", ".json"}


class Config:
    def __init__(self, raw: dict[str, Any]):
        self.data = ConfigData(raw)

    @staticmethod
    def load(
        full_path: str | Path | None = None,
        *,
        dir_name: str | Path | None = None,
        name: str | Path | None = None
    ) -> Config:
        """Loads the config file.

        Parameters
        -----------
        full_path : optional
            Full path to the config file.
        dir_name : optional
            The directory in which to search for the config file.
            Used as a search hint if `full_path` is not provided.
            By default, the directory of the file that was executed is used.
        name : optional
            The name of the config file.
            Used as a search hint if `full_path` is not provided.
        """

        if full_path is not None:
            if dir_name is not None or name is not None:
                raise ValueError('"full_path" is mutually exclusive with "dir" and "name"')
        else:
            full_path = Config._find(dir_name=dir_name, name=name)
        raw = Config._load_raw(full_path)
        return Config(raw)

    @staticmethod
    def _find(
        *,
        dir_name: str | Path | None = None,
        name: str | Path | None = None
    ) -> Path:
        """
        Searches for a config file.
        The returned path is guaranteed to have a valid suffix.

        Raises `ConfigError` if config could not be found.
        """

        # determine search directory
        if dir_name is not None:
            dir_name = Path(dir_name)

        parent_dir: Path | None
        m = sys.modules['__main__']
        if hasattr(m, '__file__') and m.__file__ is not None:
            # was executed through file; search next to file
            parent_dir = Path(m.__file__).parent.resolve()
        else:
            parent_dir = None

        if dir_name is not None:
            if dir_name.is_absolute():
                pass
            else:
                if parent_dir is not None:
                    dir_name = parent_dir / dir_name
                else:
                    raise ConfigError("Relative dirname")
        else:
            # dir_name is None
            if parent_dir is not None:
                dir_name = parent_dir
            else:
                raise ConfigError("Could not find config file")

        # search for name in directory
        if name is not None:
            name = Path(name)
            if (dir_name / name).exists():
                # name exists in directory
                files = [dir_name / name]
            else:
                # check for name + suffix
                files = [p for p in dir_name.iterdir() if p.suffix in SUFFIXES and p.stem == name.name]
        else:
            # consider all files with 'conf' in their name
            files = [p for p in dir_name.iterdir() if p.suffix in SUFFIXES and 'conf' in p.name]

        if len(files) == 0:
            raise ConfigError("Could not find config file")
        elif len(files) > 1:
            raise ConfigError("Too many potential config files")
        return files[0]

    @staticmethod
    def _load_raw(path: str | Path) -> dict[str, Any]:
        """Loads the given config file."""
        path = Path(path)
        if path.suffix == ".json":
            with open(path, "r") as f:
                raw = json.load(f)
        elif path.suffix == ".toml":
            with open(path, "r") as f:
                raw = toml.load(f)
        else:
            msg = "Could not load config: "
            if not path.suffix:
                raise ValueError(msg + "File has no suffix")
            raise ValueError(msg + f'Unknown file suffix "{path.suffix}"')
        return raw
