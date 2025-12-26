from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Literal
from urllib.parse import urlparse
import asyncio

import cdsapi.api as cdsapi
import requests

CLIENT_URL = "https://cds.climate.copernicus.eu/api"
Dataset = Literal[
    "reanalysis-cerra-single-levels",
    "reanalysis-cerra-height-levels"
]


class AsyncClient:
    api_key: str
    session: requests.Session

    def __init__(self, api_key: str | None = None, progress: bool = True):
        if api_key is None:
            api_key = os.environ.get("CDS_KEY")
        if api_key is None:
            raise ValueError("Could not find API key")
        self.api_key = api_key
        self.progress = progress

        self.session = requests.Session()

    def _create_client(self):
        return cdsapi.Client(
            url=CLIENT_URL,
            key=self.api_key,
            wait_until_complete=True,
            progress=self.progress,
            session=self.session
        )

    async def submit(self, dataset: Dataset, built_request: dict[str, Any], *, name: str | None = None, suffix: str | None = None):
        """Submit a request to CDS API."""
        client = self._create_client()

        # Submit job and wait until ready
        result =  await asyncio.to_thread(client.retrieve, dataset, built_request)
        return Result(result, name=name, suffix=suffix)


class Result:
    _result: cdsapi.Result
    name: str
    suffix: str
    
    def __init__(self, result: cdsapi.Result, *, name: str | None, suffix: str | None = None) -> None:
        self._result = result
        self._online_path = PurePosixPath(Path(urlparse(self._result.location).path))

        if name is None:
            name = self._online_path.stem
        self.name = name

        if suffix is None:
            suffix = self._online_path.suffix
        elif not suffix.startswith('.'):
            suffix = '.' + suffix
        self.suffix = suffix


    async def download(self, dest: Path | str, temp_folder: Path | str | None = None) -> None:
        """Download a result from CDS API."""
        # Resolve folders
        dest = Path(dest).resolve()
        if dest.suffix != self.suffix:
            print(f'WARNING: Specified different file suffix "{dest.suffix}"')

        if temp_folder is not None:
            temp_folder = Path(temp_folder).resolve()
            temp_folder.mkdir(exist_ok=True)

        # Download to temp dir
        with TemporaryDirectory(dir=temp_folder) as tmpdir:
            tmpdir = Path(tmpdir)
            await asyncio.to_thread(self._result.download, tmpdir / dest.name)
            # Move to correct folder
            shutil.move(tmpdir / dest.name, dest)
