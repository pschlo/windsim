from typing import Any, Callable, Literal, NotRequired, TypedDict, Unpack, cast
from collections.abc import Collection
from dataclasses import dataclass

from ..common.explode_dict import (
    Explodable, OneOrMany, _Many, explode, each
)
from ..common.copernicus import (
    AsyncClient, Dataset
)

Variable = Literal[
    "turbulent_kinetic_energy",
    "wind_speed",
    "wind_direction"
]
HeightLevel = Literal[15, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500]
DataType = Literal["reanalysis"]
ProductType = Literal["forecast", "analysis"]
Time = Literal[0, 3, 6, 9, 12, 15, 18, 21]
Leadtime = Literal[
    1,   2,  3,  4,  5,  6,  7,  8,  9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
]
DataFormat = Literal["netcdf", "grib"]


class ExplodableRequestData(TypedDict):
    variable: Explodable[OneOrMany[Variable]]
    height_level_m: Explodable[OneOrMany[HeightLevel]]
    data_type: Explodable[OneOrMany[DataType]]
    product_type: Explodable[OneOrMany[ProductType]]
    year: Explodable[OneOrMany[int]]
    month: Explodable[OneOrMany[int]]
    day: Explodable[OneOrMany[int]]
    time_h: Explodable[OneOrMany[Time]]
    leadtime_hour: NotRequired[Explodable[OneOrMany[Leadtime]]]
    data_format: Explodable[DataFormat]


class AtomicRequestData(TypedDict):
    variable: OneOrMany[Variable]
    height_level_m: OneOrMany[HeightLevel]
    data_type: OneOrMany[DataType]
    product_type: OneOrMany[ProductType]
    year: OneOrMany[int]
    month: OneOrMany[int]
    day: OneOrMany[int]
    time_h: OneOrMany[Time]
    leadtime_hour: NotRequired[OneOrMany[Leadtime]]
    data_format: DataFormat


@dataclass(kw_only=True)
class AtomicRequest:
    name: str | None = None
    suffix: str | None = None
    dataset: Dataset
    data: AtomicRequestData
    built_data: dict[str, Any]


def explode_request_data(request: ExplodableRequestData) -> list[AtomicRequestData]:
    return cast(list[AtomicRequestData], explode(request))


class ValidationError(Exception):
    pass


def build_request_data(**kwargs: Unpack[AtomicRequestData]) -> dict[str, Any]:
    def as_list(value):
        if isinstance(value, _Many):
            if not value:
                raise ValueError(f"Collection of values must not be empty")
            return list(value)
        return [value]

    req = {
        "variable": [str(x) for x in as_list(kwargs["variable"])],
        "height_level": [f"{height}_m" for height in as_list(kwargs["height_level_m"])],
        "data_type": [str(x) for x in as_list(kwargs["data_type"])],
        "product_type": [str(x) for x in as_list(kwargs["product_type"])],
        "year": [str(year) for year in as_list(kwargs["year"])],
        "month": [f"{month:02d}" for month in as_list(kwargs["month"])],
        "day": [f"{day:02d}" for day in as_list(kwargs["day"])],
        "time": [f"{hour:02d}:00" for hour in as_list(kwargs["time_h"])],
        "data_format": kwargs["data_format"]
    }
    if "leadtime_hour" in kwargs:
        req["leadtime_hour"] = [str(x) for x in as_list(kwargs["leadtime_hour"])]

    # Validate
    if "forecast" in req["product_type"] and "leadtime_hour" not in req:
        raise ValidationError("Forecast requires leadtime hour")
    if "turbulent_kinetic_energy" in req["variable"] and "forecast" not in req["product_type"]:
        raise ValidationError("TKE requires product type 'forecast'")

    return req


class CerraClient:
    client: AsyncClient

    def __init__(self, api_key: str | None = None) -> None:
        self.client = AsyncClient(api_key=api_key)

    async def submit(self, request: AtomicRequest):
        return await self.client.submit(
            dataset=request.dataset,
            built_request=request.built_data,
            name=request.name
        )

    def prepare_request(self, request: ExplodableRequestData, *, on_validation_error: Literal["raise", "skip"] = "skip", get_name: Callable[[AtomicRequestData], str] | None = None):
        """
        Submit one or multiple cerra-heights requests to CDS API.
        Yields results when they become available.
        """
        # Generate list of atomic requests
        reqs: list[AtomicRequest] = []
        for r in explode_request_data(request):
            name = get_name(r) if get_name is not None else None

            try:
                built_data = build_request_data(**r)
            except ValidationError as e:
                if on_validation_error == "skip":
                    print(f"Skipping due to ValidationError: {e}")
                    continue
                else:
                    raise

            req = AtomicRequest(
                name=name,
                dataset="reanalysis-cerra-height-levels",
                data=r,
                built_data=built_data
            )
            reqs.append(req)

        return reqs

    def prepare_wind_request(self, year: int | Collection[int], month: int | Collection[int]):
        return self.prepare_request(
            {
                "variable": ["turbulent_kinetic_energy", "wind_speed", "wind_direction"],
                "height_level_m": [15, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500],
                # "height_level_m": 50,
                "data_type": "reanalysis",
                "product_type": "forecast",
                "year": year if isinstance(year, int) else each(*year),
                "month": month if isinstance(month, int) else each(*month),
                "day": list(range(1, 32)),
                "time_h": [0, 3, 6, 9, 12, 15, 18, 21],
                "leadtime_hour": [1, 2, 3],
                "data_format": "grib"
            },
            on_validation_error="skip",
            get_name=lambda r: f"year={r['year']},month={r['month']}",
        )
