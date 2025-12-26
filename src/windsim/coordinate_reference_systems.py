import pyproj


class UTM_WGS:
    @staticmethod
    def NORTH(zone: int):
        return pyproj.CRS.from_epsg(32600 + zone)

    @staticmethod
    def SOUTH(zone: int):
        return pyproj.CRS.from_epsg(32700 + zone)


class UTM_ETRS:
    @staticmethod
    def NORTH(zone: int):
        return pyproj.CRS.from_epsg(25800 + zone)

    @staticmethod
    def SOUTH(zone: int):
        return pyproj.CRS.from_epsg(25900 + zone)


class CRS:
    WGS84 = pyproj.CRS.from_epsg(4326)
    WEB_MERCATOR = pyproj.CRS.from_epsg(3857)
    SPHERICAL_MERCATOR = pyproj.CRS.from_epsg(3857)
    PSEUDO_MERCATOR = pyproj.CRS.from_epsg(3857)
    WORLD_CYLINDRICAL_EQUAL_AREA = pyproj.CRS.from_epsg(6933)
    LAMBERT_AZIMUTHAL_EQUAL_AREA = pyproj.CRS.from_epsg(3035)
    LAMBERT_CONFORMAL_CONIC = pyproj.CRS.from_epsg(3034)
    ICELAND = pyproj.CRS.from_epsg(4386)
    WORLD_MERCATOR = pyproj.CRS.from_epsg(3395)

    UTM_WGS = UTM_WGS
    UTM_ETRS = UTM_ETRS


def _is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def parse_name(name: str):
    name = name.strip()

    _name = name.lower()
    _name = ' '.join(_name.split())  # remove duplicate whitespaces
    _name = _name.replace(' ', '_').replace('-', '_')  # convert delimiters to underscore
    _name_parts = _name.split('_')

    crs_mapping = {
        'wgs84': CRS.WGS84,
        'web_mercator': CRS.WEB_MERCATOR,
        'spherical_mercator': CRS.SPHERICAL_MERCATOR,
        'pseudo_mercator': CRS.PSEUDO_MERCATOR,
        'world_mercator': CRS.WORLD_MERCATOR
    }
    if _name in crs_mapping:
        return crs_mapping[_name]
    
    if (
        len(_name_parts) == 3
        and _name_parts[0] == 'utm'
        and _name_parts[1] in ['wgs', 'etrs']
        and _is_int(_name_parts[2][:-1])
        and _name_parts[2][-1] in ['n', 's']
    ):
        num, hemisphere = int(_name_parts[2][:-1]), _name_parts[2][-1]
        obj = UTM_WGS if _name_parts[1] == 'wgs' else UTM_ETRS
        return obj.NORTH(num) if hemisphere == 'n' else obj.SOUTH(num)

    return pyproj.CRS(name)
