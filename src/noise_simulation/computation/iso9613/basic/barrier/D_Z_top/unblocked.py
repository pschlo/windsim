import numpy as np
import numpy.typing as npt

from .shared import calc_from_parts


def D_Z_unblocked_ufunc(frequencies: npt.NDArray) -> npt.NDArray:
    """Shortcut for computing D_Z where the plane EV is known to not contain any barriers."""
    return calc_from_parts(
        z=0,
        e=0,
        K_met=1,
        frequencies=frequencies
    )
