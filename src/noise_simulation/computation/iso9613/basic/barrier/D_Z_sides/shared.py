import numpy as np
import numpy.typing as npt

ORIGIN = np.array([0,0,0])
X_AXIS = np.array([1,0,0])
Y_AXIS = np.array([0,1,0])
Z_AXIS = np.array([0,0,1])


def calc_from_parts(z, e, K_met, frequencies):
    _lambda = 340 / frequencies

    C_2 = 20

    if e == 0:
        # single diffraction
        C_3 = 1
    else:
        # double diffraction
        C_3 = (
            (1 + (5 * _lambda / e)**2)
            / (1/3 + (5 * _lambda / e)**2)
        )

    # determine barrier attenuation D_Z
    z_min = (-2 * _lambda) / (C_2 * C_3 * K_met)
    D_Z = np.where(
        z > z_min,
        10 * np.log10(3 + (C_2 / _lambda) * C_3 * z * K_met),
        0
    )

    return D_Z
