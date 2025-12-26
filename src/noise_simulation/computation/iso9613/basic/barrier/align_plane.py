from typing import Any, cast

import numpy as np
import numpy.typing as npt
import trimesh

ORIGIN = np.array([0,0,0])
X_AXIS = np.array([1,0,0])
Y_AXIS = np.array([0,1,0])
Z_AXIS = np.array([0,0,1])


def align_plane_transform(
    normal: npt.ArrayLike,
    point: npt.ArrayLike,
    align_x: npt.ArrayLike | None = None,
    align_y: npt.ArrayLike | None = None,
    direction_x: npt.ArrayLike | None = None,
    direction_y: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Returns a transformation matrix that transforms a given plane to the XY plane.

    The transformation includes the following steps:
        1. Translate so that the plane passes through the origin.
        2. Align the normal vector with the positive Z-axis.
        3. Optionally align the X and Y axes.
        4. Optionally reflect across X and Y axes based on the specified direction vectors.

    Parameters
    ----------
    normal
        Normal of the plane.
    point
        Any point on the plane. This point will also be the new origin.
    align_x
        Vector that should be aligned with the x-axis.
    align_y
        Vector that should be aligned with the y-axis.
    direction_x
        Vector that should point in positive x-direction.
    direction_y
        Vector that should point in positive y-direction.

    Returns
    ----------
    matrix
        Transformation matrix.
    """

    rotation = trimesh.transformations.rotation_matrix
    reflection = trimesh.transformations.reflection_matrix
    translation = trimesh.transformations.translation_matrix

    def _map(matrix: npt.NDArray, v: npt.ArrayLike, translate: bool) -> npt.NDArray:
        if not translate:
            return matrix[:3,:3] @ v
        return (matrix @ np.append(v, 1))[:3]

    # translate to origin
    matrix = translation(-np.asarray(point))

    # rotate normal vector to z-axis
    dot_product = np.dot(normal, Z_AXIS) / np.linalg.norm(normal)
    if not np.isclose(np.abs(dot_product), 1):
        # normal vector is not aligned with Z-axis
        angle = np.arccos(np.clip(dot_product, -1, 1))
        axis = np.cross(cast(Any, normal), Z_AXIS)
        matrix = rotation(angle, axis) @ matrix
    elif dot_product < 0:
        # normal vector is aligned, but in opposite direction
        matrix = reflection(ORIGIN, Z_AXIS) @ matrix

    # align given vectors with axes
    # this is about direction, not position; translation is thus ignored.
    if align_x is not None and align_y is not None:
        raise ValueError("Cannot align to both x and y axis")

    if align_x is not None:
        v = _map(matrix, align_x, translate=False)[:2]
        dot_product = v.dot(X_AXIS[:2]) / np.linalg.norm(v)
        if not np.isclose(np.abs(dot_product), 1):
            # vector is not aligned with X-axis
            angle = np.arccos(np.clip(dot_product, -1, 1))
            axis = np.sign(np.cross(v, X_AXIS[:2])) * Z_AXIS
            matrix = rotation(angle, axis) @ matrix

    if align_y is not None:
        v = _map(matrix, align_y, translate=False)[:2]
        dot_product = v.dot(Y_AXIS[:2]) / np.linalg.norm(v)
        if not np.isclose(np.abs(dot_product), 1):
            # vector is not aligned with Y-axis
            angle = np.arccos(np.clip(dot_product, -1, 1))
            axis = np.sign(np.cross(v, Y_AXIS[:2])) * Z_AXIS
            matrix = rotation(angle, axis) @ matrix

    # ensure direction vectors point in correct direction
    # this is about direction, not position; translation is thus ignored.
    if direction_x is not None:
        x = _map(matrix, direction_x, translate=False)[0]
        if x < 0:
            matrix = reflection(ORIGIN, X_AXIS) @ matrix

    if direction_y is not None:
        y = _map(matrix, direction_y, translate=False)[1]
        if y < 0:
            matrix = reflection(ORIGIN, Y_AXIS) @ matrix

    return matrix
