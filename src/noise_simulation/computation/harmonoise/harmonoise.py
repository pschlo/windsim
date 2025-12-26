import numpy as np
import xarray as xr


def magnitude(da: xr.DataArray, dim: str):
    return np.sqrt((da**2).sum(dim))


### INPUT

def ground_profile() -> xr.Dataset:
    """N straight line segments with endpoints P_i, i=0,..,N.
    Store the (x,z)-position of each P_i.
    Segment i corresponds to the line from P_i to P_(i+1).
    """
    points_xz = np.array([
        [0, 5],
        [3, 11],
        [3.5, 10],
        [5, 15],
        [6, 30],
        [8, 0]
    ])
    ds = xr.Dataset(
        {
            'position': (['point', 'xz'], points_xz),
            'source_height': ([], 10),
            'receiver_height': ([], 10),
            'r': ([], 5),  # source-receiver-distance
            'f': ([], 1000),  # frequency
            'T': ([], 273 + 15),  # temperature in K
        },
        coords={
            'xz': ['x', 'z'],
            'point': np.arange(len(points_xz)),
            'segment': np.arange(len(points_xz)-1),
            'c_ref': 331,  # in m/s
            'T_ref': 273  # in K
        }
    )

    ds['N'] = len(points_xz)-1

    # sound speed
    ds['c_0'] = ds['c_ref'] * np.sqrt(ds['T']/ds['T_ref'])
    ds['lambda'] = ds['c_0'] / ds['f']
    ds['k'] = 2 * np.pi * ds['f'] / ds['c_0']

    # print(ds)
    # print()
    # print()
    # print()
    # print(ds)
    # exit()
    return ds




### GENERAL

def L(ds: xr.Dataset, alpha_air: np.float64) -> xr.Dataset:
    return ds.assign(L=L_source(ds) + delta_L_prop(ds, alpha_air=alpha_air))


def L_source(ds: xr.Dataset) -> xr.Dataset:
    return ds.assing(L_source=5)

def delta_L_prop(ds: xr.Dataset, alpha_air: np.float64) -> xr.Dataset:
    return ds.assign(
        delta_L_prop=delta_L_geo(ds) + delta_L_air(ds, alpha_air=alpha_air) + delta_L_excess(ds)
    )


def delta_L_geo(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(
        delta_L_geo = -10 * np.log10(4 * np.pi * ds['r']**2)
    )

def delta_L_air(ds: xr.Dataset, alpha_air: np.float64) -> xr.Dataset:
    """r: source-receiver distance
    alpha_air: air absorption coefficient"""
    return ds.assign(
        delta_L_air = -alpha_air * ds['r']
    )

def delta_L_excess(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    ds['delta_L_excess'] = (
        10 * np.log10(
            10**(ds['delta_L']/10 + ds['delta_L_scat']/10)
        )
    )
    return ds


### EXCESS ATTENUATION

def P_i_star(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()

    pos_star = ds['position'].copy()
    pos_star.loc[dict(point=0, xz='z')] += ds['source_height']
    pos_star.loc[dict(point=ds['N'], xz="z")] += ds['receiver_height']
    ds['position_star'] = pos_star
    return ds


# Excess attenuation recursion
def get_excess_attenuation(ds: xr.Dataset, i: np.integer, j: np.integer):
    """Returns delta_L.
    The idea is that the ground profile is divided into a number of ground sections between diffraction edges.
    In each iteration, we consider a ground profile i,...,j.
    If there is a diffraction edge k, we compute the diffraction and repeat the calculation for ground profiles
    i,...,k and k,...,j.
    If there is no diffraction edge, we have a ground section and can compute the ground attenuation.
    """
    # only the points between i and j are relevant now
    ds_local = ds.sel(point=slice(i, j))
    Pi = ds_local['position_star'].sel(point=i)
    Pj = ds_local['position_star'].sel(point=j)
    between_Pi_Pj = ds_local['position_star']

    # Check for every point between Pi and Pj if it is above the line Pi_Pj.
    #   (a x b > 0): b is rotated counterclockwise from a
    #   (a x b = 0): b is colinear to a
    #   (a x b < 0): b is rotated clockwise from a
    # Because the vector Pi_Pj always points right, the cases for a := Pi_Pj and point vector b correspond to
    # (point above the line, point on the line, point below the line).
    is_above = xr.cross(Pj - Pi, between_Pi_Pj - Pi, dim='xz') > 0

    # P_k is part of the dataset that only contains data for points in the set P_k
    set_Pk = between_Pi_Pj.where(is_above, drop=True)

    if set_Pk.sizes['point'] == 0:
        return delta_L_G(ds, i, j)

    # compute distance between point i and all points in P_k
    dist_ij = magnitude(Pi - Pj, 'xz')
    dist_ik = magnitude(Pi - set_Pk['position_star'], 'xz')
    dist_kj = magnitude(Pj - set_Pk['position_star'], 'xz')
    set_Pk['path_length_diff'] = dist_ik + dist_kj - dist_ij

    # choose point k with largest path length difference
    k = set_Pk['path_length_diff'].idxmax(dim='point').item()

    delta_L_D, p_D = delta_L_D_and_p_D(ds, i, k, j)

    return delta_L_D + get_excess_attenuation(ds, i, k) + get_excess_attenuation(ds, k, j)


def delta_L_D_and_p_D(ds: xr.Dataset, s: np.integer, p: np.integer, r: np.integer):
    """computes
        - diffraction attenuation L_D for diffraction from s to r at edge p
        - diffracted sound pressure amplitude p_D
    """

    S = ds['position_star'].sel(point=s)
    P = ds['position_star'].sel(point=p)
    R = ds['position_star'].sel(point=r)
    d_S = magnitude(S - P, 'xz')
    d_R = magnitude(P - R, 'xz')
    
    # - counterclockwise angle between positive y-axis and vector is same as
    #   counterclockwise angle between positive x-axis and vector rotated 90° clockwise.
    # - vector (x,y) rotated 90 deg clockwise is (y,-x).
    # - counterclockwise angle between positive x-axis and vector (x,y) is atan2(y,x).
    # - thus, counterclockwise angle between positive y-axis and vector (x,y) is atan2(-x,y).
    # - range of atan2 is (-pi, pi). Can be changed to (0, 2pi) by computing modulo 2pi.
    # - to get clockwise angle in (0, 2pi), subtract counterclockwise angle in (0, 2pi) from 2pi.

    PS = S - P
    theta_S = np.arctan2(-PS.sel(xz='x'), PS.sel(xz='z')) % (2*np.pi)

    PR = R - P
    theta_R = 2*np.pi - np.arctan2(-PR.sel(xz='x'), PR.sel(xz='z')) % (2*np.pi)

    theta = theta_S + theta_R

    if theta <= np.pi:
        d_d = np.sqrt(
            d_S**2
            + d_R**2
            - 2 * d_S * d_R * np.cos(theta)
        )
        delta = -(d_S + d_R - d_d)
    else:
        d_d = d_S + d_R
        epsilon = (
            (np.sqrt(d_S * d_R) / (d_S + d_R))
            * (theta - np.pi)
        )
        delta = d_d * (
            (1/2) * epsilon**2
            + (1/3) * epsilon**4
        )

    N_F = 2 * delta/ds['lambda']

    if N_F < -0.25:
        delta_L_D = 0
    elif N_F < 0:
        delta_L_D = -6 + 12 * np.sqrt(-N_F)
    elif N_F < 0.25:
        delta_L_D = -6 - 12 * np.sqrt(N_F)
    elif N_F < 1:
        delta_L_D = -8 - 8 * np.sqrt(N_F)
    else:
        delta_L_D = -16 - 10 * np.log10(N_F)

    p_D = (
        (np.exp(1j * ds['k'] * d_d) / d_d)
        * 10**(delta_L_D/20)
    )

    return delta_L_D, p_D


#### GROUND ATTENUATION

def delta_L_G(ds: xr.Dataset, i: np.integer, j: np.integer):
    """Compute ground attenuation for propagation from source P_i to receiver P_j
    over ground profile P_k with k = i,...,j.
    All points P_k are below the line from source i to receiver j."""
    ds = ds.copy()

    # each ground segment has own, local coordinate system dh
    # compute new source and receiver heights and store at first point of resp. segment
    # TODO


    ds = delta_L_Gc(ds, i, j)
    ds = delta_L_Gt(ds)
    # TODO: check if correct
    ds['delta_L_G'] = ds['delta_L_Gc'] + ds['delta_L_Gt']

    return ds

def delta_L_Gc(ds: xr.Dataset, i: np.integer, j: np.integer) -> xr.Dataset:
    ds = ds.copy()

    # TODO
    Q_k = Q(ds, i, j)  # spherical-wave reflection coefficient
    D_k = D(ds, i, j)  # geometrical weighting factor
    C_k = 1  # coherence factor

    # TODO
    f_c = 1  # transition frequency
    w_k = np.array([1,1,1,1,])  # modified Fresnel weight

    N_w = np.sum(w_k) # TODO: avoid np.sum, use xr.sum
    x_G = N_w / np.sqrt(1 + (ds['f']/f_c)**2)
    F_G = 1 - np.exp(-1 / x_G**2)


    delta_L_G_flat_k = 10 * np.log10(
        np.abs(1 + C_k * D_k * Q_k)**2
        + (1 - C_k**2) * np.abs(D_k * Q_k)**2
    )

    ds['delta_L_G_flat'] = np.sum(w_k * delta_L_G_flat_k) # TODO: avoid np.sum, use xr.sum
    
    ds['delta_L_G_valley'] = 10 * np.log10(
        np.abs(1 + np.sum(w_k + C_k + D_k + Q_k))**2
        + np.sum(w_k * (1-C_k**2) * np.abs(C_k * Q_k)**2)
    )

    ds['delta_L_Gc'] = (
        F_G * ds['delta_L_G_flat']
        + (1-F_G) * ds['delta_L_G_valley']
    )

    return ds

def Q(ds: xr.Dataset, i: np.integer, j: np.integer):
    """spherical-wave reflection coefficient. Must be calculated for ground segments k = i,...,j-1."""

    ds = ds.sel(point=slice(i, j), segment=slice(i, j-1))
    S = ds['position_star'].sel(point=i)
    R = ds['position_star'].sel(point=j)

    # extract start and end position for each segment
    pos1 = (
        ds['position']
        .sel(point=slice(i, j-1))
        .rename(point='segment')
    )
    pos2 = (
        ds['position']
        .sel(point=slice(i+1, j))
        .rename(point='segment')
        .assign_coords(segment=lambda ds: ds['segment']-1)
    )

    # Compute direction vector for each segment.
    # Goes from P_i to P_(i+1), i.e. in receiver direction,
    # and thus corresponds to the d axis in the local segment coord system.
    ds['direction'] = (
        ds['position']
        .diff('point', label='lower')
        .rename(point='segment')
        .pipe(lambda x: x / magnitude(x, 'xz'))  # normalize
    )

    # Compute normalized normal vector for each segment by rotating direction vector 90° counterclockwise.
    # Corresponds to the h axis in the local segment coord system.
    rot_matrix = xr.DataArray(
        [[0, -1], [1, 0]],
        coords=dict(i=['x', 'z'], j=['x', 'z'])
    )
    ds['normal'] = (
        rot_matrix
        .rename(j='xz')
        .dot(ds['direction'], 'xz')
        .rename(i='xz')
    )

    print(f"i: {i}, j: {j}")

    # Compute projection of S onto each segment.
    # This is also the origin of the local segment coord system
    projection_length = (S - pos1).dot(ds['direction'], dim='xz')
    proj = pos1 + projection_length * ds['direction']

    # find image points S'
    S_image = 2 * proj - S

    # The reflection point is the intersection of line S'-R with the ground segment.
    # Find reflection angle, i.e. angle between normal vector of the segment and vector S'-R
    S_image_R = R - S_image
    alpha = np.arccos(
        ds['normal'].dot(S_image_R, 'xz')
        / magnitude(S_image_R, 'xz')
    )
    
    # normalized ground impedance
    Z = 1

    # boundary-loss factor
    F_Q = 1

    # Compute signed distances from segment to S and R.
    #   (a x b > 0): b is rotated counterclockwise from a
    #   (a x b = 0): b is colinear to a
    #   (a x b < 0): b is rotated clockwise from a
    # Because in the local segment coord system the direction vector always points right,
    # the cases for direction vector a and point vector b correspond to
    # (point above the segment, point on the segment, point below the segment).
    h_S = xr.cross(ds['direction'], S - pos1, dim='xz')
    h_R = xr.cross(ds['direction'], R - pos1, dim='xz')
    h_m = (h_S + h_R) / 2

    # debugging: get segment types
    # TODO: fix unknown case
    segment_type = xr.where(
        (h_S > 0) & (h_R > 0),
        'concave',
        xr.where(
            (h_S < 0) | (h_R < 0),
            'convex',
            xr.where(
                (h_S == 0) & (h_R == 0),
                'hull',
                'unknown'  # TODO: what is the segment type for e.g. (h_S == 0 & h_R > 0) ?
            )
        )
    )
    # print(f'Segment type: {segment_type}')

    h_G = ds['lambda']/32

    n_G = 1 - 0.7 * np.exp(-h_m/h_G)

    # plane-wave reflection coefficient
    R_p = (
        (Z * np.cos(alpha) - 1)
        / (Z * np.cos(alpha) + 1)
    )

    Q = R_p + (1-R_p) * F_Q**n_G

    return Q


def D(ds: xr.Dataset, i: np.integer, j: np.integer):
    """Geometrical weighting factor. Must be calculated for ground segments k = i,...,j-1."""

    # def p_F(S, R):
        # return np.exp(1j * ds['k'] * distance.euclidean(ds))
    # differentiate four cases
    # D_k = 


def delta_L_Gt(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    return ds
