import numpy as np
import pandas as pd
import astropy.units as u
import pickle
import os, sys
from scipy import interpolate
import virga.justdoit as vd
import virga.justplotit as vp
import time

from picaso import build_3d_input as threed
from picaso import disco
from scipy.spatial import cKDTree

import linecache
import matplotlib
import matplotlib.pyplot as plt
import math

import warnings

warnings.filterwarnings('ignore')
import astropy.units as u

from picaso import justdoit as jdi
from picaso import justplotit as jpi

import sys

from collections import defaultdict
from operator import itemgetter

from picaso.disco import get_angles_3d



def regrid_and_chem_3D(planet=None, input_file=None, orb_phase=None, time_after_pa=None,
                       n_gauss_angles=None, n_chebychev_angles=None, nlon=None, nlat=None, nz=None, CtoO=None, mh=None):
    '''
    Function to regrid MITgcm input data into the chosen low resolution grid in PICASO. Selects the visiblle hemisphere
    at each point in the orbit by rotating the grid. It computes the 3D chemistry for the full planet.

    Parameters
    ----------
    planet: str
        Name of planet
    input_file: str
        input MITgcm file, see necessary format at https://natashabatalha.github.io/picaso/notebooks/9_Adding3DFunctionality.htm
    chempath: str
        Path to store chemistry files
    newpt_path: str
        Path to store regridded PTK data
    orb_phase: float
        orbital phase angle (Earth facing longitude), in degrees from -180 to 180, for planet rotation
    time_after_pa: float
        Time after periapse, for eccentric planets
    n_gauss_angles: int
        number of Gauss angles
    n_chebychev_angles: int
        number of Chebyshev angles
    nlon: int
        number of longitude points in MITgcm grid
    nlat: int
        number of latitude points in MITgcm grid
    nz: int
        number of pressure layers in MITgcm grid
    CtoO: float
        C to O ratio
    mh: int
        metallicity, in NON log units (1 is solar)

    Returns
    -------
    newdat: dict
        Dictionary with regridded pressure-temperature-kzz profiles
    chem3d: dict
        Dictionary with computed chemistry, to be added into 3D atmosphere
    '''

    if time_after_pa != None:  # for eccentric planets
        infile = open(input_file, 'r')

        # skip headers and blank line -- see GCM output
        line0 = infile.readline().split()
        line1 = infile.readline().split()

    else:
        infile = open(input_file, 'r')

    gangle, gweight, tangle, tweight = disco.get_angles_3d(n_gauss_angles, n_chebychev_angles)
    ubar0, ubar1, cos_theta, latitude, longitude = disco.compute_disco(n_gauss_angles, n_chebychev_angles, gangle, tangle, phase_angle=0)

    all_lon = np.zeros(nlon * nlat)  # 128 x 64 -- first, check size of GCM output
    all_lat = np.zeros(nlon * nlat)
    p = np.zeros(nz)
    t = np.zeros((nlon, nlat, nz))
    kzz = np.zeros((nlon, nlat, nz))

    total_pts = nlon * nlat

    ctr = -1

    for ilon in range(0, nlon):
        for ilat in range(0, nlat):
            ctr += 1

            # skip blank line -- check GCM output formatting

            temp = infile.readline().split()

            if planet == 'wasp-43b':
                all_lon[ctr] = float(temp[0]) + orb_phase #- 45  make sure if you need to shift your grid by a longitude
            else:
                all_lon[ctr] = float(temp[0]) + orb_phase

            all_lat[ctr] = float(temp[1])

            # read in data for each grid point
            for iz in range(0, nz):
                temp = infile.readline().split()
               
                p[iz] = float(temp[0])
		t[ilon, ilat, iz] = float(temp[1])
                kzz[ilon, ilat, iz] = float(temp[2])
            temp = infile.readline()

    lon = np.unique(all_lon)
    lat = np.unique(all_lat)

    # REGRID PTK

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lon2d = lon2d.flatten() * 180 / 3.141592
    lat2d = lat2d.flatten() * 180 / 3.141592

    xs, ys, zs = jpi.lon_lat_to_cartesian(np.radians(all_lon), np.radians(all_lat))
    xt, yt, zt = jpi.lon_lat_to_cartesian(np.radians(lon2d), np.radians(lat2d))

    nn = int(total_pts / (n_gauss_angles * n_chebychev_angles))

    tree = cKDTree(list(zip(xs, ys, zs)))              # these are the original 128x64 angles
    d, inds = tree.query(list(zip(xt, yt, zt)), k=nn)  # this grid is for ngxnt angles, regridding done here

    new_t = np.zeros((n_gauss_angles * n_chebychev_angles, nz))
    new_kzz = np.zeros((n_gauss_angles * n_chebychev_angles, nz))

    for iz in range(0, nz):
        new_t[:, iz] = np.sum(t[:, :, iz].flatten()[inds], axis=1) / nn
        new_kzz[:, iz] = np.sum(kzz[:, :, iz].flatten()[inds], axis=1) / nn

    newdat = {'lon': lon2d, 'lat': lat2d, 'temp': new_t, 'P': p, 'kzz': new_kzz}

    print('T, Kzz ready')

    # START CHEMISTRY

    input3d = {i: {} for i in latitude}

    print('starting chem')
    for ilat in range(n_chebychev_angles):
        for ilon in range(n_gauss_angles):
            case1 = jdi.inputs(chemeq=True)
            df = pd.DataFrame({'temperature': new_t[ilat * n_chebychev_angles + ilon, :], 'pressure': p})
            case1.inputs['atmosphere']['profile'] = df
            case1.chemeq(CtoO, mh)

            # Save as df within dictionary
            df2 = case1.inputs['atmosphere']['profile']
            df2['kzz'] = new_kzz[ilat * n_chebychev_angles + ilon, :]

            input3d[latitude[ilat]][longitude[ilon]] = df2

    print('chem ready')

    return (newdat, input3d)
