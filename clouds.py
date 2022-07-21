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

def virga_calc(optics_dir=None, mmw=None, fsed=None, radius=None, mass=None, p=None, t=None, kz=None, metallicity=1):

    '''
    Compute 1D cloud profile and optical properties given a 1D pressure temperature ad kzz profile

    Parameters
    ----------
    optics_dir: directory for virga optical properties files
    mmw: float
        mean molecular weight of atm
    fsed: float
        sedimentation efficiency
    radius: float
        radius of planet in Rjup
    mass:  float
        mass of planet in Mjup
    p: array
        pressure (1D)
    t: array
        temperature (1D)
    kz: array
        mixing coefficient (1D)
    metallicity:
        metallicity in NON log units, default 1 (solar)

    Returns
    -------
    all_out: dict
        1D cloud profile and optical properties
    '''

    mean_molecular_weight = mmw
    gases = vd.recommend_gas(p, t, metallicity, mean_molecular_weight)

    if 'KCl' in gases:
        gases.remove('KCl')
    print('Virga Gases:', gases)

    sum_planet = vd.Atmosphere(gases, fsed=fsed, mh=metallicity, mmw=mean_molecular_weight)
    sum_planet.gravity(radius=radius, radius_unit=u.Unit('R_jup'), mass=mass, mass_unit=u.Unit('M_jup'))
    sum_planet.ptk(df=pd.DataFrame({'pressure': p, 'temperature': t, 'kz': kz}))  # will add to this dict from MITgcm file

    all_out = sum_planet.compute(as_dict=True, directory=optics_dir)
    return (all_out)


def virga_3D(optics_dir=None, mmw=None, fsed=None, radius=None, mass=None, ptk_dat=None):
    '''

    This function runs Virga for a 3D atmosphere, by computing cloud profiles for every 1D column in the atmosphere.
    It formats the result to be compatible with picaso, and outputs a dictionary with all the necessary cloud
    parameters in 3D. Written by Danica Adams (Caltech)

    Parameters
    ----------

    optics_dir: str
        directory for Virga optical properties files
    mmw: float
        mean molecular weight
    fsed: float
        sedimentation efficiency
    radius: float
        planet radius in Jupiter radius
    mass: float
        planet mass in Jupiter mass
    ptk_dat: dictionary
        input PTK profile database or dict, use regridded by PICASO.

    Returns
    -------
    cld3d: dict
        3D dictionary with cloud properties that can be added to 3D atmosphere in PICASO
    '''

    shortdat = ptk_dat

    # CREATE EMPTY DICTIONARY
    cld3d = {i: {} for i in np.unique(shortdat['lat'])}  # dict for each lat, all lons,

    for ilat in np.unique(shortdat['lat']):

        #for each lon/lat, will take outputs from virga run

        cld3d[ilat] = {i: {} for i in np.unique(shortdat['lon'])}  # and make a 3D dictionary, cld3d, with them

    # VIRGA CALCULATIONS

    for ipos in range(0, len(shortdat['lat'])):
        outdat = virga_calc(optics_dir, mmw, fsed, radius, mass, shortdat['P'], shortdat['temp'][ipos, :],
                            shortdat['kzz'][ipos, :])
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]] = outdat  # add results from virga run to dict
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]]['g0'] = cld3d[shortdat['lat'][ipos]][
            shortdat['lon'][ipos]].pop('asymmetry')
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]]['opd'] = cld3d[shortdat['lat'][ipos]][
            shortdat['lon'][ipos]].pop('opd_per_layer')
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]]['w0'] = cld3d[shortdat['lat'][ipos]][
            shortdat['lon'][ipos]].pop('single_scattering')
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]]['wavenumber'] = cld3d[shortdat['lat'][ipos]][
            shortdat['lon'][ipos]].pop('wave')

    return (cld3d)