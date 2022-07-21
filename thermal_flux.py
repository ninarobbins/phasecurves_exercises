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


def correct_star_units(star_model, w_unit, f_unit):

    '''
    Correct from stellar units of original input file into PICASO units of ergs/cm3/s without regridding

    Parameters
    ----------
    star_model : str
        path to input stellar file
    w_unit : str
        units of wavelength in input stellar file
    f_units : str
        units of flux in input stellar file

    Returns
    -------
    ndarray
        Stellar wavelength in corrected units
    ndarray
        Stellar flux in corrected units
    '''

    # Correct from stellar units of original file into PICASO units of ergs/cm3/s

    star = np.genfromtxt(star_model, dtype=(float, float), names='w, f')
    flux = star['f']
    wave = star['w']

    # sort if not in ascending order
    sort = np.array([wave, flux]).T
    sort = sort[sort[:, 0].argsort()]
    wave = sort[:, 0]
    flux = sort[:, 1]
    if w_unit == 'um':
        WAVEUNITS = 'um'
    elif w_unit == 'nm':
        WAVEUNITS = 'nm'
    elif w_unit == 'cm':
        WAVEUNITS = 'cm'
    elif w_unit == 'Angs':
        WAVEUNITS = 'angstrom'
    elif w_unit == 'Hz':
        WAVEUNITS = 'Hz'
    else:
        raise Exception('Stellar units are not correct. Pick um, nm, cm, hz, or Angs')

        # http://www.gemini.edu/sciops/instruments/integration-time-calculators/itc-help/source-definition
    if f_unit == 'Jy':
        FLUXUNITS = 'jy'
    elif f_unit == 'FLAM':
        FLUXUNITS = 'FLAM'
    elif f_unit == 'erg/cm2/s/Hz':
        flux = flux * 1e23
        FLUXUNITS = 'jy'
    else:
        raise Exception('Stellar units are not correct. Pick FLAM or Jy or erg/cm2/s/Hz')

    sp = jdi.psyn.ArraySpectrum(wave, flux, waveunits=WAVEUNITS,
                                fluxunits=FLUXUNITS)  # Convert everything to nanometer for conversion based on gemini.edu
    sp.convert("um")
    sp.convert('flam')  # ergs/cm2/s/ang
    wno_star = 1e4 / sp.wave[::-1]  # convert to wave number and flip
    wv_star = 1e4 / wno_star
    flux_star = sp.flux[::-1] * 1e8  # flip and convert to ergs/cm3/s here to get correct order

    return (wv_star, flux_star)


def mask_star(wv_star, flux_star, choose_wave):
    '''

    Select wavelengths from stellar grid that correspond to range of filter wavelengths

    Parameters
    ----------
    wv_star : ndarray
        High resolution stellar wavelength grid with correct units
    flux_star : ndarray
        High resolution stellar flux grid with correct units
    choose_wave : list of floats
        Wavelength range of interest corresponding to filter wavelengths, in microns. e.g. [0.95,1.77] for HST WFC3 G141

    Returns
    -------
    wv_star_masked : ndarray
        Masked stellar wavelength array
    flux_star_masked : ndarray
        Masked stellar flux array
    '''

    l1 = choose_wave[0]
    l2 = choose_wave[1]

    mask_s = [(wv_star > l1) & (wv_star < l2)]
    wv_star_masked = wv_star[mask_s]
    flux_star_masked = flux_star[mask_s]

    return (wv_star_masked, flux_star_masked)


def interpolate_filter(filter_file, wv_star):
    '''
    Interpolate transmission function (e.g. HST WFC3 filter) onto high resolution stellar grid

    Parameters
    ----------
    filter_file : str
        Path to instrument filter transmission function (must be 2 columns: wavelength (in meters) and sensitivity)
    wv_star : ndarray
        High resolution stellar wavelength grid with correct units (microns and erg/cm3/s)

    Returns
    -------
    resp : ndarray
        Interpolated response function, onto higher resolution stellar wavelength grid
    '''

    wv_filter = np.genfromtxt(filter_file, usecols=0) * 1e6
    sns_filter = np.genfromtxt(filter_file, usecols=1)

    interpfunc = interpolate.interp1d(wv_filter, sns_filter, bounds_error=False, fill_value=0)
    resp = interpfunc(wv_star)

    ########## Make transmission 0 where is is supposed to be 0 #############
    start = wv_filter[0]
    end = wv_filter[len(wv_filter) - 1]

    wws = np.where(wv_star <= start)
    wwe = np.where(wv_star >= end)
    resp[wwe] = 0.0
    resp[wws] = 0.0

    return (resp)


def interpolate_planet(planet_flux, planet_wv, wv_star):
    '''
    Interpolate planet flux onto high resolution stellar grid.
    If using filter function, make sure wv_star is masked using mask_star

    Parameters
    ----------
    planet_flux: ndarray
        Thermal flux coming directly from PICASO spectrum function, flux to be interpolated
    planet_wv: ndarray
        Wavelength coming directly from PICASO spectrum function
    wv_star: ndarray
        Masked stellar wavelength onto which interpolate planet flux

    Returns
    -------
    fplan: ndarray
        Interpolated planet flux (onto masked stellar grid)
    '''

    interpfunc = interpolate.interp1d(planet_wv, planet_flux, bounds_error=False, fill_value=0)
    fplan = interpfunc(wv_star)

    return (fplan)


def transmitted_fpfs(planet_flux, star_flux, wv_star, resp, rprs, R):
    '''
    Calculate fraction of flux transmitted by instrument response.

    Parameters
    ----------
    planet_flux: ndarray
        1D array of planetary flux interpolated onto high resolution masked stellar grid
    star_flux : ndarray
        High resolution stellar flux
    wv_star : ndarray
        High resolution stellar wavelength
    resp : ndarray
        Instrument response function interpolated over high resolution masked stellar grid
    rprs : float
        Planet to star radius ratio

    Returns
    -------
    planet_trans :  ndarray
        Transmitted planetary thermal flux
    star_trans : ndarray
        Transmitted stellar flux
    '''

    h = 6.626e-27
    c = 2.998e14  # dist. units=microns

    diff = np.diff(wv_star)
    diff = (np.append(diff, diff[-1:]) + np.append(diff[1:], diff[-2:])) / 2

    planet_trans = planet_flux * resp * diff * wv_star / (h * c)  # apply weighting factor (filter) to every flux
    star_trans = star_flux * resp * diff * wv_star / (h * c)

    return (planet_trans, star_trans)


def whitelight_flux(planet_trans, star_trans, resp):
    '''
    Sum up fluxes from each wavelength band to obtain white light flux

    Parameters
    ----------
    star_trans : ndarray
        Stellar flux transmitted through filter (output of transmitted fpfs)
    planet_trans : ndarray
        Planet flux transmitted through filter (output of transmitted fpfs)
    resp : ndarray
        Instrument response function interpolated over high resolution masked stellar grid

    Returns
    -------
    sum_pflux : float
        White light planetary thermal flux
    sum_sflux : float
        White light stellar flux
    sum_weights :
        Sum of filter response weighting function
    '''

    sum_pflux = 0
    sum_sflux = 0
    sum_weights = 0

    for j in range(len(planet_trans)):  # planet_trans is array of len = # wavelengths

        sum_pflux += planet_trans[j]
        sum_sflux += star_trans[j]
        sum_weights += resp[j]

    return (sum_pflux, sum_sflux, sum_weights)


##### Main Function to compute 3D fluxes at different phases (e.g. phase curves) #####

def thermal_phasecurve(planet=None, in_ptk=None, filt_path=None, wv_range=None,
                       res=None, nphases=None,
                       p_range=None, ng=None, nt=None, nlon=None, nlat=None, nz=None, CtoO=None, mh=None, mmw=None,
                       rp=None, mp=None, rs=None, rprs=None, sma=None, R=None,
                       sw_units=None, sf_units=None, in_star=None, Ts=None,
                       logg_s=None, met_s=None, cloudy=False, fsed=None, optics_dir=None):
    
    '''
    Compute thermal phase curves from 3D pressure-temperature input (MITgcm). This function rotates the input 3D grid to select the visible hemisphere at each orbital phase angle, it computes the 3D chemistry profile and thermal spectrum at each orbital point. If clouds are turned on, it will also run virga to compute the 3D cloud profile, and use it to compute the spectrum.
    The 3D flux at each orbital point is then integrated vertically and for all angles, and then integrated for desired wavelength range to obtain one flux value per orbital phase.

    Parameters
    ----------
    planet: str
        Name of planet
    in_ptk: str
        Path to MITgcm pressure-temperature-Kzz profile file
    newpt_path: str
        Path to store regridded P-T-Kzz profiles
    filt_path: str
        Path to instrument response function
    wv_range: list
        Wavelength range of interest in microns (e.g. [0.95, 1.77] for HST WFC3 G141
    res: int
        Resampling value, default=1(increasing values reduce wavelength resolution)
    nphases: int
        Number of phases around the orbit (orbital phase resolution)
    p_range: list of floats
        Orbital phase range, in degrees from 0 to 360
    ng: int
        Number of Gauss angles
    nt: int
         Number of Chebyshev angles
    nlon: int
        Number of longitude points in MITgcm input file
    nlat: int
        Number of latitude points in MITgcm input file
    nz: int
        Number of pressure layers in MITgcm input file
    CtoO: float
        Carbon to Oxygen ratio
    mh: float
        Metallicity of planet
    mmw: float
        Atmospheric mean molecular weight
    rp: float
        Radius of planet in Jupiter radius
    mp: float
        Mass of planet in Jupiter mass
    rs: float
        Radius of star in Sun radius
    rprs: float
        Planet to star radius ratio
    sma: float
        Semi-major axis
    R: int
        Final wavelength resolution for plotting spectrum
    sw_units: str
        Stellar wavelength units from input stellar file (e.g. 'microns')
    sf_units: str
        Stellar flux units from input stellar file (e.g. erg/cm2/Hz/s)
    in_star: str
        Path to input stellar file
    Ts: float
        Star temperature
    logg_s: float
        Star Log G
    met_s: float
        Star metallicity
    cloudy: bool
        If True, clouds added into the calculation. Default = False
    fsed: float
        Sedimentation efficiency for cloud calculation (Virga)
    cld_path: str
        Path to save Virga cloud dictionaries
    optics_dir: str
        Path to virga Mie parameters and refractive indices for all species
    reuse_pt: bool
        If True, this function will search existing regridded chemistry (with the same resolution) in chempath to save time. Default=False
    reuse_cld: bool
        If True, this function will search existing Virga cloud files (with the same resolution) in cld_path to save time. Default=False

    Returns
    -------
    fluxes : dict
        Dictionary with thermal flux at every orbital phase.
    '''

    phases360 = np.linspace(p_range[0], p_range[1], nphases)
    phases = phases360 - 180  # go from 0-360 to -180-180

    print('THERMAL PHASE CURVE')
    print('Resolution: ', ng, 'x', nt)

    flag = 0
    fluxes = defaultdict(dict)  # inititate dictionary where phase curve data will be stored
    all_fpfs = np.zeros(nphases)
    all_phases = np.zeros(nphases)

    for iphase in range(0, nphases):

        # Regrid data and compute chemistry

        phase360 = phases360[iphase]
        phase = phases[iphase]

        print('PHASE:', round(phase360, 1))

        newptk, chem = regrid_and_chem_3D(planet=planet, input_file=in_ptk, orb_phase=phase, n_gauss_angles=ng,
                                              n_chebychev_angles=nt, nlon=nlon, nlat=nlat, nz=nz, CtoO=CtoO, mh=mh)

        # Compute cloud profiles with Virga

        if cloudy == True:
            cld_dat = virga_3D(optics_dir=optics_dir, mmw=mmw, fsed=fsed,
                                radius=rp, mass=mp, ptk_dat=newptk)

        else:
            cld_dat = None

        # Compute spectrum

        spectrum, star_dict = spectrum_3D(calc='thermal', res=res, ng=ng, nt=nt,
                                          waverange=wv_range, radius=rp, mass=mp, s_radius=rs, logg=logg_s, s_temp=Ts,
                                          s_met=met_s, sma=sma, chemdata=chem, starfile=in_star,
                                          s_w_units=sw_units, s_f_units=sf_units, clouds_dat=cld_dat, fsed=fsed)

        # Compute average flux over hemisphere

        thermal_3d = spectrum['full_output']['thermal_3d']
        fpfs_pic = spectrum['fpfs_thermal']
        wno = spectrum['wavenumber']
        wvl_pic = 1e4 / wno  # in microns

        if flag == 0:
            # write stellar flux into file only once
            fluxes['star_flux_picaso'] = star_dict['flux']

        ###### Calculate Fp/Fs for each phase and wavelength ###

        pflux = spectrum['thermal']  # get directly from picaso output spectrum
        ws, fs = 1e4 / star_dict['wno'], star_dict['flux']

        ##### Apply mask with wavelengths of interest ######

        wv_star, flux_star = mask_star(ws, fs, wv_range)

        ######### WFC3 sensitivity function, interpolate over stellar wavelength #########

        wfc3 = filt_path
        resp = interpolate_filter(wfc3, wv_star)

        ######## Calculate transmission #######

        avg_pflux = np.zeros(len(all_phases))  # to plot phase curves
        avg_sflux = np.zeros(len(all_phases))  # to plot phase curves

        # Interpolate planet flux onto stellar grid
        interpfunc = interpolate.interp1d(wvl_pic, pflux, bounds_error=False, fill_value=0)
        fplan = interpfunc(wv_star)

        # Calculate transmission in the filter band and FpFs for every phase

        planet_trans, star_trans = transmitted_fpfs(fplan, flux_star, wv_star, resp, rprs=rprs, R=R)

        ##### integrate flux over wavelength for white light phasecurve ######

        sum_pflux, sum_sflux, sum_weights = whitelight_flux(planet_trans, star_trans, resp)

        avg_pflux[flag] = sum_pflux  # array with weighted average planet flux
        avg_sflux[flag] = sum_sflux  # array with weighted average stellar flux

        fpfs_pc = (avg_pflux / avg_sflux) * (rprs ** 2)
        all_fpfs[flag] = fpfs_pc[flag]
        all_phases[flag] = round(phase360, 2)

        fluxes[round(phase360, 2)] = {'fpfs_spect': fpfs_pic, 'thermal': pflux, 'pflux_trans': planet_trans, 'sflux_trans': star_trans, 'wavelength': wv_star, 'thermal_3d': thermal_3d}

        print('Thermal flux added to phasecurve dictionary')
        
        flag += 1

    # Write into file

    fluxes['all_fpfs'] = all_fpfs
    fluxes['all_phases'] = all_phases

    return (fluxes)  # this is a dictionary
