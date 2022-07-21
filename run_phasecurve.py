
# coding: utf-8

# In[2]:


############## IMPORTS ###############

from phasecurves import thermal_phasecurve_3d

import numpy as np
import pandas as pd
import astropy.units as u
import pickle
import time

import linecache

import warnings
warnings.filterwarnings('ignore')
import astropy.units as u

import os,sys

from collections import defaultdict
from operator import itemgetter

############## INPUTS  ################

# MITgcm input properties

planet = 'wasp-43b'
nlon = 128  #number of longitudes
nlat = 64   #number of latitudes
nz = 40     #pressure layers
nwv = 196   #number of wavenumbers

# Planet properties

planet_C_to_O = 0.55  #C/O ratio
planet_mh = 1  #metallicity (xsolar)
meanmw = 2.2
rp = 1.036
mp = 2.034
rprs = 0.15942

# Star properties

rs = 0.667 #Rsun
T_star = 4520 #K
sma = 0.01526 #AU

# include these if you don't have a stellar file
logg_star = None
met_star = None

# Choose if you want clouds
add_clouds = False
f_sed = ['0.3'] #sedimentation efficiencies

# Choose if you want to reuse regridded PTK and chemistry
reuse_ptk = False

# Resolution
phase = 0
ng = 10  #lon
nt = 10  #lat

res = 1
wave_range = [0.95,1.77] # microns
nphases = 15
phase_range = [21.599999999999997, 338.4]

# Location of original MITgcm output PTK profiles
ptk_file = '/home/ninarobbins/data/mit_gcm/wasp-43b/wasp43b_solar_PTkzz_3d.dat'

# Location of Virga output cloud files
virgapath = '/home/ninarobbins/modeling_hot_jupiters/THERMAL_TESTS/virga/1xsolar/cloudy0.01/'
optics = '/home/ninarobbins/virga_optics/optics/'
# Location of stellar file
starfile= '/home/ninarobbins/data/stellar_models/new_wasp43_star_2013.dat'

# Location of filter response function

filt = '/home/ninarobbins/modeling_hot_jupiters/new_WFC3_G141_sensitivity.csv'

# Where spectra, chemistry and phase curves will be stored
pcpath = '/home/ninarobbins/modeling_hot_jupiters/THERMAL_TESTS/phasecurves/'+planet+'/1xsolar/newcode/'+planet + '-phasecurve-' + str(ng) + 'x' + str(nt) + '-newcode.pickle'
newpt_path = '/home/ninarobbins/modeling_hot_jupiters/THERMAL_TESTS/ptk/'+planet+'/5xsolar/'

###### RUN PHASE CURVE ######

t1 = time.time()

pc = thermal_phasecurve_3d(planet=planet, in_ptk=ptk_file, pcpath=pcpath, newpt_path=newpt_path, filt_path=filt, wv_range=wave_range, res=res, nphases=nphases, phase_range=phase_range, ng=ng, nt=nt, nlon=nlon, nlat=nlat, nz=nz, CtoO=planet_C_to_O, mh=planet_mh, mmw=meanmw, rp=rp, mp=mp, rs=rs, logg_s=logg_star, Ts=T_star, met_s=met_star, rprs=rprs, sma=sma, sw_units='um', sf_units='erg/cm2/s/Hz', in_star=starfile, cloudy=add_clouds, fsed = f_sed, optics_dir=optics, reuse_pt=reuse_ptk)

# This gives the 3D thermal flux for every orbital phase, Fp/Fs, wavelength and PICASO interpolated stellar flux

print(pc)

# Write into file

print('Writing phasecurve into file')
with open(pcpath, 'wb') as handle:
    pickle.dump(pc, handle, protocol=pickle.HIGHEST_PROTOCOL)


t2 = time.time()
print(t2-t1)







