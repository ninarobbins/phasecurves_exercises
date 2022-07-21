
# coding: utf-8

# In[2]:


############## IMPORTS ###############

from phasecurves import thermal_phasecurve

import numpy as np
import pandas as pd
import pickle
import os,sys
import time

############## INPUTS  ################

planet = 'wasp-43b'
nlon = 128  #number of longitudes
nlat = 64   #number of latitudes
nz = 40     #pressure layers
nwv = 196   #number of wavenumbers

planet_C_to_O = 0.55  #C/O ratio
planet_mh = 1         #metallicity (xsolar)
meanmw = 2.2

rp = 1.036
mp = 2.034
rprs = 0.15942

rs = 0.667 #Rsun
Ts = 4520 #K
sma = 0.01526 #AU

f_sed = 0.01 #sedimentation efficiencies

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
virgapath = '/home/ninarobbins/modeling_hot_jupiters/THERMAL_TESTS/virga/1xsolar/virga_1.1/fsed_'+str(f_sed)+'/'
#optics = '/home/ninarobbins/virga_optics/optics/'
optics = '/home/ninarobbins/virga_optics_1.1/'

# Location of stellar file
starfile= '/home/ninarobbins/data/stellar_models/new_wasp43_star_2013.dat'

# Location of filter response function
filt = '/home/ninarobbins/modeling_hot_jupiters/THERMAL_TESTS/filters/spitzer-ch1-3.6microns.txt'

# Where chemistry and phase curves will be stored
chem_path = '/home/ninarobbins/modeling_hot_jupiters/THERMAL_TESTS/chemistry/'+planet+'/1xsolar/10x10/'
pcpath = '/home/ninarobbins/modeling_hot_jupiters/THERMAL_TESTS/phasecurves/'+planet+'/1xsolar/cloudy/virga_1.1/spitzer/'+planet + '-phasecurve-' + str(ng) + 'x' + str(nt) + '-spitzer_3.6um-fsed_'+str(f_sed)+'.pickle'
newpt_path = '/home/ninarobbins/modeling_hot_jupiters/THERMAL_TESTS/ptk/'+planet+'/1xsolar/10x10/'

###### RUN PHASE CURVE ######

t1 = time.time()

pc = thermal_phasecurve(planet=planet, in_ptk=ptk_file, chempath=chem_path, newpt_path=newpt_path, filt_path=filt, wv_range=wave_range, res=res, nphases=nphases, p_range=phase_range, ng=ng, nt=nt, nlon=nlon, nlat=nlat, nz=nz, CtoO=planet_C_to_O, mh=planet_mh, mmw=meanmw, rp=rp, mp=mp, rs=rs, rprs=rprs, sma=sma, R=100, sw_units='um', sf_units='erg/cm2/s/Hz', in_star=starfile, fsed = f_sed, cld_path = virgapath, cloudy = False, optics_dir=optics, reuse_pt=True, reuse_cld = True)

print('Writing phasecurve into file')
with open(pcpath, 'wb') as handle:
    pickle.dump(pc, handle, protocol=pickle.HIGHEST_PROTOCOL)


# This gives the 3D thermal flux for every orbital phase, Fp/Fs, wavelength and PICASO interpolated stellar flux

t2 = time.time()
print(t2-t1)

