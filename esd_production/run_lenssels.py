#!/usr/bin/python

"Determine the shear as a function of radius from a galaxy."

import pyfits
import numpy as np
import distance
import itertools
import sys
import os
import time
from astropy import constants as const, units as u
import numpy.core._dotblas
import subprocess as sub
import shlex
import argparse

start_tot = time.time()

# Important constants
inf = np.inf

"""
purposes = ['shearcatalog']
#purposes = ['binlimits']
obsbins = ['LumB', 'fLightBCG', 'MassA', 'MassAfunc', 'Nfof', 'Rad50', 'VelDisp']
centers = ['BCG']#, 'IterCen', 'Cen']
Rbins = ['10:20:2000']



for obsbin in obsbins:
	for purpose in purposes:
		
		selname = 'python run_shearcode.py --rank 1 1 --Nfof 5 inf --purpose %s --obsbins %s --path_obsbins AdBinMV_%s.dat --path_output obsbin_test/output_%sbins/old --path_Rbins 10:20:2000'%(purpose, obsbin, obsbin, obsbin)
		p = sub.Popen(shlex.split(selname))
		p.wait()


for purpose in purposes:
	for obsbin in obsbins:
		
		binlimits = 'binlimits/binlimits_rankBCG1-1_Nfof5-inf_%sbins6_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100.txt'%obsbin
#		binlimits = 6

		selname = 'python run_shearcode.py --rank 1 1 --Nfof 5 inf --purpose %s --path_output ../../shearcode_output/output_%s --obsbins %s --path_obsbins %s --path_Rbins 10:20:2000 --Ncores 3'%(purpose, obsbin, obsbin, binlimits)
		p = sub.Popen(shlex.split(selname))
		p.wait()

"""
# Marcello and Edo's GG-project:

purposes = ['covariance']
obsbin = 'logmstar'
#selections = ['', '--rank -999 -999', '--obslim uminusr --val_obslim 0 1.8', '--obslim uminusr --val_obslim 1.8 inf', '--obslim logLWage --val_obslim 0 9.4', '--obslim logLWage --val_obslim 9.4 inf', '--rank 1 1', '--rank 1 1 --Nfof 2 5', '--rank 1 1 --Nfof 5 inf', '--rank 2 inf', '--rank 2 inf --Nfof 2 5', '--rank 2 inf --Nfof 5 inf', '--obslim Z --val_obslim 0 0.2', '--obslim Z --val_obslim 0.2 0.5']

selections = ['--obslim Z --val_obslim 0.01 binlimits/zmax_logmstarbins_8_Edo.txt']

for purpose in purposes:
	for sel in selections:

		selname = 'python run_shearcode.py --purpose %s --obsbins %s --path_obsbins binlimits/%sbins_8_Edo.txt --path_Rbins 10:20:2000 --filename_addition oldcatmatch %s'%(purpose, obsbin, obsbin, sel)

		p = sub.Popen(shlex.split(selname))
		p.wait()
"""



# Cristobal's satellite project:

purposes = ['shearcatalog']

for purpose in purposes:
	selname = 'python run_shearcode.py --rank 2 inf --Nfof 5 inf --purpose %s --obsbins AngSepBCG --path_obsbins binlimits/DistanceFromBCG_50_200_350_1000.dat --path_Rbins 12:20:2000 --filename_addition oldcatmatch'%purpose
	p = sub.Popen(shlex.split(selname))
	p.wait()


purposes = ['covariance']

for purpose in purposes:
	selname = 'python run_shearcode.py --rank 2 inf --Nfof 5 inf --purpose %s --obsbins corr-AngSepBCG --path_obsbins binlimits/DistanceFromBCG_50_200_350_1000.dat --path_Rbins binlimits/9bins.txt --filename_addition oldcatmatch'%purpose
	p = sub.Popen(shlex.split(selname))
	p.wait()



# Environment project:

#obslims = np.array([6.0, 9.23128796, 10.23492718, 10.73139954, 13.0]) # Stellar mass bin limits
#obslims = np.array([6.0, 10.4, 10.9, 13.0])
#obslims = np.array([6.0, 10.2, 10.7, 13.0]) # Stellar mass bin limits
#obslims = np.array([6.0, 10.23492718, 13.0]) # Stellar mass bin limits
#obslims = np.array([6.0, 11.2, 13.0]) # Stellar mass bin limits

obslims = np.array([6.0, 10.23492718, 10.73139954, 13.0])
obsbins = 'binlimits/env_bins.txt'



for i in xrange(len(obslims)-1):

	selname = 'python run_shearcode.py --rank -999 -999 --obsbins shuffenvR4 --path_obsbins %s --obslim corr-logmstar --val_obslim %g %g --path_Rbins 5:20:2000'%(obsbins, obslims[i], obslims[i+1])
	p = sub.Popen(shlex.split(selname))
	p.wait()
	


for i in xrange(1):

	selname = 'python run_shearcode.py --rank -999 -999 --obsbins corr-logmstar --path_obsbins binlimits/logmstarbins_3_Margot --path_Rbins 5:20:2000'
	p = sub.Popen(shlex.split(selname))
	p.wait()
	



# New GAMA catalogue for Aaron

for i in xrange(1):

	selname = 'python run_shearcode.py --rank -999 inf --path_Rbins binlimits/16bins.txt --path_output /disks/shear10/brouwer/Aaron --Ncores 3'
	p = sub.Popen(shlex.split(selname))
	p.wait()

"""

