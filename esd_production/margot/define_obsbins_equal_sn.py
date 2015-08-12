#!/usr/bin/python

"Part of the module to determine the shear as a function of radius from a galaxy."

# Import the necessary libraries
import pyfits
import numpy as np
import distance
import sys
import os
import time
import shearcode_modules as shear
from astropy import constants as const, units as u
import numpy.core._dotblas


# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number

# Input parameters
Nsplit, Ncores, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, binnum, path_obsbins, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, path_output, path_catalogs, path_splits, path_results, purpose, O_matter, O_lambda, h, filename_addition, Ncat, splitslist, blindcat, blindcatnum, path_kidscats = shear.input_variables()
purpose = 'shearcatalog'

# Define the list of variables for the input catalog
filename_var = shear.define_filename_var(purpose, centering, rankmin, rankmax, 2, inf, ZBmin, ZBmax, 'No', 1, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h)

# Importing the relevant data from the shear catalog
shearcatname = shear.define_filename_results(path_results, purpose.replace('bootstrap', 'catalog'), filename_var, filename_addition, Nsplit, blindcat)
sheardat = pyfits.open(shearcatname)[1].data

print 'Importing:', shearcatname

# Importing all GAMA data, and the information on radial bins and lens-field matching.
catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, Rcenters, nRbins, galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, obslist, obslimlist = shear.import_data(path_Rbins, path_kidscats, centering, 'bootstrap', Ncat, rankmin, rankmax, O_matter, O_lambda, h, binname, obslim)
galIDlist_matched = np.unique(np.hstack(catmatch.values()))

purpose = 'binlimits'

if not os.path.isdir(purpose):
	os.mkdir(purpose)

# Define the list of variables for the output filename
filename_var = shear.define_filename_var(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, binnum, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h)
filename_var = filename_var.replace('%iof'%Nobsbins, 's')
filename = '%s/%s_%s%s.txt'%('binlimits', purpose, filename_var, filename_addition)

# Defining the observable binnning range of the lenses

galIDlist = sheardat['ID'] # ID of all galaxies in the shear catalog
gammatlist = sheardat['gammat_%s'%blindcat] # Shear profile of each galaxy
gammaxlist = sheardat['gammax_%s'%blindcat] # Cross profile of each galaxy
wk2list = sheardat['lfweight*k^2'] # Weight profile of each galaxy
w2k2list = sheardat['lfweight^2*k^2'] # Squared lensfit weight times squared lensing efficiency
srcmlist = sheardat['bias_m'] # Bias profile of each galaxy
variance = sheardat['variance(e[A,B,C,D])'][0] # The variance

# Defining the observable binnning range of the groups
lenssel_binning = shear.define_lenssel(purpose, galranklist, rankmin, rankmax, Nfoflist, Nfofmin, Nfofmax, 'No', binnum, [], -inf, inf, obslim, obslimlist, obslim_min, obslim_max) # Mask the galaxies in the shear catalog, WITHOUT binning (for the bin creation)
galIDs, gammats, gammaxs, wk2s, w2k2s, srcms = shear.mask_shearcat(lenssel_binning, galIDlist, gammatlist, gammaxlist, wk2list, w2k2list, srcmlist) # Mask all quantities
obslist = obslist[lenssel_binning]

weightmask = (np.sum(wk2s, 1) > 0)

gammat = np.sum(gammats, 1)[weightmask]
gammax = np.sum(gammaxs, 1)[weightmask]
wk2 = np.sum(wk2s, 1)[weightmask]
w2k2 = np.sum(w2k2s, 1)[weightmask]
srcm = np.sum(srcms, 1)[weightmask]
obslist = obslist[weightmask]


ESDt, ESDx, error, bias = shear.calc_stack(gammat, gammax, wk2, w2k2, srcm, variance, 0) # Write the output to the bootstrap sample table

SNlist = ESDt/error

# Create a number of observable bins containing an equal S/N
sorted_obslist = np.vstack([SNlist, obslist])
sorted_obslist = ((sorted_obslist.T)[np.lexsort(sorted_obslist)]).T # Sort the observable values

sorted_SNlist = sorted_obslist[0]
sorted_obslist = sorted_obslist[1]

tot_SN = np.sum(SNlist)
cum_SN = np.cumsum(sorted_SNlist)

binrange = np.array([]) # This array will contain the binning range

for n in xrange(Nobsbins):

	for i in xrange(len(SNlist)):
		
		SN = cum_SN[i]

		if SN >= tot_SN * n/Nobsbins:
			binrange = np.append(binrange, sorted_obslist[i])
			break
binrange = np.append(binrange, sorted_obslist[-1])

print '%s bins:'%binname, binrange

# Print the observable bin limits to a file
with open(filename, 'w') as file:
	print >>file, '# Limits of %i %s bins:'%(Nobsbins, binname)

with open(filename, 'a') as file:
	for n in xrange(Nobsbins+1):
		
		print >>file, binrange[n]

print
print 'Written: Observable bin limits:', filename
