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
G = const.G.to('pc3/Msun s2') # Gravitational constant
c = const.c.to('pc/s') # Speed of light
pix = 0.187 # Used to translate pixel to arcsec
alpha = 0.057 # Used to calculate m
beta = -0.37 # Used to calculate m
inf = np.inf # Infinity

# Input parameters
Nsplit, Ncores, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, binnum, path_obsbins, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, path_output, path_catalogs, path_splits, path_results, purpose, O_matter, O_lambda, h, filename_addition, Ncat, splitslist, blindcat, blindcatnum, path_kidscats = shear.input_variables()
print 'Final step: Plot the ESD profiles and correlation matrix'
print

# Plot settings:

# Plotting the data for the separate observable bins
if 'random' in purpose:
	plotstyle = 'lin' # What plotting style is used (lin or log)
else:
	plotstyle = 'log'
subplots = binnum # Are there subplots?
Nrows = 1 # If so, how into many rows will the subplots be devided?

# Creating the ueber-matrix plot (covlin, covlog, corlin, corlog)
plotstyle_matrix = 'corlin'


# Define the list of variables for the output filename
filename_var = shear.define_filename_var(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, 'binnum', Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h) # Defining the name of the output files

if ('random' in purpose) or ('star' in purpose):
	filename_var = '%i_%s'%(Ncat, filename_var) # Ncat is the number of existing randoms
	print 'Number of existing random catalogs:', Ncat

# Paths to the resulting files
outname = shear.define_filename_results(path_results, purpose, filename_var, filename_addition, Nsplit, blindcat)

# Importing all GAMA data, and the information on radial bins and lens-field matching.
catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, Rcenters, nRbins, galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, obslist, obslimlist = shear.import_data(path_Rbins, path_kidscats, centering, purpose, Ncat, rankmin, rankmax, O_matter, O_lambda, h, binname, obslim)

# Binnning information of the groups
lenssel = shear.define_lenssel(purpose, galranklist, rankmin, rankmax, Nfoflist, Nfofmin, Nfofmax, 'No', binnum, [], -inf, inf, obslim, obslimlist, obslim_min, obslim_max) # Mask the galaxies in the shear catalog, WITHOUT binning (for the bin creation)
binrange, binmin, binmax = shear.define_obsbins(obslist, binname, path_obsbins, binnum, lenssel)


# Writing and showing the plots

plottitle1 = shear.define_plottitle(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, obslim, obslim_min, obslim_max, binname, binrange, ZBmin, ZBmax)

xlabel = r'radius R [kpc/h$_{%g}$]'%(h*100)
ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h$_{%g}$ M$_{\odot}$/pc$^2$]'%(h*100)

if 'bootstrap' not in purpose:
	# Plotting the shear profiles for all observable bins
	for N1 in xrange(Nobsbins):

		binrange, binmin, binmax = shear.define_obsbins(obslist, binname, path_obsbins, N1+1, lenssel)

		filename_N1 = filename_var.replace('binnum', '%i'%(N1+1))
		filenameESD = shear.define_filename_results(path_results, purpose, filename_N1, filename_addition, Nsplit, blindcat)

		plotlabel = r'%g $\leq$ %s $\textless$ %g'%(binmin, binname.replace('_', ''), binmax)
		try:
			shear.define_plot(filenameESD, plotlabel, plottitle1, plotstyle, subplots, xlabel, ylabel, N1+1)
		except:
			pass
	try:
		shear.write_plot(filenameESD, plotstyle)
	except:
		pass

# Creating the ueber-matrix plot
filename_N1 = filename_var.replace('binnumof', 's')
filenamecov = '%s/%s_matrix_%s%s_%s.txt'%(path_results, purpose, filename_N1, filename_addition, blindcat)


# The Group bins
if binname == 'No': # If there is no binning
	plottitle2 = ''
else: # If there is binning
	plottitle2 = r'for %i %s bins between %g and %g.'%(Nobsbins, binname, binrange[0], binrange[Nobsbins])

	shear.plot_covariance_matrix(filenamecov, plottitle1, plottitle2, plotstyle_matrix, binname, binrange, Rbins, h)

# Remove the used splits
if (Nsplit==0) and (blindcat=='A') and (binnum<=1):
	filelist = os.listdir(path_splits)

	for filename in filelist:
		os.remove('%s/%s'%(path_splits, filename))
