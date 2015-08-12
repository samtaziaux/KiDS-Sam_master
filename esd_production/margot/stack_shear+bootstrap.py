#!/usr/bin/python

"Part of the module to determine the shear as a function of radius from a galaxy."

debug = False

# Import the necessary libraries
import pyfits
import numpy as np
import distance
import sys
import os
import time
import shearcode_modules as shear
from astropy import constants as const, units as u

# Important constants
G = const.G.to('pc3/Msun s2') # Gravitational constant
c = const.c.to('pc/s') # Speed of light
pix = 0.187 # Used to translate pixel to arcsec
alpha = 0.057 # Used to calculate m
beta = -0.37 # Used to calculate m
inf = np.inf # Infinity
nan = np.nan # Not a number


# Input parameters
Nsplit, Ncores, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, binnum, path_obsbins, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, path_output, path_catalogs, path_splits, path_results, purpose, O_matter, O_lambda, h, filename_addition, Ncat, splitslist, blindcat, blindcatnum, path_kidscats = shear.input_variables()

# Path to the output splits and results
path_splits = '%s/splits_%s'%(path_output, purpose)
path_results = '%s/results_%s'%(path_output, purpose)
path_catalog_splits = '%s/splits_%s'%(path_catalogs, purpose)
path_catalog_results = '%s/results_%s'%(path_catalogs, purpose)


if 'bootstrap' in purpose:
	print 'Step 3: Stack the lenses and create bootstrap samples'
else:
	print 'Step 3: Stack the lenses and create the ESD profile'

# Stop if the output ESD profile already exists
filename_var = shear.define_filename_var(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, binnum, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h)
if ('random' in purpose):
	filename_var = '%i_%s'%(Ncat, filename_var) # Ncat is the number of existing catalogs

filename_N1 = filename_var.replace('binnum', '%i'%binnum)
filenameESD = shear.define_filename_results(path_results, purpose, filename_N1, filename_addition, Nsplit, blindcat)
print filenameESD

if os.path.isfile(filenameESD):
	print 'This output already exists:', filenameESD
	print
	quit()

print


# Define the list of variables for the input catalog

if all(np.array([rankmin,rankmax]) > 0):
	filename_var = shear.define_filename_var('catalog', centering, 1, inf, 2, inf, ZBmin, ZBmax, 'No', 1, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h)
else:
	filename_var = shear.define_filename_var('catalog', centering, -999, inf, 2, inf, ZBmin, ZBmax, 'No', 1, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h)
if ('random' in purpose):
	filename_var = '%i_%s'%(Ncat, filename_var) # Ncat is the number of existing randoms

# Importing the relevant data from the shear catalog
shearcatname = shear.define_filename_results(path_catalog_results.replace('bootstrap', 'catalog'), purpose.replace('bootstrap', 'catalog'), filename_var, filename_addition, Nsplit, blindcat)

sheardat = pyfits.open(shearcatname)[1].data
print 'Importing:', shearcatname


# Importing all GAMA data, and the information on radial bins and lens-field matching.
catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, Rcenters, nRbins, galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, obslist, obslimlist = shear.import_data(path_Rbins, path_kidscats, centering, purpose.replace('catalog', 'bootstrap'), Ncat, rankmin, rankmax, O_matter, O_lambda, h, binname, obslim)
galIDlist_matched = np.unique(np.hstack(catmatch.values()))

# Define the list of variables for the output filename
if 'catalog' in purpose:
	filename_var = shear.define_filename_var(purpose.replace('catalog',''), centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, 'binnum', Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h)
else:
	filename_var = shear.define_filename_var(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, 'binnum', Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h)
if 'random' in purpose:
	filename_var = '%i_%s'%(Ncat, filename_var) # Ncat is the number of existing randoms
	print 'Number of existing random catalogs:', Ncat

galIDlist = sheardat['ID'] # ID of all galaxies in the shear catalog
gammatlist = sheardat['gammat_%s'%blindcat] # Shear profile of each galaxy
gammaxlist = sheardat['gammax_%s'%blindcat] # Cross profile of each galaxy
wk2list = sheardat['lfweight*k^2'] # Weight profile of each galaxy
w2k2list = sheardat['lfweight^2*k^2'] # Squared lensfit weight times squared lensing efficiency
srcmlist = sheardat['bias_m'] # Bias profile of each galaxy
variance = sheardat['variance(e[A,B,C,D])'][0] # The variance

# Defining the observable binnning range of the groups
lenssel_binning = shear.define_lenssel(purpose, galranklist, rankmin, rankmax, Nfoflist, Nfofmin, Nfofmax, 'No', binnum, [], -inf, inf, obslim, obslimlist, obslim_min, obslim_max) # Mask the galaxies in the shear catalog, WITHOUT binning (for the bin creation)
binrange, binmin, binmax = shear.define_obsbins(obslist, binname, path_obsbins, binnum, lenssel_binning)
Nobsbins = len(binrange)-1
lenssel_binning = [] # Empty unused lists


# Defining the number of bootstrap samples ( = 1 for normal shear stack)
if 'bootstrap' in purpose:
	Nbootstraps = 1e3
	
	# Selecting the random fields (must be the same for all observable bins)
	bootstrap_nums = np.random.random_integers(0,len(kidscats)-1,[Nbootstraps, len(kidscats)]) # Select Nkidsfields random KiDS fields between 0 and Nkidsfields-1 (Nbootstraps times)

#	if debug:
#		print 'randoms', np.shape(bootstrap_nums)
#		print bootstrap_nums

else:
	Nbootstraps = 1


# Define the labels for the plot
xlabel = r'radius R [kpc/h$_{%g}$]'%(h*100)
ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h$_{%g}$ M$_{\odot}$/pc$^2$]'%(h*100)


# Masking the list to keep only the specified galaxies
for binnum in np.arange(Nobsbins)+1:
	
	# Defining the min/max value of each observable bin
	binrange, binmin, binmax = shear.define_obsbins(obslist, binname, path_obsbins, binnum, lenssel_binning)

	print
	print '%s-bin %i of %i: %g - %g'%(binname, binnum, Nobsbins, binmin, binmax)
	
	# Mask the galaxies in the shear catalog
	lenssel = shear.define_lenssel(purpose, galranklist, rankmin, rankmax, Nfoflist, Nfofmin, Nfofmax, binname, binnum, obslist, binmin, binmax, obslim, obslimlist, obslim_min, obslim_max)

	if debug:
		print 'lenssel:', len(lenssel)
		print 'galIDlist:', len(galIDlist)

	galIDs, gammats, gammaxs, wk2s, w2k2s, srcms = shear.mask_shearcat(lenssel, galIDlist, gammatlist, gammaxlist, wk2list, w2k2list, srcmlist) # Mask all quantities
	galIDs_matched = galIDs[np.in1d(galIDs, galIDlist_matched)]
	galIDs_matched_infield = galIDs[np.in1d(galIDs, galIDs_infield)]

	if binname != 'No':
		print 'Mean %s value: %g'%(binname, np.mean(obslist[lenssel]))
	print 'Selected:', len(galIDs), 'galaxies, of which', len(galIDs_matched), 'overlap with KiDS.'
	print
	
#	if debug:
#		print 'galIDlist_matched', galIDlist_matched
#		print 'galIDs', galIDs

	# Paths to the resulting files
	filename_bin = filename_var.replace('binnum', '%s'%(binnum))

	# These arrays will contain the stacked profiles...
	field_shears = np.zeros([len(kidscats), 5, nRbins])
	outputnames = ['ESDt', 'ESDx', 'ESD(error)', 'bias']

	# Stack one shearprofile per KiDS field
	for k in xrange(len(kidscats)):
		
		# Mask all objects that are not in this field
		matched_galIDs = np.array(catmatch[kidscats[k]]) # The ID's of the galaxies that lie in this field
		field_mask = np.in1d(galIDs, matched_galIDs) # Define the mask
		galID, gammat, gammax, wk2, w2k2, srcm = shear.mask_shearcat(field_mask, galIDs, gammats, gammaxs, wk2s, w2k2s, srcms) # Mask all quantities

		if len(gammat) > 0: # If there are lenses in this field...
			field_shears[k] = np.array([sum(gammat,0), sum(gammax,0), sum(wk2,0), sum(w2k2,0), sum(srcm,0)]) # Add the field to the ESD-profile table
	
	# Taking the bootstrap samples
	if 'bootstrap' in purpose:
		print 'Number of bootstraps: %g'%Nbootstraps
		shear_bootstrap = np.sum(field_shears[bootstrap_nums], 1) # Randomly pick a number of fields equal to the total number of fields and sum them
		
		gammat, gammax, wk2, w2k2, srcm = [shear_bootstrap[:, x] for x in xrange(5)] # The summed quantities
		output_bootstrap = np.array(shear.calc_stack(gammat, gammax, wk2, w2k2, srcm, variance, blindcatnum)) # Write the output to the bootstrap sample table
		
		splitname = shear.define_filename_splits(path_splits, purpose, filename_bin, 0, 0, filename_addition, blindcat)
		
		# Write the results to a bootstrap catalog
		e1 = e2 = w = srcm = []
		shear.write_catalog(splitname, np.arange(Nbootstraps), Rbins, Rcenters, nRbins, output_bootstrap, outputnames, variance, purpose, e1, e2, w, srcm)

		error_tot = np.zeros(nRbins)
		for r in xrange(nRbins):
			bootstrap_mask = (0 < wk2[:, r]) & (wk2[:, r] < inf) # Mask values that do not have finite values (inf or nan)
			
			if sum(bootstrap_mask) > 0: # If there are any values that are not masked
				error_tot[r] = (np.var((output_bootstrap[0,:,r])[bootstrap_mask], 0))**0.5

	# Calculating the normal shear profile
	shear_sample = np.array(np.sum(field_shears, 0)) # Sum all fields
	
	gammat, gammax, wk2, w2k2, srcm = [shear_sample[x] for x in xrange(5)] # The summed quantities
	output = np.array(shear.calc_stack(gammat, gammax, wk2, w2k2, srcm, variance, blindcatnum)) # Write the output to the bootstrap sample table

	ESDt_tot, ESDx_tot, error_poisson, bias_tot = [output[x] for x in xrange(len(outputnames))]
	if 'bootstrap' not in purpose:
		error_tot = error_poisson


	# Printing the ESD profile to a file
	
	# Path to the output plot and text files
	shearcatname = shear.define_filename_results(path_results, purpose, filename_var, filename_addition, Nsplit, blindcat)
	plotname = '%s/%s_%s%s_%s.txt'%(path_results, purpose, filename_bin, filename_addition, blindcat)

	# Printing stacked shear profile to a txt file
	shear.write_stack(plotname, Rcenters, ESDt_tot, ESDx_tot, error_tot, bias_tot, h, variance, blindcatnum, galIDs_matched, galIDs_matched_infield)
	

	# Plotting the data for the separate observable bins
	plottitle = shear.define_plottitle(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, obslim, obslim_min, obslim_max, binname, binrange, ZBmin, ZBmax)
	
	# What plotting style is used (lin or log)
	if 'random' in purpose:
		plotstyle = 'lin'
	else:
		plotstyle = 'log'
	
	if binname == 'No':
		plotlabel = ylabel
	else:
		plotlabel = r'%.3g $\leq$ %s $\textless$ %.3g (%i lenses)'%(binmin, binname.replace('_', ''), binmax, len(galIDs_matched))

	try:
		shear.define_plot(plotname, plotlabel, plottitle, plotstyle, Nobsbins, xlabel, ylabel, binnum)
	except:
		pass

# Writing and showing the plot
try:
	shear.write_plot(plotname, plotstyle)
except:
	pass
