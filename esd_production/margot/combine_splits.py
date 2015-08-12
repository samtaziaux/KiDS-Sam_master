#!/usr/bin/python

"Part 4 of the module to determine the shear as a function of radius from a galaxy."

import pyfits
import numpy as np
import distance
import sys
import os
import time
import shearcode_modules as shear
from astropy import constants as const, units as u
import numpy.core._dotblas
import glob
from decimal import *


# Important constants
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
pix = 0.187 # Used to translate pixel to arcsec
alpha = 0.057 # Used to calculate m
beta = -0.37 # Used to calculate m
inf = np.inf


# Input parameters
Nsplit, Ncores, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, binnum, path_obsbins, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, path_output, path_catalogs, path_splits, path_results, purpose, O_matter, O_lambda, h, filename_addition, Ncat, splitslist, blindcat, blindcatnum, path_kidscats = shear.input_variables()

print 'Step 2: Combine splits into one catalogue'
print

if 'bootstrap' in purpose:
	purpose = purpose.replace('bootstrap', 'catalog')
	path_splits = '%s/splits_%s'%(path_catalogs, purpose)
	path_results = '%s/results_%s'%(path_catalogs, purpose)

# You can make two kinds of catalog
if 'catalog' in purpose:

	Nfofmin = 2
	Nfofmax = inf
	binname = 'No'
	path_obsbins = 'No'
	
	if Ncores < Nobsbins:
		Ncores = Nobsbins
		Nsplit = binnum-1

	Nobsbins = 1
	binnum = 1

	if centering == 'Cen':
		rankmin = rankmax = 1

	else:
		if all(np.array([rankmin,rankmax]) > 0): # Group catalog
			rankmin = 1
			rankmax = inf
		else: # Galaxy catalog
			rankmin = -999
			rankmax = inf


# Define the list of variables for the output filename
filename_var = shear.define_filename_var(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, 'binnum', Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h)

splitslist = np.array([])
# Find all created random splits
if ('random' in purpose):
	for x in xrange(Ncat):
		splitname = '%s/%s_%i_%s%s_split*.fits'%(path_splits.replace('bootstrap', 'catalog'), purpose.replace('bootstrap', 'catalog'), x+1, filename_var, filename_addition)
		splitfiles = glob.glob(splitname)
		splitslist = np.append(splitslist, splitfiles)
	splitslist = np.sort(splitslist)

	filename_var = '%i_%s'%(Ncat, filename_var) # Ncat is the number of existing catalogs
	print 'Number of new catalog:', Ncat

# Paths to the resulting files
outname = shear.define_filename_results(path_results, purpose, filename_var, filename_addition, Nsplit, blindcat)

# Stop if the output already exists.
if os.path.isfile(outname):
	print 'This output already exists:', outname
	print
	quit()

# Load the first split 
if ('random' in purpose):
	shearcatname = splitslist[0]
else:
	shearcatname = shear.define_filename_splits(path_splits, purpose, filename_var, 1, Ncores, filename_addition, blindcat)

shearcat = pyfits.open(shearcatname)
sheardat = shearcat[1].data

Rcenters = sheardat['Rcenter'][0]
Rbins = len(Rcenters)

columns = sheardat.columns.names # Names of all the columns
rmcolumns = ['ID', 'Rmin', 'Rmax', 'Rcenter', 'variance(e[A,B,C,D])']
index = [columns.index(rm) for rm in rmcolumns]
columns = np.delete(columns, index) # Names of the columns that need to be stacked

combcol = []

# Adding the lens ID's and the radial bins R
combcol.append(pyfits.Column(name='ID', format='J', array=sheardat['ID']))
print 'Combining: ID'

centers = ['Rmin', 'Rmax', 'Rcenter']

for r in centers:
	combcol.append(pyfits.Column(name=r, format='%iD'%Rbins, array=sheardat[r], unit='kpc/h%g'%h*100))
	print 'Combining:', r

# Adding all the columns that need to be stacked

for col in range(len(columns)):
	sumcol = 0
	print 'Combining:', columns[col]

	if ('random' in purpose):
		for shearcatname in splitslist:
			# Reading the shear catalogue			
			
			print '	', shearcatname
			shearcat = pyfits.open(shearcatname)
			sheardat = shearcat[1].data
			
			sumcol = sumcol+np.array(sheardat[columns[col]])
		combcol.append(pyfits.Column(name=columns[col], format='%iD'%Rbins, array=sumcol))
		
	else:
		for i in xrange(Ncores):

			# Reading the shear catalogue			
			shearcatname = shear.define_filename_splits(path_splits, purpose, filename_var, i+1, Ncores, filename_addition, blindcat)
			
			print '	', shearcatname
			shearcat = pyfits.open(shearcatname)
			sheardat = shearcat[1].data
			
			sumcol = sumcol+np.array(sheardat[columns[col]])
		combcol.append(pyfits.Column(name=columns[col], format='%iD'%Rbins, array=sumcol))

# Adding the values of the variances
combcol.append(pyfits.Column(name='variance(e[A,B,C,D])', format='4D', array=sheardat['variance(e[A,B,C,D])']))
print 'Combining: variance(e[A,B,C,D])'
print

# Writing the combined columns to a combined fits file
cols=pyfits.ColDefs(combcol)
tbhdu=pyfits.new_table(cols)

if os.path.isfile(outname):
	os.remove(outname)
	print 'Old file "%s" overwritten.'%outname
else:
	print 'New file "%s" written.'%outname

tbhdu.writeto(outname)
