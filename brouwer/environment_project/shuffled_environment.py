#!/usr/bin/python

""" Create "shuffled environments" that contain an equal number of random lenses inside each delta bin as the "real environments" """

import pyfits
import numpy as np
import sys
import os

import environment_utils as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['Computer Modern']})

# Names of the local densities in the delta_r catalog
deltanames = ['delta_R%i'%r for r in np.arange(8)+1]


# Import the GAMA and delta_r catalogues
gamacatdir = '/data2/brouwer/MergedCatalogues'
if os.path.isdir(gamacatdir):
    pass
else:
    gamacatdir = '/disks/shear10/brouwer_veersemeer/MergedCatalogues'

print gamacatdir

gamacatname = '%s/GAMACatalogue_1.0.fits'%gamacatdir
deltacatname = '%s/delta_r_catalog.fits'%gamacatdir
gamacat = pyfits.open(gamacatname)[1].data
deltacat =  pyfits.open(deltacatname)[1].data

galIDlist = gamacat['ID'] # IDs of all galaxies
envlist = gamacat['envS4'] # Environment of the galaxy (0 void, 1 sheet, 2 filament, 3 knot)

# This array will contain the shuffled environments for each Rmax
shuffle_results = np.zeros([len(deltanames), len(galIDlist)])


# For each value of Rmax ...
for d in xrange(len(deltanames)):
#for d in [3]:

    print
    print 'delta r:', d+1
    
    # Import the local density of the galaxies
    deltalist = deltacat[deltanames[d]]
    deltamask = deltalist > -999
    print 'Fraction deltamask:', np.float(sum(deltamask))/np.float(len(galIDlist))

    plotname = 'results/environment_histogram_delta'

    envnames = ['Void', 'Sheet', 'Filament', 'Knot']
    envcolors = ['red', 'green', 'blue', 'orange']

    # Create the delta_r histogram for every environment
    nbins = 60.
    deltabins, deltahists, histcens = utils.create_histogram(r'Local overdensity $\delta_%i$'%(d+1), deltalist[deltamask], nbins, envnames, envlist[deltamask], 'lin', False, False, '%s%i'%(plotname, d+1))

    # Create a dictionary with the IDs in each shuffled environment
    shuffled_IDs = dict()
    for env in np.arange(len(envnames)):
        shuffled_IDs[env] = np.array([])

 
    # For every delta_r bin ...
    for b in xrange(len(deltabins)-1):
        #print 
        #print 'delta_r bin:', b
        
        # Create a dictionary with the IDs of all galaxies in that bin
        deltabinmask = (deltabins[b] <= deltalist) & (deltalist <= deltabins[b+1]) & (envlist <= 3)
        galIDs_bin = galIDlist[deltabinmask]
        shuffIDs_bin = np.random.permutation(galIDs_bin) # ... and shuffle these IDs
        
        nmin = 0
        nmax = 0
        
        # For each environment ...
        for env in np.arange(len(envnames)):
    
            # Determine the number of shuffled galaxies that should go into this bin
            nmax = nmax + deltahists[env, b]

            # Add the right amount of shuffled galaxies in this bin to the shuffled environment dictionary
            shuffled_IDs[env] = np.hstack([shuffled_IDs[env], shuffIDs_bin[nmin:nmax]])
            
            # Make sure the next environment contains different galaxies
            nmin = nmin + deltahists[env, b]

    shuffled_IDs_tot = np.hstack(shuffled_IDs.values())
    print 'Shuffled fraction:', np.float(len(shuffled_IDs_tot))/np.float(len(galIDlist))
    
    # Create the list that will be printed to the fits file
    shuffled_env = np.ones(len(galIDlist))*-999

    # For each environment ...
    for env in np.arange(len(envnames)):
        print 'shuffled IDs in', envnames[env], ':', len(shuffled_IDs[env])

        # ... mask the galaxies that are not in this shuffled environment
        galIDmask = np.in1d(galIDlist, shuffled_IDs[env])
        
        # Assign the number of this shuffled environment to these galaxies
        shuffled_env[galIDmask] = env

    #print len(shuffled_IDs_tot), 'Unique', len(np.unique(shuffled_IDs_tot))

    shuffle_results[d] = shuffled_env

    # Test plot
    #deltabins, deltahists, histcens = utils.create_histogram(r'Local overdensity $\delta_%i$'%(d+1), deltalist[deltamask], nbins, envnames, (shuffle_results[d])[deltamask], 'lin', False, False, False)

"""
# Names of the shuffled environments
shuffenvnames = ['shuffenvR%i'%r for r in np.arange(8)+1]

# Write the results to a fits file
filename = '/data2/brouwer/MergedCatalogues/shuffled_environment_S4_deltaR.fits'

utils.write_catalog(filename, galIDlist, shuffenvnames, shuffle_results)
"""
