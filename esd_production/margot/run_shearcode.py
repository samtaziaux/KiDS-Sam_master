#!/usr/bin/python

"Determine the shear as a function of radius from a galaxy."

import pyfits
import numpy as np
import sys
import os
import time
from astropy import constants as const, units as u
import numpy.core._dotblas
import subprocess as sub
import shlex

# local
import distance
import esd_utils


start_tot = time.time()

# Important constants
inf = np.inf


def define_runparams(purpose, lens_binning, ncores, blindcats):

    # The number of catalogues that will be ran
    if 'random' in purpose:
        nruns = 100
        if 'combine' in purpose:
            nruns = 0
    else:
        nruns = 1

    # Binnning information of the groups
    binname = lens_binning.keys()[0]
    if binname != 'None': # If there is binning
        obsbins = lens_binning[binname]
        nobsbins = len(obsbins)
    else: # If there is no binning
        nobsbins = 1
    nobsbin = nobsbins # Starting value

    # Prepare the values for Ncores and Nobsbins
    nsplits = ncores/nobsbins
    if nsplits == 0:
        nsplits = 1
    nsplit = 1 # Starting value

    # Names of the blind catalogs
    blindcat = blindcats[0]

    return nruns, nsplits, nsplit, nobsbins, nobsbin, blindcat

    
def run_shearcodes(purpose, Nruns, Ncore, Ncores, Nobsbins, binnum, blindcats, blindcat):
    
    # The shear calculation starts here

    # Creating the splits
    for n in xrange(Nruns):
        ps = []
        for binnum in np.arange(Nobsbins)+1:
            for Ncore in np.arange(Ncores)+1:
                splitsname = 'python -W ignore shear+covariance.py'
                splitsname += ' %i %i %i %s &' \
                            %(Ncore, Ncores, binnum, blindcat)
                p = sub.Popen(shlex.split(splitsname))
                ps.append(p)
        for p in ps:
            p.wait()

    quit()

    # Combining the catalog splits to a single output
    if ('bootstrap' in purpose) or ('catalog' in purpose):
        runblinds('combine_splits.py', blindcats, Ncore, Ncores, binnum)

    # Stacking the lenses into an ESD profile
    if ('bootstrap' in purpose) or ('catalog' in purpose):
        runblinds('stack_shear+bootstrap.py', blindcats, Ncore, Ncores, binnum)        

    # Creating the analytical/bootstrap covariance and ESD profiles
    if ('bootstrap' in purpose) or ('covariance' in purpose):
        runblinds('combine_covariance+bootstrap.py', blindcats, Ncore, Ncores, binnum)

    # Plotting the analytical/bootstrap covariance and ESD profiles
    if ('bootstrap' in purpose) or ('covariance' in purpose):
        runblinds('plot_covariance+bootstrap.py', blindcats, Ncore, Ncores, binnum)

    return
    
    
def runblinds(codename, blindcats, Ncore, Ncores, binnum):
    
    ps = []
    for blindcat in blindcats:
        runname = 'python -W ignore %s'%codename
        runname += ' %i %i %i %s &' \
                %(Ncore, Ncores, binnum, blindcat)
        p = sub.Popen(shlex.split(runname))
        ps.append(p)
    for p in ps:
        p.wait()

    return


def main():

    # Importing the input parameters from the config file
    config_file = '/data2/brouwer/shearprofile/KiDS-GGL/ggl.config'

    # Input for the codes
    [kids_path, gama_path,
            Om, Ol, Ok, h,
            folder, filename, purpose, Rbins, ncores,
            lensid_file, group_centre, lens_binning, lens_selection,
            src_selection, blindcats] = esd_utils.read_config(config_file)

    print blindcats


    print '\n \n \n \n \n \n \n \n \n \n'
    print 'Running:', purpose
    print

    # Define the initial parameters for this shearcode run
    nruns, nsplits, nsplit, nobsbins, nobsbin, blindcat = define_runparams(purpose, lens_binning, ncores, blindcats)
    
    # Excecute the parallelized shearcode run
    run_shearcodes(purpose, nruns, nsplits, nsplit, nobsbins, nobsbin, blindcats, blindcat)

    end_tot = (time.time()-start_tot)/60
    print 'Finished in: %g minutes' %end_tot
    print

    return


main()

