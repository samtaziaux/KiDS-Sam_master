#!/usr/bin/python

"""
"Determine the shear as a function of radius from a galaxy."
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import astropy.io.fits as pyfits
import multiprocessing as mp
import numpy as np
import sys
import os
import time
from astropy import constants as const, units as u
import subprocess as sub
import shlex

# local
from . import (
    combine_covariance_plus_bootstrap as combine_covboot,
    combine_splits,
    plot_covariance_plus_bootstrap as plot_covboot,
    shear_plus_covariance_process as shearcov,
    shearcode_modules as shear,
    stack_shear_plus_bootstrap as stack_shearboot,
    distance,
    esd_utils)

if sys.version_info[0] == 3:
    xrange = range


start_tot = time.time()

# Important constants
inf = np.inf


def define_runparams(purpose, lens_binning, ncores, blindcats):

    # The number of catalogues that will be ran
    
    nruns = 1

    # Binnning information of the groups
    obsbins = shear.define_obsbins(1, lens_binning, [], [])
    binname, lens_binning, nobsbins, binmin, binmax = obsbins
    nobsbin = nobsbins # Starting value

    # Prepare the values for nsplits and nobsbins
    # this new variable is redundant
    #nsplits = ncores#/nobsbins
    #if nsplits == 0:
        #nsplits = 1
    if ncores == 0:
        ncores = 1
    nsplit = 1 # Starting value

    # Names of the blind catalogs
    blindcat = blindcats[0]

    #return nruns, nsplits, nsplit, nobsbins, nobsbin, blindcat
    return nruns, ncores, nsplit, nobsbins, nobsbin, blindcat


def run_shearcodes(purpose, nruns, nsplit, nsplits, nobsbin, nobsbins,
                   blindcat, blindcats, config_file):

    # The shear calculation starts here
    
    out = shearcov.main(nsplit, nsplits, nobsbin, blindcat, config_file, 0)
    
    # Combine the splits according to the purpose

    # Combining the catalog splits to a single output
    if ('bootstrap' in purpose) or ('catalog' in purpose):
        combine_splits.main(nsplit, nsplits, nobsbin, blindcat, config_file, 0)
    
    # Stacking the lenses into an ESD profile
    if ('bootstrap' in purpose) or ('catalog' in purpose):
        runblinds(stack_shearboot.main, blindcats, nsplit, nsplits, nobsbin,
                  config_file, purpose)
    
    # Creating the analytical/bootstrap covariance and ESD profiles
    if ('bootstrap' in purpose) or ('covariance' in purpose):
        runblinds(combine_covboot.main, blindcats, nsplit, nsplits, nobsbin,
                  config_file, purpose)
    
    # Plotting the analytical/bootstrap covariance and ESD profiles
    if ('bootstrap' in purpose) or ('covariance' in purpose):
        runblinds(plot_covboot.main, blindcats, nsplit, nsplits, nobsbin,
                  config_file, purpose)
    return


#def runblinds(codename, blindcats, nsplit, nsplits, nobsbin, config_file, purpose):
def runblinds(func, blindcats, nsplit, nsplits, nobsbin, config_file, purpose):

    # This allows STDIN to work in child processes
    fn = sys.stdin.fileno()

    # this allows for a single blindcat to have a name with more than one letter
    #if hasattr(blindcats, '__iter__') and len(blindcats) > 1:
    if 'bootstrap' in purpose:
        for blindcat in blindcats:
            func(nsplit, nsplits, nobsbin, blindcat, config_file, fn)

    else:
        if len(blindcats) > 1:
            pool = mp.Pool(len(blindcats))

        if len(blindcats) > 1:
            out = [pool.apply_async(func, args=(nsplit,nsplits,nobsbin,
                                            blindcat,config_file, fn))
                   for blindcat in blindcats]
            pool.close()
            pool.join()
            for i in out:
                i.get()
        else:
            func(nsplit, nsplits, nobsbin, blindcats, config_file, fn)

    return


def run_esd(config_file):

    np.seterr(divide='ignore', over='ignore', under='ignore',
          invalid='ignore')
          
    # Input for the codes
    kids_path, gama_path, colnames, kidscolnames, specz_file, m_corr_file, Om, Ol, Ok, h, z_epsilon, \
        folder, filename, purpose, Rbins, \
        Runit, ncores, lensid_file, lens_weights, lens_binning, \
        lens_selection, src_selection, cat_version, n_boot, \
        cross_cov, com, blindcats = \
            esd_utils.read_config(config_file)

    if cat_version == 2:
        print('\n \n \n \n \n')
        print('KiDS-DR1/2 is no longer supported, please use v1.7')
        raise SystemExit()

    print('\n \n \n \n \n')
    print('Running KiDS-GGL pipeline - signal extraction')
    #print 'Running:', purpose
    print()

    # Define the initial parameters for this shearcode run
    runparams = define_runparams(purpose, lens_binning, ncores, blindcats)
    nruns, nsplits, nsplit, nobsbins, nobsbin, blindcat = runparams

    # Excecute the parallelized shearcode run
    run_shearcodes(purpose, nruns, nsplit, nsplits, nobsbin, nobsbins,
                   blindcat, blindcats, config_file)

    end_tot = (time.time()-start_tot)/60
    print('Finished in: %g minutes' %end_tot)
    print()

    return
