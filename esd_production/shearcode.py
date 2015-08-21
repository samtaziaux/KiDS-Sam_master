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
import shearcode_modules as shear
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
    binname, lens_binning, nobsbins, binmin, binmax = shear.define_obsbins(1, lens_binning, [], [])
    nobsbin = nobsbins # Starting value

    # Prepare the values for nsplits and nobsbins
    nsplits = ncores/nobsbins

    if nsplits == 0:
        nsplits = 1
    nsplit = 1 # Starting value

    # Names of the blind catalogs
    blindcat = blindcats[0]

    return nruns, nsplits, nsplit, nobsbins, nobsbin, blindcat

    
def run_shearcodes(purpose, nruns, nsplit, nsplits, nobsbin, nobsbins, blindcat, blindcats, config_file):

    # The shear calculation starts here
    directory = os.getcwd()
    path_shearcodes = 'esd_production/'
    if 'esd_production' in directory:
        path_shearcodes = ''
        

    # Creating the splits
    for n in xrange(nruns):
        ps = []
        for nobsbin in np.arange(nobsbins)+1:

            for nsplit in np.arange(nsplits)+1:

                splitsname = 'python -W ignore %sshear+covariance.py'%(path_shearcodes)
                splitsname += ' %i %i %i %s %s &' \
                            %(nsplit, nsplits, nobsbin, blindcat, config_file)
                p = sub.Popen(shlex.split(splitsname))
                ps.append(p)
        for p in ps:
            p.wait()

    
    # Combine the splits according to the purpose

    # Combining the catalog splits to a single output
    if ('bootstrap' in purpose) or ('catalog' in purpose):
        runblinds('%scombine_splits.py'%(path_shearcodes), blindcats, nsplit, nsplits, nobsbin, config_file)

    # Stacking the lenses into an ESD profile
    if ('bootstrap' in purpose) or ('catalog' in purpose):
        runblinds('%sstack_shear+bootstrap.py'%(path_shearcodes), blindcats, nsplit, nsplits, nobsbin, config_file)        

    # Creating the analytical/bootstrap covariance and ESD profiles
    if ('bootstrap' in purpose) or ('covariance' in purpose):
        runblinds('%scombine_covariance+bootstrap.py'%(path_shearcodes), blindcats, nsplit, nsplits, nobsbin, config_file)

    # Plotting the analytical/bootstrap covariance and ESD profiles
    if ('bootstrap' in purpose) or ('covariance' in purpose):
        runblinds('%splot_covariance+bootstrap.py'%(path_shearcodes), blindcats, nsplit, nsplits, nobsbin, config_file)

    return
    
    
def runblinds(codename, blindcats, nsplit, nsplits, nobsbin, config_file):
    
    ps = []
    for blindcat in blindcats:
        runname = 'python -W ignore %s'%codename
        runname += ' %i %i %i %s %s &' \
                %(nsplit, nsplits, nobsbin, blindcat, config_file)
        p = sub.Popen(shlex.split(runname))
        ps.append(p)
    for p in ps:
        p.wait()

    return


def run_esd(config_file):

    # Input for the codes
    [kids_path, gama_path,
            Om, Ol, Ok, h,
            folder, filename, purpose, Rbins, Runit, ncores,
            lensid_file, group_centre, ranks, lens_binning, lens_selection,
            src_selection, blindcats] = esd_utils.read_config(config_file)

    print '\n \n \n \n \n \n \n \n \n \n'
    print 'Running:', purpose
    print

    # Define the initial parameters for this shearcode run
    nruns, nsplits, nsplit, nobsbins, nobsbin, blindcat = define_runparams(purpose, lens_binning, ncores, blindcats)
    
    # Excecute the parallelized shearcode run
    run_shearcodes(purpose, nruns, nsplit, nsplits, nobsbin, nobsbins, blindcat, blindcats, config_file)

    end_tot = (time.time()-start_tot)/60
    print 'Finished in: %g minutes' %end_tot
    print

    return
