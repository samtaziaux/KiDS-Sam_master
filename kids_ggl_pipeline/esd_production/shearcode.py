#!/usr/bin/python

"""
"Determine the shear as a function of radius from a galaxy."
"""

import astropy.io.fits as pyfits
import numpy as np
import sys
import os
import time
from astropy import constants as const, units as u
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
        if ncores == 0:
            nruns = 0
    else:
        nruns = 1

    # Binnning information of the groups
    obsbins = shear.define_obsbins(1, lens_binning, [], [])
    binname, lens_binning, nobsbins, binmin, binmax = obsbins
    nobsbin = nobsbins # Starting value

    # Prepare the values for nsplits and nobsbins
    nsplits = ncores#/nobsbins

    if nsplits == 0:
        nsplits = 1
    nsplit = 1 # Starting value

    # Names of the blind catalogs
    blindcat = blindcats[0]

    return nruns, nsplits, nsplit, nobsbins, nobsbin, blindcat

    
def run_shearcodes(purpose, nruns, nsplit, nsplits, nobsbin, nobsbins,
                   blindcat, blindcats, config_file):

    # The shear calculation starts here
    directory = os.path.dirname(os.path.realpath(__file__)) #os.getcwd()
    indirectory = os.listdir('.')
    #path_shearcodes = 'esd_production'
    if 'esd_production' in directory:
        path_shearcodes = directory + '/'
    if 'esd_production' in indirectory:
        path_shearcodes = 'esd_production'
    
    """
    # Creating the splits
    for n in xrange(nruns):
        ps = []
        for nobsbin in np.arange(nobsbins)+1:

            for nsplit in np.arange(nsplits)+1:
                
                splitsname = 'python -W ignore %sshear+covariance.py' \
                             %(path_shearcodes)
                splitsname += ' %i %i %i %s %s &' \
                              %(nsplit, nsplits, nobsbin,
                                blindcat, config_file)
                p = sub.Popen(shlex.split(splitsname))
                ps.append(p)
        for p in ps:
            p.wait()
    """
    
    for n in xrange(nruns):
        ps = []
        splitsname = 'python -W ignore %sshear+covariance.py'%(path_shearcodes)
        splitsname += ' %i %i %i %s %s &'%(nsplit, nsplits, nobsbin, \
                                           blindcat, config_file)
        p = sub.Popen(shlex.split(splitsname))
        ps.append(p)
        for p in ps:
            p.wait()
    
    # Combine the splits according to the purpose

    # Combining the catalog splits to a single output
    if ('bootstrap' in purpose) or ('catalog' in purpose):
        ps = []
        codename = '%scombine_splits.py'%(path_shearcodes)
        runname = 'python -W ignore %s'%codename
        runname += ' %i %i %i %s %s &' \
                %(nsplit, nsplits, nobsbin, blindcat, config_file)
        p = sub.Popen(shlex.split(runname))
        ps.append(p)
        for p in ps:
            p.wait()

    # Stacking the lenses into an ESD profile
    if ('bootstrap' in purpose) or ('catalog' in purpose):
        runblinds('%sstack_shear+bootstrap.py' \
                  %(path_shearcodes), blindcats,
                    nsplit, nsplits, nobsbin, config_file, purpose)

    # Creating the analytical/bootstrap covariance and ESD profiles
    if ('bootstrap' in purpose) or ('covariance' in purpose):
        runblinds('%scombine_covariance+bootstrap.py' \
                  %(path_shearcodes), blindcats,
                    nsplit, nsplits, nobsbin, config_file, purpose)
    
    # Plotting the analytical/bootstrap covariance and ESD profiles
    if ('bootstrap' in purpose) or ('covariance' in purpose):
        runblinds('%splot_covariance+bootstrap.py' \
                  %(path_shearcodes), blindcats,
                    nsplit, nsplits, nobsbin, config_file, 'covariance')
    
    return
    
    
def runblinds(codename, blindcats, nsplit, nsplits, nobsbin, config_file, purpose):
    
    if 'bootstrap' in purpose:
        ps = []
        for blindcat in blindcats:
            runname = 'python -W ignore %s'%codename
            runname += ' %i %i %i %s %s &' \
                %(nsplit, nsplits, nobsbin, blindcat, config_file)
            p = sub.Popen(shlex.split(runname))
            p.wait()
    else:
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
    [kids_path, gama_path, Om, Ol, Ok, h, folder, filename, purpose, Rbins, \
     Runit, ncores, lensid_file, lens_weights, lens_binning, lens_selection, \
     src_selection, cat_version, wizz, blindcats] = esd_utils.read_config(config_file)

    print '\n \n \n \n \n \n \n \n \n \n'
    print 'Running:', purpose
    print

    # Define the initial parameters for this shearcode run
    runparams = define_runparams(purpose, lens_binning, ncores, blindcats)
    nruns, nsplits, nsplit, nobsbins, nobsbin, blindcat = runparams
    
    # Excecute the parallelized shearcode run
    run_shearcodes(purpose, nruns, nsplit, nsplits, nobsbin, nobsbins,
                   blindcat, blindcats, config_file)

    end_tot = (time.time()-start_tot)/60
    print 'Finished in: %g minutes' %end_tot
    print

    return
