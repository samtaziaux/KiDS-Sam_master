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
from decimal import *

# local
import esd_utils

start_tot = time.time()

# Important constants
inf = np.inf

# Taking the input parameters

config_file = '/data2/brouwer/shearprofile/KiDS-GGL/ggl.config'

def import_config(config_filename):
    # Default values
    info = esd_utils.read_config(config_file)

    # Input for the codes
	[purpose, folder, filename, kidsversion, gamaversion,
    ncores, Rbins, src_selection,
    lens_selection, group_centre, lensid_file, binparam,
    Om, Ol, Ok, h] = [info[i] for i in xrange(len(info))]

def main():
	
	print '\n \n \n \n \n \n \n \n \n \n'

	print 'Running:', purpose

	# Binnning information of the groups
    binparam = lens_selection.keys()[0]
	obsbins = lens_selection[binparam].split(',')
	Nobsbins = len(obsbins)

	# What if there is no observable binning?

	else: # If there is binning...
		if os.path.isfile(path_obsbins): # from a file
			Nobsbins = len(np.loadtxt(path_obsbins).T)-1
		else:
			try:
				Nobsbins = int(path_obsbins) # from a specified number (of bins)
			except:
				print 'Observable bin file does not exist:', path_obsbins
				quit()
		print '%s bins: %s'%(name_obsbins, Nobsbins)


	# The number of catalogues that will be ran
	if ('random' in purpose):
		Nruns = 0
	else:
		Nruns = 1


	# Prepare the values for Ncores and Nobsbins
	Ncores = Ncores/Nobsbins
	if Ncores == 0:
		Ncores = 1


	# Names of the blind catalogs
	#blindcats = ['D']


    nsplit,blindcat binnum


    
    inputstring = '%s %g %g %g %g %g %g %s %s %i'
    inputstring += '%s %s %s %s %s %s %g %g %s %s'
    inputstring = inputstring %inputlist

    blindcats = ['A', 'B', 'C', 'D']
    blindcat = blindcats[0]
    binnum = Nobsbins
    Ncore = 1

    if purpose == 'binlimits':
        # Define observable bin limits with equal weight
        selname = 'python define_obsbins_equal_sn.py %i %i %i %s %s &'\
                %(Ncore, Ncores, binnum, blindcat, inputstring)
        p = sub.Popen(shlex.split(selname))
        p.wait()
    else:
        # The calculation starts here:
        # Creating the splits
        for n in xrange(Nruns):
            ps = []
            for binnum in np.arange(Nobsbins)+1:
                for Ncore in np.arange(Ncores)+1:
                    splitsname = 'python -W ignore shear+covariance.py'
                    splitsname += '%i %i %i %s %s     &' \
                                %(Ncore, Ncores, binnum, blindcat, inputstring)
                    p = sub.Popen(shlex.split(splitsname))
                    ps.append(p)
            for p in ps:
                p.wait()

    # Combining the splits to a single output
    if ('bootstrap' in purpose) or ('catalog' in purpose):

        ps = []
        combname = 'python -W ignore combine_splits.py %i %i %i %s %s &' \
                   %(Ncore, Ncores, binnum, blindcat, inputstring)
        runcomb =  sub.Popen(shlex.split(combname))
        ps.append(runcomb)
        for p in ps:
            p.wait()


    if ('bootstrap' in purpose) or ('catalog' in purpose):
        ps = []
        for blindcat in blindcats:
            plotname = 'python -W ignore stack_shear+bootstrap.py'
            plotname += '%i %i %i %s %s     &' \
                        %(Ncore, Ncores, binnum, blindcat, inputstring)
            runplot = sub.Popen(shlex.split(plotname))
            ps.append(runplot)
        for p in ps:
            p.wait()


    if ('bootstrap' in purpose) or ('covariance' in purpose):

        ps = []
        for blindcat in blindcats:
            combname = 'python -W ignore combine_covariance+bootstrap.py'
            combname += '%i %i %i %s %s &' \
                        %(Ncore, Ncores, binnum, blindcat, inputstring)
            runcomb = sub.Popen(shlex.split(combname))
            ps.append(runcomb)
        for p in ps:
            p.wait()


    if ('bootstrap' in purpose) or ('covariance' in purpose):
        ps = []
        for blindcat in blindcats:
            combname = 'python -W ignore plot_covariance+bootstrap.py'
            combname = '%i %i %i %s %s     &' \
                       %(Ncore, Ncores, binnum, blindcat, inputstring)
            runcomb = sub.Popen(shlex.split(combname))
            ps.append(runcomb)
        for p in ps:
            p.wait()


    end_tot = (time.time()-start_tot)/60
    print 'Finished in: %g minutes' %end_tot
    print
    return

main()

