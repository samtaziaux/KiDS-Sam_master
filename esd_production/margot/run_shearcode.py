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
import argparse
from decimal import *

# local
import esd_utils

start_tot = time.time()

# Important constants
inf = np.inf

# Taking the input parameters
#parser = argparse.ArgumentParser()

#parser.add_argument('-c', '--config', dest='config_file')

"""
parser.add_argument('--purpose', action='store', dest='purpose', choices=['shearcatalog', 'randomcatalog', 'starcatalog', 'covariance', 'shearbootstrap', 'randombootstrap', 'binlimits'], default='shearcatalog', type=str, help='The purpose of this run determines the final output that will be generated (default = shearcatalog).')
parser.add_argument('--Ncores', action='store', dest='Ncores', default=3, type=int, help='The number of cores over which this calculation will be spread (default = 1).')
parser.add_argument('--center', action='store', dest='center', choices=['Cen', 'IterCen', 'BCG'], default='BCG', type=str, help='The center definition (default = BCG).')
parser.add_argument('--rank', action=('store'), dest='rank',  nargs=2, default=[-999, inf], type=float, help='The minimum and maximum rank of galaxies (rankmin <= galaxy rank <= rankmax, default = [-999, inf]).')
parser.add_argument('--Nfof', action=('store'), dest='Nfof',  nargs=2, default=[2, inf], type=float, help='(Group members only) The minimum and maximum number of members in the group (Nfofmin <= Nfof <= Nfofmax, default = [2, inf]).')
parser.add_argument('--obsbins', action='store', dest='obsbins', default='No', type=str, help='The name of the binning observable in the GAMA catalog (for no binning, use: No (default)).')
parser.add_argument('--path_obsbins', action='store', dest='path_obsbins', default='No', type=str, help='The text file containing the observable bin limits, or the number of observable bins (obsmin <= obs < obsmax, default = No).')
parser.add_argument('--obslim', action='store', dest='obslim', default='No', type=str, help='The name of the observable limit on the lenses (for no limit, use: No (default)).')
parser.add_argument('--val_obslim', action=('store'), dest='val_obslim', nargs=2, default=[-inf, inf], type=str, help='The values of the minimum and maximum observable limit on the lenses (obslim[min] <= obslim <= obslim[max], default = [-inf, inf]).')
parser.add_argument('--zB', action=('store'), dest='ZB',  nargs=2, default=[0.005, 1.2], type=float, help='The minimum and maximum zB (best redshift) of the sources (zBmin <= zB <= zBmax, default = [0.005, 1.2]).')
parser.add_argument('--O_matter', action='store', dest='O_matter', default=0.315, type=float, help='The value of Omega(matter) (Flat universe: Omega(Lambda) = 1 - Omega(matter), default = 0.315).')
parser.add_argument('--h0', action='store', dest='h0', default='1.0', type=float, help='The value of the reduced Hubble constant h0 (H0 = h0*100km/s/Mpc, default = 1.0).')
parser.add_argument('--path_Rbins', action='store', dest='path_Rbins', default='10:20:2000', type=str, help='The number:minimum:maximum (divided by ":") of logarithmic radial bins, or the text file containing the radial bin limits (default = 10:20:2000).')
parser.add_argument('--path_output', action='store', dest='path_output', default='/disks/shear10/brouwer_veersemeer/shearcode_output', type=str, help='The directory where the output will be written (default = /disks/shear10/brouwer_veersemeer/shearcode_output).')
parser.add_argument('--filename_addition', action='store', dest='filename_addition', default='No', type=str, help='Any additional information you would like to add to the file name (for nothing, use: No (default)).')
parser.add_argument('--path_kidscats', action='store', dest='path_kidscats', default='/data2/brouwer/KidsCatalogues/LF_cat_DR2_v2', type=str, help='The directory of the KiDS catalogues (default = /data2/brouwer/KidsCatalogues/LF_cat_DR2_v2.')

# Old kidscats directory: /disks/shear10/Catalogues/KiDS/ManMask

args = parser.parse_args()

# Converting the input parameters to those readible by the codes
purpose = args.purpose
Ncores = args.Ncores
center = args.center
rankmin = args.rank[0]
rankmax = args.rank[1]
Nfofmin = args.Nfof[0]
Nfofmax = args.Nfof[1]
ZBmin = args.ZB[0]
ZBmax = args.ZB[1]
name_obsbins = args.obsbins
path_obsbins = args.path_obsbins
name_obslim = args.obslim
val_obslim = args.val_obslim
obslim_min = args.val_obslim[0]
obslim_max = args.val_obslim[1]
path_Rbins = args.path_Rbins
path_output = args.path_output
O_matter = args.O_matter
h0 = args.h0
filename_addition = args.filename_addition
path_kidscats = args.path_kidscats

print '\n \n \n \n \n \n \n \n \n \n'

print 'Running:', purpose

# Check if Nfof is not used for field galaxies
if (rankmin < 1) or (rankmax < 1):
	if (Nfofmin != 2) or (Nfofmax != inf):
		print '*** Error (rank & Nfof): Only group members can be selected according to group multiplicity ***'
		quit()

# Check if both the name and the limits of the observable bins are given
if name_obsbins != 'No' or path_obsbins != 'No':
	if name_obsbins == 'No' or path_obsbins == 'No':
		print '*** Error (obsbins & path_obsbins): Specify both the name and the limits of the observable bins ***'
		quit()
if name_obslim != 'No' or val_obslim != [-inf, inf]:
	if name_obslim == 'No' or val_obslim == [-inf, inf]:
		print '*** Error (obslim & val_obslim): Specify both the name and the limits of the observable ***'
		quit()

# Centrals with the 'Cen' definition can only be group centers
if center == 'Cen':
	rankmin = rankmax = 1
	print '*** Warning (center): "Cen" center does not correspond to a galaxy: only Group centers available ***'

if 'star' in purpose:
	rankmin = -999
	rankmax = inf
	Nfofmin = 2
	Nfofmax = inf

# Binnning information of the groups
if name_obsbins == 'No': # There is no binning
	Nobsbins = 1
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

"""

def run(config_file):
    # Default values
    info = esd_utils.read_config(config_file)
    # Input for the codes
    inputlist = (center, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax,
                name_obsbins, path_obsbins, Nobsbins, name_obslim, obslim_min,
                obslim_max, path_Rbins, path_output, purpose, O_matter, h0,
                filename_addition, path_kidscats)
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

