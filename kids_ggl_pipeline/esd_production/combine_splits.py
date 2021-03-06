#!/usr/bin/python

"""
# Part of the module to determine the shear
# as a function of radius from a galaxy.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import astropy.io.fits as pyfits
import numpy as np
import sys
import os
import time
from astropy import constants as const, units as u
import glob

from . import shearcode_modules as shear

if sys.version_info[0] == 3:
    basestring = str
    xrange = range

# Important constants
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
pix = 0.187 # Used to translate pixel to arcsec
alpha = 0.057 # Used to calculate m
beta = -0.37 # Used to calculate m
inf = np.inf


def main(Nsplit, Nsplits, binnum, blindcat, config_file, fn):

    # Input parameters
    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
        lens_selection, lens_weights, binname, Nobsbins, src_selection, \
        cat_version, path_Rbins, name_Rbins, Runit, path_output, \
        path_splits, path_results, purpose, O_matter, O_lambda, Ok, h, \
        filename_addition, Ncat, splitslist, blindcats, blindcat, \
        blindcatnum, path_kidscats, path_gamacat, colnames, kidscolnames, specz_file, m_corr_file, \
        z_epsilon, n_boot, cross_cov, com, lens_photoz, galSigma, lens_pz_redshift = \
            shear.input_variables(
                Nsplit, Nsplits, binnum, blindcat, config_file)

    
    print('Step 2: Combine splits into one catalogue')
    print()


    if 'bootstrap' in purpose:
        purpose = purpose.replace('bootstrap', 'catalog')
        
        path_catalogs = os.path.join(path_output.rsplit('/',1)[0], 'catalogs')
        path_splits = os.path.join(path_catalogs, 'splits_{0}'.format(purpose))
        path_results = os.path.join(path_catalogs, 'results_{0}'.format(purpose))

    if 'catalog' in purpose:

        binname = 'None'
        lens_binning = {'None': ['self', np.array([])]}
        #if Nsplits < Nobsbins:
        #    Nsplits = Nobsbins
        #    Nsplit = binnum-1

        if centering == 'Cen':
            lens_selection = {'rank{0}'.format(centering): ['self', np.array([1])]}
        else:
            lens_selection = {}
        
        Nobsbins = 1
        binnum = 1


    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(purpose, centering, binname, \
                                             'binnum', Nobsbins, \
                                             lens_selection, lens_binning, src_selection, \
                                             lens_weights, name_Rbins, \
                                             O_matter, O_lambda, Ok, h)

    splitslist = np.array([])

    # Paths to the resulting files
    outname = shear.define_filename_results(path_results, purpose, \
                                            filename_var, filename_addition, \
                                            Nsplit, blindcat)

    # Stop if the output already exists.
    if os.path.isfile(outname):
        print('(in combine_splits)')
        print('This output already exists:', outname)
        print()
        return

    # Load the first split 
    shearcatname = shear.define_filename_splits(path_splits, purpose, \
                                                    filename_var, 1, Nsplits, \
                                                    filename_addition, blindcat)
    shearcat = pyfits.open(shearcatname)
    sheardat = shearcat[1].data

    Rcenters = sheardat['Rcenter'][0]
    Rbins = len(Rcenters)

    columns = sheardat.columns.names # Names of all the columns
    rmcolumns = ['ID', 'Rmin', 'Rmax', 'Rcenter', 'variance(e[A,B,C,D])']
    index = [columns.index(rm) for rm in rmcolumns]
    # Names of the columns that need to be stacked
    columns = np.delete(columns, index)

    combcol = []

    # Adding the lens ID's and the radial bins R
    # This need to be figured out, it is causing pipeline to stall if
    # there is an empty lens list passed.
    if isinstance(sheardat['ID'][0], basestring):
        fmt = '50A'
    elif isinstance(sheardat['ID'][0], int):
        fmt = 'J'
    else:
        fmt = 'E'
    combcol.append(pyfits.Column(name='ID', format=fmt, array=sheardat['ID']))
    print('Combining: ID')

    centers = ['Rmin', 'Rmax', 'Rcenter']

    for r in centers:
        combcol.append(pyfits.Column(name=r, format='%iD'%Rbins, \
                    array=sheardat[r], unit='{0}/h{1:g}'.format(Runit, h*100)))
                                     
        print('Combining:', r)

    # Adding all the columns that need to be stacked

    for col in xrange(len(columns)):
        sumcol = 0
        print('Combining:', columns[col])
        for i in xrange(Nsplits):
            # Reading the shear catalogue
            shearcatname = shear.define_filename_splits(path_splits, \
                                                            purpose, \
                                                            filename_var, i+1, \
                                                            Nsplits, \
                                                            filename_addition, \
                                                            blindcat)
            print('    ', shearcatname)
            shearcat = pyfits.open(shearcatname)
            sheardat = shearcat[1].data
                
            sumcol = sumcol+np.array(sheardat[columns[col]])
        combcol.append(pyfits.Column(name=columns[col], \
                                         format='%iD'%Rbins, array=sumcol))

    # Adding the values of the variances
    combcol.append(pyfits.Column(name='variance(e[A,B,C,D])', format='4D', \
                                 array=sheardat['variance(e[A,B,C,D])']))
    print('Combining: variance(e[A,B,C,D])')
    print()

    # Writing the combined columns to a combined fits file
    cols=pyfits.ColDefs(combcol)
    tbhdu=pyfits.BinTableHDU.from_columns(cols)

    if os.path.isfile(outname):
        os.remove(outname)
        print('Old file "{}" overwritten.'.format(outname))
    else:
        print('New file "{}" written.'.format(outname))

    tbhdu.writeto(outname)

    return

#main()
