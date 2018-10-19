#!/usr/bin/python

"""
# Part of the module to determine the shear
# as a function of radius from a galaxy.
"""
# Import the necessary libraries
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import astropy.io.fits as pyfits
import numpy as np
import sys
import os
import shutil
import time
from astropy import constants as const, units as u

from . import shearcode_modules as shear

if sys.version_info[0] == 3:
    xrange = range

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number


def main(nsplit, nsplits, nobsbin, blindcat, config_file, fn):
    
    # This allows STDIN to work in child processes
    #sys.stdin = os.fdopen(fn)

    # Input parameters
    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
        lens_selection, lens_weights, binname, Nobsbins, src_selection, \
        cat_version, path_Rbins, name_Rbins, Runit, path_output, \
        path_splits, path_results, purpose, O_matter, O_lambda, Ok, h, \
        filename_addition, Ncat, splitslist, blindcats, blindcat, \
        blindcatnum, path_kidscats, path_gamacat, colnames, kidscolnames, specz_file, m_corr_file,\
        z_epsilon, n_boot, cross_cov, com = \
            shear.input_variables(
                nsplit, nsplits, nobsbin, blindcat, config_file)

    print('Final step: Plot the ESD profiles and correlation matrix')
    print()

    blindcat = blindcat[0]
    
    # Plot settings:

    # Plotting the data for the separate observable bins
    
    plotstyle = 'log'
    subplots = binnum # Are there subplots?
    Nrows = 1 # If so, how into many rows will the subplots be devided?

    # Creating the ueber-matrix plot (covlin, covlog, corlin, corlog)
    plotstyle_matrix = 'corlin'

    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(purpose, centering, binname, \
                                             'binnum', Nobsbins, \
                                             lens_selection, lens_binning, src_selection, \
                                             lens_weights, name_Rbins, \
                                             O_matter, O_lambda, Ok, h)

    # Paths to the resulting files
    outname = shear.define_filename_results(path_results, purpose, \
                                            filename_var, filename_addition, \
                                            Nsplit, blindcat)

    # Importing all GAMA data, and the information on
    # radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, \
        Rcenters, nRbins, Rconst, gamacat, galIDlist, galRAlist, galDEClist, \
        galweightlist, galZlist, Dcllist, Dallist = shear.import_data(
            path_Rbins, Runit, path_gamacat, colnames, kidscolnames, path_kidscats,
            centering, purpose, Ncat, O_matter, O_lambda, Ok, h, lens_weights,
            filename_addition, cat_version, com)
    
    # Binnning information of the groups
    lenssel_binning = shear.define_lenssel(
        gamacat, colnames, centering, lens_selection, 'None', 'None',
        0, -inf, inf, Dcllist, galZlist, h)
    # Mask the galaxies in the shear catalog,
    # WITHOUT binning (for the bin creation)
    binname, lens_binning, Nobsbins, binmin, binmax = \
        shear.define_obsbins(
            binnum, lens_binning, lenssel_binning, gamacat, Dcllist, galZlist)


    # Writing and showing the plots

    plottitle1 = shear.define_plottitle(
        purpose, centering, lens_selection, binname, Nobsbins, src_selection)

    if 'bootstrap' not in purpose:
        # Plotting the shear profiles for all observable bins
        for N1 in xrange(Nobsbins):

            binname, lens_binning, Nobsbins, binmin, binmax = \
                shear.define_obsbins(
                    N1+1, lens_binning, lenssel_binning, gamacat, Dcllist,
                    galZlist)

            filename_var_N1 = filename_var.replace('binnum', '%i'%(N1+1))
            filename_N1 = shear.define_filename_results(
                path_results, purpose, filename_var_N1, filename_addition,
                Nsplit, blindcat)
            filenameESD = shear.define_filename_results(
                path_results, purpose, 
                filename_var_N1.replace('_bin_%i'%(N1+1), ''),
                filename_addition, Nsplit, blindcat)
            if 'No' in binname:
                plotlabel = r'ESD$_t$'
            else:
                plotlabel = r'%g $\leq$ %s $\textless$ %g'%(binmin, \
                                            binname.replace('_', ''), binmax)
            
            #try:
            shear.define_plot(filename_N1, plotlabel, plottitle1, \
                                  plotstyle, subplots, N1+1, Runit, h)
            #except:
            #    pass
        #try:
        shear.write_plot(filenameESD, plotstyle)
        #except:
            #print "Failed to create ESD Plot of:", filenameESD

    # Creating the ueber-matrix plot
    filename_cov = filename_var.replace('_binnum', 's')
    filename_cov = filename_cov.replace('_bins', '')
    filenamecov = '{0}/{1}_matrix_{2}.txt'.format(
        path_results, filename_cov, blindcat)
    
    # the group bins
    if binname == 'No': # If there is no binning
        plottitle2 = ''
    else: # If there is binning
        plottitle2 = r'for {0} {1} bins between {2} and {3}.'.format(
            Nobsbins, binname.replace('_', '\\_'),
            list(lens_binning.values())[0][1][0],
            list(lens_binning.values())[0][1][-1])
    #try:
    shear.plot_covariance_matrix(
        filenamecov, plottitle1, plottitle2, plotstyle_matrix, binname,
        lens_binning, Rbins, Runit, h)
    #except:
    #    print "Failed to create Matrix Plot of", filenamecov

    # Remove the used splits
    # CS: path_splits doesn't exist when I run it - why?
    if (Nsplit==1) and (blindcat==blindcats[-1]) and os.path.isdir(path_splits):
        filelist = os.listdir(path_splits)
        for filename in filelist:
            try:
                #os.remove('%s/%s'%(path_splits, filename))
                os.remove(os.path.join(path_splits, filename))
            except OSError:
                #shutil.rmtree('%s/%s'%(path_splits, filename))
                shutil.rmtree(os.path.join(path_splits, filename))

    return


