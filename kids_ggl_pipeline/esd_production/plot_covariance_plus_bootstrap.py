#!/usr/bin/python

"""
# Part of the module to determine the shear
# as a function of radius from a galaxy.
"""

# Import the necessary libraries
import astropy.io.fits as pyfits
import numpy as np
import distance
import sys
import os
import shutil
import time
import shearcode_modules as shear
from astropy import constants as const, units as u

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number


def main(nsplit, nsplits, nobsbin, blindcat, config_file, fn):
    
    # This allows STDIN to work in child processes
    #sys.stdin = os.fdopen(fn)

    # Input parameters
    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
        lens_selection, lens_weights, binname, Nobsbins, src_selection, \
        cat_version, wizz, path_Rbins, name_Rbins, Runit, path_output, path_splits, \
        path_results, purpose, O_matter, O_lambda, Ok, h, filename_addition, Ncat, \
        splitslist, blindcats, blindcat, blindcatnum, \
        path_kidscats, path_gamacat, specz_file, z_epsilon = \
        shear.input_variables(nsplit, nsplits, nobsbin, blindcat, config_file)

    print 'Final step: Plot the ESD profiles and correlation matrix'
    print

    blindcat = blindcat[0]
    
    # Plot settings:

    # Plotting the data for the separate observable bins
    if 'random' in purpose:
        plotstyle = 'lin' # What plotting style is used (lin or log)
    else:
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

    if ('random' in purpose) or ('star' in purpose):
        filename_var = '%i_%s'%(Ncat, filename_var)
        # Ncat is the number of existing randoms
        print 'Number of existing random catalogs:', Ncat

    # Paths to the resulting files
    outname = shear.define_filename_results(path_results, purpose, \
                                            filename_var, filename_addition, \
                                            Nsplit, blindcat)

    # Importing all GAMA data, and the information on
    # radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, \
        Rcenters, nRbins, Rconst, gamacat, galIDlist, galRAlist, galDEClist, \
        galweightlist, galZlist, Dcllist, Dallist = shear.import_data(
            path_Rbins, Runit, path_gamacat, path_kidscats, centering,
            purpose, Ncat, O_matter, O_lambda, Ok, h, lens_weights,
            filename_addition, cat_version)
    
    # Binnning information of the groups
    lenssel_binning = shear.define_lenssel(gamacat, centering,
                                           lens_selection, 'None',
                                           'None', 0, -inf, inf, Dcllist)
    # Mask the galaxies in the shear catalog, 
    # WITHOUT binning (for the bin creation)
    binname, lens_binning, \
    Nobsbins, binmin, binmax = shear.define_obsbins(binnum, lens_binning,
                                                    lenssel_binning, gamacat,
                                                    Dcllist)


    # Writing and showing the plots

    plottitle1 = shear.define_plottitle(purpose, centering, \
                                        lens_selection, binname, \
                                        Nobsbins, src_selection)
    
    if 'bootstrap' not in purpose:
        # Plotting the shear profiles for all observable bins
        for N1 in xrange(Nobsbins):

            binname, lens_binning, \
            Nobsbins, binmin, binmax = shear.define_obsbins(N1+1, lens_binning, \
                                                        lenssel_binning, \
                                                        gamacat, Dcllist)

            filename_N1 = filename_var.replace('binnum', '%i'%(N1+1))
            filenameESD = shear.define_filename_results(path_results, purpose, \
                                                        filename_N1.replace('_bin_%i'%(N1+1), ''), \
                                                        filename_addition, \
                                                        Nsplit, blindcat)
            if 'No' in binname:
                plotlabel = r'ESD$_t$'
            else:
                plotlabel = r'%g $\leq$ %s $\textless$ %g'%(binmin, \
                                            binname.replace('_', ''), binmax)
            
            try:
                shear.define_plot(filenameESD, plotlabel, plottitle1, \
                                  plotstyle, subplots, N1+1, Runit, h)
            except:
                pass
        #try:
        shear.write_plot(filenameESD, plotstyle)
        #except:
            #print "Failed to create ESD Plot of:", filenameESD

    # Creating the ueber-matrix plot
    filename_N1 = filename_var.replace('_binnum', 's')
    filename_cov = filename_var.replace('_binnum', 's')
    filename_cov = filename_cov.replace('_bins', '')
    filenamecov = '%s/%s_matrix_%s.txt'%(path_results, filename_cov, blindcat)
    
    # The Group bins
    if binname == 'No': # If there is no binning
        plottitle2 = ''
    else: # If there is binning
        plottitle2 = r'for %i %s bins between %g and %g.'%(Nobsbins, binname, \
                                            (lens_binning.values()[0])[1][0], \
                                            (lens_binning.values()[0])[1][-1])

    #try:
    shear.plot_covariance_matrix(filenamecov, plottitle1, plottitle2, \
                                     plotstyle_matrix, binname, \
                                     lens_binning, Rbins, Runit, h)
    #except:
    #    print "Failed to create Matrix Plot of", filenamecov
    
    # Addapted the removal of splits. They might be useful for sanity checks.
    # 6.4.2016 - Andrej D.
    
    # Remove the used splits
    if (Nsplit==1) and (blindcat==blindcats[-1]):
        #time.sleep(3) # Wait untill all blinds are done.
        #msg = '\nWarning: do you want to delete split files? [y/n] \n'
        #answer = raw_input(msg)
        #if len(answer) == 0:
        #    exit()
        #if answer.lower() not in ('y', 'yes'):
        #    exit()
        filelist = os.listdir(path_splits)
        
        for filename in filelist:
            #os.remove('%s/%s'%(path_splits, filename))
            try:
                os.remove('%s/%s'%(path_splits, filename))
            except OSError:
                shutil.rmtree('%s/%s'%(path_splits, filename))
    

    return
    

