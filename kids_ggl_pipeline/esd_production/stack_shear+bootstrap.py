#!/usr/bin/python

"""
# Part of the module to determine the shear
# as a function of radius from a galaxy.
"""

debug = False

# Import the necessary libraries
import astropy.io.fits as pyfits
import numpy as np
import distance
import sys
import os
import time
import shearcode_modules as shear
from astropy import constants as const, units as u

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number


def main():

    # Input parameters
    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
    lens_selection, lens_weights, binname, Nobsbins, src_selection, \
    cat_version, path_Rbins, name_Rbins, Runit, path_output, path_splits, \
    path_results, purpose, O_matter, O_lambda, Ok, h, filename_addition, Ncat, \
    splitslist, blindcats, blindcat, blindcatnum, path_kidscats, \
    path_gamacat = shear.input_variables()


    # Path to the output splits and results
    path_catalogs = '%s/catalogs'%(path_output.rsplit('/',1)[0])
    
    path_splits = '%s/splits_%s'%(path_output, purpose)
    path_results = '%s/results_%s'%(path_output, purpose)
    path_catalog_splits = '%s/splits_%s'%(path_catalogs, purpose)
    path_catalog_results = '%s/results_%s'%(path_catalogs, purpose)


    if 'bootstrap' in purpose:
        print 'Step 3: Stack the lenses and create bootstrap samples'
    else:
        print 'Step 3: Stack the lenses and create the ESD profile'
    print
    
    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(purpose.replace('catalog',''), \
                                             centering, binname, Nobsbins, \
                                             Nobsbins, lens_selection, \
                                             src_selection, lens_weights, \
                                             name_Rbins, O_matter, \
                                             O_lambda, Ok, h)
    if ('random' or 'star') in purpose:
        filename_var = '%i_%s'%(Ncat, filename_var)
        # Ncat is the number of existing randoms
        print 'Number of existing random catalogs:', Ncat

    filenameESD = shear.define_filename_results(path_results, purpose, \
                                                filename_var, \
                                                filename_addition, \
                                                Nsplit, blindcat)
    print 'Requested file:', filenameESD
    
    # Define the list of variables for the input catalog
    filename_var = shear.define_filename_var('shearcatalog', centering, \
                                             'None', -999, -999, \
                                             {'None': np.array([])}, \
                                             src_selection, ['None', ''], \
                                             name_Rbins, O_matter, \
                                             O_lambda, Ok, h)
    if ('random' in purpose):
        filename_var = '%i_%s'%(Ncat, filename_var)
        # Ncat is the number of existing randoms

    # Importing the relevant data from the shear catalog
    shearcatname = shear.define_filename_results(path_catalog_results.replace('bootstrap', 'catalog'), \
                                                 purpose.replace('bootstrap', 'catalog'), \
                                                 filename_var, \
                                                 filename_addition, \
                                                 Nsplit, blindcat)

    sheardat = pyfits.open(shearcatname)[1].data
    print 'Importing:', shearcatname


    # Importing all GAMA data, and the information
    # on radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, \
    Rcenters, nRbins, Rconst, gamacat, galIDlist, galRAlist, galDEClist, \
    galweightlist, galZlist, Dcllist, Dallist = shear.import_data(path_Rbins, \
    Runit, path_gamacat, path_kidscats, centering, \
    purpose.replace('catalog', 'bootstrap'), Ncat, O_matter, O_lambda, Ok, h, \
    lens_weights, filename_addition, cat_version)


    # The bootstrap lens-field matching is used to prevent duplicated lenses.
    
    galIDlist_matched = np.array([], dtype=np.int32)
    for kidscatname in kidscats:
        galIDlist_matched = np.append(galIDlist_matched, catmatch[kidscatname][0])
    galIDlist_matched = np.unique(galIDlist_matched)
    
    # The ID's of the galaxies that lie in this field
    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(purpose.replace('catalog',''), \
                                             centering, binname, 'binnum', \
                                             Nobsbins, lens_selection, \
                                             src_selection, lens_weights, \
                                             name_Rbins, O_matter, \
                                             O_lambda, Ok, h)

    if 'random' in purpose:
        filename_var = '%i_%s'%(Ncat, filename_var)
        # Ncat is the number of existing randoms
        print 'Number of existing random catalogs:', Ncat

    galIDlist = sheardat['ID'] # ID of all galaxies in the shear catalog
    gammatlist = sheardat['gammat_%s'%blindcat] # Shear profile of each galaxy
    gammaxlist = sheardat['gammax_%s'%blindcat] # Cross profile of each galaxy
    wk2list = sheardat['lfweight_%s*k^2'%blindcat] # Weight profile of each galaxy
    w2k2list = sheardat['lfweight_%s^2*k^2'%blindcat] # Squared lensfit weight times squared lensing efficiency
    srcmlist = sheardat['bias_m_%s'%blindcat] # Bias profile of each galaxy
    variance = sheardat['variance(e[A,B,C,D])'][0] # The variance

    # Adding the lens weights
    galweightlist = np.reshape(galweightlist, [len(galIDlist),1])
    [gammatlist, gammaxlist, wk2list, \
     w2k2list, srcmlist] = [gallist*galweightlist for gallist in [gammatlist, \
                                                    gammaxlist, wk2list, \
                                                    w2k2list*galweightlist,\
                                                    srcmlist]]
    
    # Mask the galaxies in the shear catalog, WITHOUT binning
    lenssel_binning = shear.define_lenssel(gamacat, centering, lens_selection, \
                                           'None', 'None', 0, -inf, inf) \
    # Defining the observable binnning range of the groups
    binname, lens_binning, Nobsbins, \
    binmin, binmax = shear.define_obsbins(binnum, lens_binning, \
                                          lenssel_binning, gamacat)


    # Defining the number of bootstrap samples ( = 1 for normal shear stack)
    if 'bootstrap' in purpose:
        Nbootstraps = 1e5
        
        # Selecting the random fields (must be the same for all observable bins)
        # Select Nkidsfields random KiDS fields between 0 and
        # Nkidsfields-1 (Nbootstraps times)
        bootstrap_nums = np.random.random_integers(0,len(kidscats)-1,\
                                                   [Nbootstraps, len(kidscats)])
    else:
        Nbootstraps = 1


    # Masking the list to keep only the specified galaxies
    for binnum in np.arange(Nobsbins)+1:
        
        # Defining the min/max value of each observable bin
        binname, lens_binning, Nobsbins, \
        binmin, binmax = shear.define_obsbins(binnum, lens_binning, \
                                              lenssel_binning, gamacat)

        print
        print '%s-bin %i of %i: %g - %g'%(binname, binnum, Nobsbins, \
                                          binmin, binmax)
        
        # Mask the galaxies in the shear catalog
        lenssel = shear.define_lenssel(gamacat, centering, lens_selection, \
                                       lens_binning, binname, binnum, \
                                       binmin, binmax)

        if debug:
            print 'lenssel:', sum(lenssel)
            print 'galIDlist:', len(galIDlist)

        [galIDs, gammats, gammaxs, \
         wk2s, w2k2s, srcms] = [gallist[lenssel] for gallist in [galIDlist, \
                                                                 gammatlist, \
                                                                 gammaxlist, \
                                                                 wk2list, \
                                                                 w2k2list, \
                                                                 srcmlist]]

        galIDs_matched = galIDs[np.in1d(galIDs, galIDlist_matched)]
        galIDs_matched_infield = galIDs[np.in1d(galIDs, galIDs_infield)]

        print 'Selected:', len(galIDs), 'galaxies,', len(galIDs_matched), 'of which overlap with KiDS.'
        print

        # Paths to the resulting files
        filename_bin = filename_var.replace('binnum', '%s'%(binnum))

        # These arrays will contain the stacked profiles...
        field_shears = np.zeros([len(kidscats), 5, nRbins])
        outputnames = ['ESDt', 'ESDx', 'ESD(error)', 'bias']

        # Stack one shearprofile per KiDS field
        for k in xrange(len(kidscats)):    
            # Mask all objects that are not in this field

            matched_galIDs = np.array(catmatch[kidscats[k]])[0]
            
            field_mask = np.in1d(galIDs, matched_galIDs) # Define the mask
            [galID, gammat, gammax, \
             wk2, w2k2, srcm] = [gallist[field_mask] for gallist in [galIDs, \
                                                                     gammats, \
                                                                     gammaxs, \
                                                                     wk2s, \
                                                                     w2k2s, \
                                                                     srcms]]

            if len(gammat) > 0:
                # If there are lenses in this field...
                # Add the field to the ESD-profile table
                field_shears[k] = np.array([sum(gammat,0), sum(gammax,0), \
                                            sum(wk2,0), sum(w2k2,0), \
                                            sum(srcm,0)])
            
        # Taking the bootstrap samples
        if 'bootstrap' in purpose:
            print 'Number of bootstraps: %g'%Nbootstraps
            # Randomly pick a number of fields equal to
            # the total number of fields and sum them
            shear_bootstrap = np.sum(field_shears[bootstrap_nums], 1)
            
            
            # The summed quantities
            gammat, gammax, wk2, w2k2, srcm = [shear_bootstrap[:, x] \
                                               for x in xrange(5)]
            # Write the output to the bootstrap sample table
            output_bootstrap = np.array(shear.calc_stack(gammat, gammax, \
                                                         wk2, w2k2, srcm, \
                                                         variance, blindcatnum))
            
            splitname = shear.define_filename_splits(path_splits, purpose, \
                                                     filename_bin, 0, 0, \
                                                     filename_addition, \
                                                     blindcat)
            
            # Write the results to a bootstrap catalog
            e1 = e2 = w = srcm = []
            shear.write_catalog(splitname, np.arange(Nbootstraps), Rbins, \
                                Rcenters, nRbins, Rconst, output_bootstrap, \
                                outputnames, variance, purpose, e1, e2, w, srcm)

            error_tot = np.zeros(nRbins)
            for r in xrange(nRbins):
                # Mask values that do not have finite values (inf or nan)
                bootstrap_mask = (0 < wk2[:, r]) & (wk2[:, r] < inf)
                
                if sum(bootstrap_mask) > 0:
                    # If there are any values that are not masked
                    error_tot[r] = (np.var((output_bootstrap[0,:,r])\
                                           [bootstrap_mask], 0))**0.5

        # Calculating the normal shear profile

        # Sum all fields
        shear_sample = np.array(np.sum(field_shears, 0))

        # The summed quantities
        gammat, gammax, wk2, w2k2, srcm = [shear_sample[x] for x in xrange(5)]
        if debug:
            print 'gammat:', gammat/1.e6

         # Calculate the stacked final output
        output = np.array(shear.calc_stack(gammat, gammax, wk2, w2k2, srcm, \
                                           variance, blindcatnum))

        ESDt_tot, ESDx_tot, error_poisson, \
        bias_tot = [output[x] for x in xrange(len(outputnames))]

        if 'bootstrap' not in purpose:
            error_tot = error_poisson


        # Printing the ESD profile to a file
        
        # Path to the output plot and text files
        stackname = shear.define_filename_results(path_results, purpose, \
                                                  filename_bin, \
                                                  filename_addition, Nsplit, \
                                                  blindcat)

        # Printing stacked shear profile to a txt file
        shear.write_stack(stackname, Rcenters, Runit, ESDt_tot, ESDx_tot, \
                          error_tot, bias_tot, h, variance, blindcatnum, \
                          galIDs_matched, galIDs_matched_infield)
        

        # Plotting the data for the separate observable bins
        plottitle = shear.define_plottitle(purpose, centering, lens_selection, \
                                           binname, Nobsbins, src_selection)
        
        # What plotting style is used (lin or log)
        if 'random' in purpose:
            plotstyle = 'lin'
        else:
            plotstyle = 'log'
        
        if 'No' in binname:
            plotlabel = r'ESD$_t$'
        else:
            plotlabel = r'%.3g $\leq$ %s $\textless$ %.3g (%i lenses)'%(binmin,\
                        binname.replace('_', ''), binmax, len(galIDs_matched))
        
        try:
            shear.define_plot(stackname, plotlabel, plottitle, plotstyle, \
                              Nobsbins, binnum, Runit, h)
        except:
            pass
        
    # Writing and showing the plot
    try:
        shear.write_plot(stackname, plotstyle)
    except:
        print 'Failed to write ESD plot for:', stackname 
    
    return

main()
