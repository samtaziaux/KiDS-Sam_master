#!/usr/bin/python

"""
# Part of the module to determine the shear
# as a function of radius from a galaxy.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Import the necessary libraries
import astropy.io.fits as pyfits
import collections
import numpy as np
import os
import sys
import time
from astropy import constants as const, units as u
if sys.version_info[0] == 2:
    range = xrange

from . import shearcode_modules as shear

debug = False
# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number


def main(nsplit, nsplits, nobsbin, blindcat, config_file, fn):

    # Input parameters
    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
        lens_selection, lens_weights, binname, Nobsbins, src_selection, \
        cat_version, path_Rbins, name_Rbins, Runit, path_output, path_splits, \
        path_results, purpose, O_matter, O_lambda, Ok, h, filename_addition, Ncat, \
        splitslist, blindcats, blindcat, blindcatnum, path_kidscats, \
        path_gamacat, colnames, kidscolnames, specz_file, m_corr_file, z_epsilon, n_boot, cross_cov, lens_photoz, galSigma, lens_pz_redshift, com = \
            shear.input_variables(
                nsplit, nsplits, nobsbin, blindcat, config_file)

    # blindcat is the current catalog but it arrives here as a list with len=1
    blindcat = blindcat[0]

    # Path to the output splits and results
    path_catalogs = '{}/catalogs'.format(path_output.rsplit('/',1)[0])
    path_splits = '{}/splits_{}'.format(path_output, purpose)
    path_results = path_output #'%s/results_%s'%(path_output, purpose)
    path_catalog_splits = '{}/splits_{}'.format(path_catalogs, purpose)
    path_catalog_results = '{}/results_{}'.format(path_catalogs, purpose)

    if 'bootstrap' in purpose:
        print('Step 3: Stack the lenses and create bootstrap samples')
    else:
        print('Step 3: Stack the lenses and create the ESD profile')
    print()

    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(
        purpose.replace('catalog',''), centering, binname, Nobsbins,
        Nobsbins, lens_selection, lens_binning, src_selection, lens_weights,
        name_Rbins, O_matter, O_lambda, Ok, h)

    filenameESD = shear.define_filename_results(
        path_results, purpose, filename_var, filename_addition,
        Nsplit, blindcat)
    print('Requested file:', filenameESD)

    # Define the list of variables for the input catalog
    filename_var = shear.define_filename_var(
        'shearcatalog', centering, 'None', -999, -999, {'None': np.array([])},
        {'None': np.array([])}, src_selection, ['None', ''], name_Rbins,
        O_matter, O_lambda, Ok, h)

    # Importing the relevant data from the shear catalog
    shearcatname = shear.define_filename_results(
        path_catalog_results.replace('bootstrap', 'catalog'),
        purpose.replace('bootstrap', 'catalog'),
        filename_var, filename_addition, Nsplit, blindcat)

    sheardat = pyfits.open(shearcatname)[1].data
    print('Importing:', shearcatname)

    # Importing all GAMA data, and the information
    # on radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, \
    Rcenters, nRbins, Rconst, gamacat, galIDlist, galRAlist, galDEClist, \
    galweightlist, galZlist, Dcllist, Dallist = \
        shear.import_data(
            path_Rbins, Runit, path_gamacat, colnames, kidscolnames, path_kidscats,
            centering, purpose.replace('catalog', 'bootstrap'), Ncat,
            O_matter, O_lambda, Ok, h, lens_weights, filename_addition,
            cat_version, com)

    # The bootstrap lens-field matching is used to prevent duplicated lenses.
    galIDlist_matched = np.array([], dtype=np.int32)
    for kidscatname in kidscats:
        galIDlist_matched = np.append(
            galIDlist_matched, catmatch[kidscatname][0])
    galIDlist_matched = np.unique(galIDlist_matched)

    # The ID's of the galaxies that lie in this field
    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(
        purpose.replace('catalog',''), centering, binname, 'binnum', Nobsbins,
        lens_selection, lens_binning, src_selection, lens_weights, name_Rbins,
        O_matter, O_lambda, Ok, h)

    # ID of all galaxies in the shear catalog
    galIDlist = sheardat['ID']
    # Shear profile of each galaxy
    gammatlist = np.nan_to_num(sheardat['gammat_{}'.format(blindcat)])
    # Cross profile of each galaxy
    gammaxlist = np.nan_to_num(sheardat['gammax_{}'.format(blindcat)])
    # Weight profile of each galaxy
    wk2list = np.nan_to_num(sheardat['lfweight_{}*k^2'.format(blindcat)])
    # Squared lensfit weight times squared lensing efficiency
    w2k2list = np.nan_to_num(sheardat['lfweight_{}^2*k^2'.format(blindcat)])
    # Bias profile of each galaxy
    srcmlist = np.nan_to_num(sheardat['bias_m_{}'.format(blindcat)])
    # The distance R of the sources
    Rsrclist = np.nan_to_num(sheardat['Rsources_{}'.format(blindcat)])
    # The variance
    variance = sheardat['variance(e[A,B,C,D])'][0]
    Nsrclist = sheardat['Nsources']


    # Adding the lens weights
    galweightlist = np.reshape(galweightlist, [len(galIDlist),1])
    gammatlist, gammaxlist, wk2list, w2k2list, srcmlist, Rsrclist, Nsrclist = \
        [gallist*galweightlist for gallist
         in [gammatlist, gammaxlist, wk2list, w2k2list*galweightlist,
             srcmlist, Rsrclist, Nsrclist]]

    # Mask the galaxies in the shear catalog, WITHOUT binning
    lenssel_binning = shear.define_lenssel(
        gamacat, colnames, centering, lens_selection, 'None', 'None',
        0, -inf, inf, Dcllist, galZlist,  h)
    # Defining the observable binnning range of the groups
    binname, lens_binning, Nobsbins, \
    binmin, binmax = shear.define_obsbins(
        binnum, lens_binning, lenssel_binning, gamacat)

    # Defining the number of bootstrap samples ( = 1 for normal shear stack)
    if 'bootstrap' in purpose:
        Nbootstraps = 1e5
        if n_boot == 1:
            print('Using 1x1 KiDS tiles for bootstrapping purposes.')
            # Selecting the random fields (must be the same for all observable bins)
            # Select Nkidsfields random KiDS fields between 0 and
            # Nkidsfields-1 (Nbootstraps times)
            bootstrap_nums = np.random.random_integers(
                0,len(kidscats)-1, [np.int(Nbootstraps), len(kidscats)])

        if n_boot != 1:
            print('Using {0} KiDS tiles for bootstrapping purposes.'.format(n_boot))

            matched_i = {}
            coords = np.array([])
            for i, kidscat in enumerate(kidscats):
                matched_i[(
                    np.float64(catmatch[kidscat][1].split('_')[1].replace('m', '-').replace('p', '.')),
                    np.float64(catmatch[kidscat][1].split('_')[2].replace('m', '-').replace('p', '.')))] \
                        = kidscat, i
                coords = np.append(
                    coords,
                    np.array([np.float64(catmatch[kidscat][1].split('_')[1].replace('m', '-').replace('p', '.')),
                    np.float64(catmatch[kidscat][1].split('_')[2].replace('m', '-').replace('p', '.'))]))

            matched_i = collections.OrderedDict(sorted(matched_i.items()))
            matched = matched_i.copy()
            coords = coords.reshape(-1, 2)
            n_patches = 4
            result = []
            for m in matched.keys():
                x_coord, y_coord = m
                result_i = []
                plot_i = np.array([])
                for x, y in [(x_coord+i,y_coord+j) \
                        for i in (0,1) for j in (0,1)]:
                    if (x,y) in matched_i:
                        result_i.append(matched_i[(x,y)][1])
                        plot_i = np.append(plot_i, np.array([x,y]))
                        if len(result_i) == n_patches:
                            del matched_i[(x,y)]
                result.append(result_i)

            result_final = [x for x in result if x != []]
            result = np.array(
                [np.array(xi) for xi in result_final if len(xi)==n_patches])
            rands = np.random.random_integers(
                0, len(result)-1, [np.int(Nbootstraps),len(result)]).flatten()
            bootstrap_nums2 = result[rands]
            bootstrap_nums2 = np.concatenate(bootstrap_nums2).ravel()
            bootstrap_nums = bootstrap_nums2.reshape((np.int(Nbootstraps),-1))

    else:
        Nbootstraps = 1


    # Masking the list to keep only the specified galaxies
    for binnum in np.arange(Nobsbins)+1:
        # Defining the min/max value of each observable bin
        binname, lens_binning, Nobsbins, \
        binmin, binmax = shear.define_obsbins(
            binnum, lens_binning, lenssel_binning, gamacat)
        print()
        print('{}-bin {} of {}: {} - {}'.format(
                binname, binnum, Nobsbins, binmin, binmax))
        # Mask the galaxies in the shear catalog
        lenssel = shear.define_lenssel(
            gamacat, colnames,  centering, lens_selection, lens_binning,
            binname, binnum, binmin, binmax, Dcllist, galZlist, h)

        if debug:
            print('lenssel:', sum(lenssel))
            print('galIDlist:', len(galIDlist))

        [galIDs, gammats, gammaxs, wk2s, w2k2s, srcms, Rsrcs, Nsrcs] \
            = [gallist[lenssel]
               for gallist in [galIDlist, gammatlist, gammaxlist, wk2list,
                               w2k2list, srcmlist, Rsrclist, Nsrclist]]
        galIDs_matched = galIDs[np.in1d(galIDs, galIDlist_matched)]
        galIDs_matched_infield = galIDs[np.in1d(galIDs, galIDs_infield)]

        print('Selected:', len(galIDs), 'galaxies,', len(galIDs_matched), \
                'of which overlap with KiDS.')
        print()

        # Paths to the resulting files
        filename_bin = filename_var.replace('binnum', str(binnum))
        # These arrays will contain the stacked profiles...
        field_shears = np.zeros([len(kidscats), 7, nRbins])
        outputnames = ['ESDt', 'ESDx', 'ESD(error)', 'bias', 'Rsource']

        # Stack one shearprofile per KiDS field
        for k in range(len(kidscats)):
            # Mask all objects that are not in this field
            matched_galIDs = np.array(catmatch[kidscats[k]])[0]
            field_mask = np.in1d(galIDs, matched_galIDs) # Define the mask
            [galID, gammat, gammax, wk2, w2k2, srcm, Rsrc, Nsrc] \
                = [gallist[field_mask]
                   for gallist in [galIDs, gammats, gammaxs, wk2s, w2k2s,
                                   srcms, Rsrcs, Nsrcs]]
            # If there are lenses in this field...
            if len(gammat) > 0:
                # Add the field to the ESD-profile table
                field_shears[k] = np.array(
                    [sum(gammat,0), sum(gammax,0), sum(wk2,0), sum(w2k2,0),
                     sum(srcm,0), sum(Rsrc,0), sum(Nsrc,0)])

        # Taking the bootstrap samples
        if 'bootstrap' in purpose:
            print('Number of bootstraps: {}'.format(Nbootstraps))
            # Randomly pick a number of fields equal to
            # the total number of fields and sum them
            shear_bootstrap = np.sum(field_shears[bootstrap_nums], 1)
            # The summed quantities
            gammat, gammax, wk2, w2k2, srcm, Rsrc, Nsrc \
                = [shear_bootstrap[:, x] for x in range(7)]
            # Write the output to the bootstrap sample table
            output_bootstrap = np.array(shear.calc_stack(
                gammat, gammax, wk2, w2k2, srcm, Rsrc, variance, blindcatnum))

            splitname = shear.define_filename_splits(
                path_splits, purpose, filename_bin, 0, 0, filename_addition,
                blindcat)

            # Write the results to a bootstrap catalog
            e1 = e2 = w = srcm = Rsrc = []
            shear.write_catalog(
                splitname, np.arange(Nbootstraps), Rbins, Rcenters, nRbins,
                Rconst, output_bootstrap, outputnames, variance, purpose, e1,
                e2, w, srcm, blindcats)

            error_tot = np.zeros(nRbins)
            for r in range(nRbins):
                # Mask values that do not have finite values (inf or nan)
                bootstrap_mask = (0 < wk2[:, r]) & (wk2[:, r] < inf)
                if sum(bootstrap_mask) > 0:
                    # If there are any values that are not masked
                    error_tot[r] = (np.var(
                        (output_bootstrap[0,:,r])[bootstrap_mask], 0))**0.5

        # Calculating the normal shear profile

        # Sum all fields
        shear_sample = np.array(np.sum(field_shears, 0))
        # The summed quantities
        gammat, gammax, wk2, w2k2, srcm, Rsrc, Nsrc \
            = [shear_sample[x] for x in range(7)]
        if debug:
            print('gammat:', gammat/1.e6)

         # Calculate the stacked final output
        output = np.array(shear.calc_stack(
            gammat, gammax, wk2, w2k2, srcm, Rsrc, variance, blindcatnum))
            
        ESDt_tot, ESDx_tot, error_poisson, bias_tot, Rsrc_tot \
            = [output[x] for x in range(len(outputnames))]

        if 'bootstrap' not in purpose:
            error_tot = error_poisson

        ## Printing the ESD profile to a file
        # Path to the output plot and text files
        stackname = shear.define_filename_results(
            path_results, purpose, filename_bin, filename_addition, Nsplit,
            blindcat)

        # Printing stacked shear profile to a txt file
        shear.write_stack(
            stackname, filename_var, Rcenters, Runit, ESDt_tot, ESDx_tot,
            error_tot, bias_tot, h, variance, wk2, w2k2, Nsrc, Rsrc_tot, blindcat,
            blindcats, blindcatnum, galIDs_matched, galIDs_matched_infield)

        # Plotting the data for the separate observable bins
        plottitle = shear.define_plottitle(
            purpose, centering, lens_selection, binname, Nobsbins,
            src_selection)

        # What plotting style is used (lin or log)
        plotstyle = 'log'
        if 'No' in binname:
            plotlabel = r'ESD$_t$'
        else:
            plotlabel = r'%.3g $\leq$ %s $\textless$ %.3g (%i lenses)' \
                %(binmin, binname.replace('_', ''), binmax,
                  len(galIDs_matched))
        try:
            shear.define_plot(stackname, plotlabel, plottitle, plotstyle,
                              Nobsbins, binnum, Runit, h)
        except:
            pass

    # Writing and showing the plot
    #try:
    #shear.write_plot(stackname.replace('_bin_%s'%(binnum), ''), plotstyle)
    #except:
        #print 'Failed to write ESD plot for:', stackname 
    
    return

