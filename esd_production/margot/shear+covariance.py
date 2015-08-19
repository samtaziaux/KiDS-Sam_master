#!/usr/bin/python

"Part of the module to determine the shear as a function of radius from a galaxy."

# Import the necessary libraries
import pyfits
import numpy as np
import distance
import sys
import os
import time

from astropy import constants as const, units as u
import numpy.core._dotblas
import memory_test as memory
import time
import gc

import shearcode_modules as shear


# Important constants
G = const.G.to('pc3/Msun s2') # Gravitational constant
c = const.c.to('pc/s') # Speed of light
inf = np.inf # Infinity
nan = np.nan # Not a number


def main():

    start_tot = time.time()

    Nsplit, Nsplits, centering, ranks, lensid_file, lens_binning, binnum, \
            lens_selection, binname, Nobsbins, src_selection, path_Rbins, name_Rbins, path_output, \
            path_splits, path_results, purpose, O_matter, O_lambda, Ok, h, \
            filename_addition, Ncat, splitslist, blindcats, blindcat, blindcatnum, \
            path_kidscats, path_gamacats = shear.input_variables()

    
    print 'Step 1: Create split catalogues in parallel'
    print

    if 'bootstrap' in purpose:
        purpose = purpose.replace('bootstrap', 'catalog')
        path_splits = '%s/splits_%s'%(path_catalogs, purpose)
        path_results = '%s/results_%s'%(path_catalogs, purpose)

        if (Nsplit==0) and (blindcat==blindcats[0]):
            if not os.path.isdir(path_splits):
                os.makedirs(path_splits)
            if not os.path.isdir(path_results):
                os.makedirs(path_results)


    # You can make two kinds of catalog
    if 'catalog' in purpose:

        Nfofmin = 2
        Nfofmax = inf
        binname = 'None'
        lens_binning = {'None': np.array([])}
        
        if Nsplits < Nobsbins:
            Nsplits = Nobsbins
            Nsplit = binnum-1

        Nobsbins = 1
        binnum = 1

        if centering == 'Cen':
            ranks = np.array([1, 1])

        else:
            if all(ranks > 0): # Group catalog
                ranks = np.array([1, inf])
            else: # Galaxy catalog
                ranks = np.array([-999, inf])


    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(purpose, centering, ranks, binname, \
    binnum, Nobsbins, lens_selection, src_selection, name_Rbins, O_matter, O_lambda, Ok, h)

    if ('random' in purpose):
        filename_var = '%i_%s'%(Ncat+1, filename_var) # Ncat is the number of existing catalogs, we want to go one beyond
        print 'Number of new catalog:', Ncat+1
    #		print 'Splits already written: \n', splitslist

    # Stop if the catalog already exists.
    outname = shear.define_filename_results(path_results, purpose, filename_var, filename_addition, Nsplit, blindcat)
    print 'Requested file:', outname
    print

    if os.path.isfile(outname):
        print 'This output already exists:', outname
        print
        quit()


    # Printing a placeholder file, that tells other codes that this catalog is being written.
    if ('random' in purpose):
        filename = shear.define_filename_splits(path_splits, purpose, filename_var, Nsplit+1, 0, filename_addition, blindcat)
        with open(filename, 'w') as file:
            print >>file, ''
        print 'Placeholder:', filename, 'is written.'


    # Importing all GAMA data, and the information on radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, Rcenters, nRbins, \
    gamacat, galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist = \
    shear.import_data(path_Rbins, path_gamacats, path_kidscats, centering, \
    purpose, Ncat, ranks, O_matter, O_lambda, Ok, h)


    # Calculate the source variance
    """
    w_varlist = np.array([])
    e1_varlist = np.array([[]]*4) # These lists will contain all used ellipticities for the variance calculation
    e2_varlist = np.array([[]]*4)

    print 'Importing KiDS catalogs from: %s'%path_kidscats
    i = 0
    for kidscatname in kidscats:
        i += 1
        print '	%i/%i:'%(i, len(kidscats)), kidscatname
        
        # Import and mask all used data from the sources in this KiDS field
        srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm = \
        shear.import_kidscat(path_kidscats, kidscatname, kidscat_end, src_selection)

        # Make ellipticity- and lfweight-lists for the variance calculation
        w_varlist = np.append(w_varlist, w)
        e1_varlist = np.hstack([e1_varlist, e1.T])
        e2_varlist = np.hstack([e2_varlist, e2.T])

    variance = shear.calc_variance(e1_varlist, e2_varlist, w_varlist) # Calculating the variance of the ellipticity for this source selection
    """
    # !!!!!!! This is temporary, to speed up the code !!!!!!
    variance = np.array([0.06423509, 0.08734607, 0.07392235, 0.07810604])


    # Binnning information of the groups
    lenssel = shear.define_lenssel(gamacat, ranks, lens_selection, 'None', -inf, inf) # Mask the galaxies in the shear catalog, WITHOUT binning (for the bin creation)
    obsbins, binmin, binmax = shear.define_obsbins(binname, binnum, lens_selection, lenssel, gamacat)


    # We translate the range in source redshifts to a range in source distances Ds
    zsrcbins = np.arange(0.025,3.5,0.05) # The range of redshifts corresponding to the 70 values in the probability distribution
    Dcsbins = np.array([distance.comoving(y, O_matter, O_lambda, h) for y in zsrcbins])

    # Printing the made choices

    print
    print '%s split'%purpose, Nsplit+1, '/', Nsplits, ', Center definition = %s'%centering
    print

# If the KiDS catalogs are chosen manually:
#	kidscats = ['KIDS_231p0_0p5'] # Fewest lenses
#    kidscats = ['KIDS_172p5_2p5'] # 710 lenses
#	kidscats = ['KIDS_179p0_m0p5'] # Most lenses
#	kidscats = ['KIDS_130p0_m0p5']#, 'KIDS_130p0_0p5', 'KIDS_131p0_0p5', 'KIDS_130p0_1p5'] # corresponds to Edo's test group: group ID 100065 (galaxy ID 214484)


    if 'catalog' in purpose:
        # These lists will contain the final output
        outputnames = ['gammat_A', 'gammax_A', 'gammat_B', 'gammax_B', 'gammat_C', 'gammax_C', 'gammat_D', 'gammax_D', 'lfweight', 'lfweight^2', 'k', 'k^2', 'lfweight*k^2', 'lfweight^2*k^4', 'lfweight^2*k^2', 'Nsources', 'bias_m']
        output = np.zeros([len(outputnames), len(galIDlist), nRbins])

    # Split up the list of KiDS fields, for parallel processing
    splitkidscats = np.array(shear.split(kidscats, Nsplits))

    # Start of the reduction of one KiDS field
    kidscatN = 0

    start_tot = time.time()


    for kidscatname in splitkidscats[Nsplit]:

        memfrac = memory.test() # Check which fraction of the memory is full
        while memfrac > 80: # If it is too high...
            print 'Waiting: More memory required'
            time.sleep(30) # wait before continuing the calculation

        kidscatN = kidscatN+1

        lenssel = shear.define_lenssel(gamacat, ranks, lens_selection, binname, binmin, binmax)
        matched_galIDs = np.array(catmatch[kidscatname]) # The ID's of the galaxies that lie in this field

        galIDs, galRAs, galDECs, galZs, Dcls, Dals, galIDmask = shear.mask_gamacat(purpose, matched_galIDs, lenssel, galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist)


        print 'Analysing field %i/%i: %s (contains %i objects)'%(kidscatN, len(splitkidscats[Nsplit]), kidscatname, len(galIDs))
        if ('random' in purpose):
            print '	of catalog:', Ncat+1

        # Split the list of lenses into chunks of 100 max.
        lenssplits = np.append(np.arange(0, len(galIDs), 100), len(galIDs))

        # Import and mask all used data from the sources in this KiDS field
        srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm = shear.import_kidscat(path_kidscats, kidscatname, kidscat_end, src_selection)


        if 'covariance' in purpose:
            # These lists will contain the final covariance output [Noutput, Nsrc, NRbins]
            outputnames = ['Cs', 'Ss', 'Zs']
            output = [Cs_out, Ss_out, Zs_out] = np.zeros([len(outputnames), len(w), nRbins])


        for l in xrange(len(lenssplits)-1):
            print 'Lens split %i/%i:'%(l+1, len(lenssplits))#, lenssplits[l], '-', lenssplits[l+1]

            # Select all the lens properties that are in this lens split
            galID_split, galRA_split, galDEC_split, galZ_split, Dcl_split, Dal_split = [galIDs[lenssplits[l] : lenssplits[l+1]], galRAs[lenssplits[l] : lenssplits[l+1]], galDECs[lenssplits[l] : lenssplits[l+1]], galZs[lenssplits[l] : lenssplits[l+1]], Dcls[lenssplits[l] : lenssplits[l+1]], Dals[lenssplits[l] : lenssplits[l+1]]]

            galIDmask_split = np.in1d(galIDlist, galID_split) # Create a mask for the complete list of lenses, that only highlights the lenses in this lens split

            # Calculate the projected distance (srcR) and the shear (gamma_t and gamma_x) of all lens-source pairs.
            srcR, incosphi, insinphi = shear.calc_shear(Dal_split, galRA_split, galDEC_split, srcRA, srcDEC, e1, e2, Rmin, Rmax)

            # Calculate k (=1/Sigma_crit) and the weight-mask of every lens-source pair
            k, kmask = shear.calc_Sigmacrit(Dcl_split, Dal_split, Dcsbins, srcPZ)
            Nsrc = np.ones(np.shape(k))
            srcR = np.ma.filled(np.ma.array(srcR, mask = kmask, fill_value = 0)) # Mask all invalid lens-source pairs using the value of the radius

            w_meshed, foo = np.meshgrid(w,np.zeros(len(k))) # Create an lfweight matrix that can be masked according to lens-source distance
            foo = [] # Remove unused lists

            # Start the reduction of one radial bin
            for r in xrange(nRbins):
    #					print 'Rbin', r+1, Rbins[r], '-', Rbins[r+1]

                # Masking the data of all lens-source pairs according to the radial binning
                Rmask = np.logical_not((Rbins[r] < srcR) & (srcR < Rbins[r+1]))
                kmask = [] # Remove unused lists

                if np.sum(Rmask) != np.size(Rmask):
    #					print 'fullness:', (float(np.size(Rmask)) - float(np.sum(Rmask)))/float(np.size(Rmask))

                    unmasked = [incosphi, insinphi, k, w_meshed, Nsrc]
                    masked = [incosphilist, insinphilist, klist, wlist, Nsrclist] = [np.ma.filled(np.ma.array(u, mask = Rmask, fill_value = 0)) for u in unmasked]

                    if 'catalog' in purpose:
                        # For each radial bin of each lens we calculate the weights and weighted shears
                        output_onebin = [gammat_tot_A, gammax_tot_A, gammat_tot_B, gammax_tot_B, gammat_tot_C, gammax_tot_C, gammat_tot_D, gammax_tot_D, w_tot, w2_tot, k_tot, k2_tot, wk2_tot, w2k4_tot, w2k2_tot, Nsrc_tot, srcm_tot] = shear.calc_shear_output(incosphilist, insinphilist, e1, e2, Rmask, klist, wlist, Nsrclist, srcm)
    #							print 'Bin', r+1, 'gammat_A', gammat_tot_A

                        # Writing the lenssplit list to the complete output lists: [galIDs, Rbins] for every variable
                        for o in xrange(len(output)):
                            output[o, : ,r][galIDmask_split] = output[o, : ,r][galIDmask_split] + output_onebin[o]

                    if purpose == 'covariance':
                        # For each radial bin of each lens we calculate the weighted Cs, Ss and Zs
                        output_onebin = [C_tot, S_tot, Z_tot] = shear.calc_covariance_output(incosphilist, insinphilist, klist)

                        # Writing the complete output to the output lists
                        for o in xrange(len(output)):
                            output[o, : ,r] = output[o, : ,r] + output_onebin[o]

        # Write the final output of this split to a fits file
        if purpose == 'covariance':
            filename = shear.define_filename_splits(path_splits, purpose, filename_var, kidscatname, 0, filename_addition, blindcat)
            shear.write_catalog(filename, srcNr, Rbins, Rcenters, nRbins, output, outputnames, variance, purpose, e1, e2, w, srcm)

    if ('random' in purpose):
        if os.path.isfile(filename):
            os.remove(filename)
            print 'Placeholder:', filename, 'is removed.'

    if 'catalog' in purpose:
        filename = shear.define_filename_splits(path_splits, purpose, filename_var, Nsplit+1, Nsplits, filename_addition, blindcat)
        shear.write_catalog(filename, galIDlist, Rbins, Rcenters, nRbins, output, outputnames, variance, purpose, e1, e2, w, srcm)
        print 'Written:', filename

    end_tot = time.time()-start_tot
    print 'Split finished in', end_tot/60., 'minutes'
    print

    return

main()
