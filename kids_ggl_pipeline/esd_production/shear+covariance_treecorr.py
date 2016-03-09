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
import time
import multiprocessing as multi

from astropy import constants as const, units as u
import memory_test as memory
import time
import gc
import treecorr

import shearcode_modules as shear


# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number


def loop_multi(Nsplit, output, outputnames, gamacat, centering, \
               lens_selection, lens_binning, binname, binnum, \
               binmin, binmax, filename_var, filename):
    
    q1 = multi.Queue()
    procs = []
    #chunk = int(np.ceil(len(R)/float(nprocs)))
    
    for j in xrange(Nsplit):
    
        work = multi.Process(target=loop, args=(j, output, outputnames, \
                                                gamacat, centering, \
                                                lens_selection, lens_binning, \
                                                binname, binnum, binmin, \
                                                binmax, filename_var, filename))
        procs.append(work)
        work.start()

    return

def loop(Nsplit, output, outputnames, gamacat, centering, \
         lens_selection, lens_binning, binname, binnum, \
         binmin, binmax, filename_var, filename):
    
    if 'catalog' in purpose:
        # These lists will contain the final output
        
        outputnames = ['gammat_A', 'gammax_A', 'gammat_B', 'gammax_B', \
            'gammat_C', 'gammax_C', 'gammat_D', 'gammax_D', \
            'k', 'k^2', \
            'lfweight_A*k^2', 'lfweight_B*k^2', 'lfweight_C*k^2', \
            'lfweight_D*k^2', \
            'lfweight_A^2*k^2', 'lfweight_B^2*k^2', \
            'lfweight_C^2*k^2', 'lfweight_D^2*k^2', 'Nsources', \
            'bias_m_A', 'bias_m_B', 'bias_m_C', 'bias_m_D']
            
        output = np.zeros([len(outputnames), len(galIDlist), nRbins])
    
    # Start of the reduction of one KiDS field
    kidscatN = 0
    
    if cat_version == 2:
        print('TreeCorr based calculation is only supported with KiDS-450 or newer. Quitting...')
        quit()

    if cat_version == 3:
        
        for kidscatname in splitkidscats[Nsplit]:
            start = time.time()
            index = np.array(np.where(tile_varlist == catmatch[kidscatname][1]))[0]
            
            memfrac = memory.test() # Check which fraction of the memory is full
            while memfrac > 90: # If it is too high...
                print 'Waiting: More memory required'
                time.sleep(30) # wait before continuing the calculation
            
            kidscatN = kidscatN+1
            lenssel = shear.define_lenssel(gamacat, centering, lens_selection, \
                                           lens_binning, binname, binnum, \
                                           binmin, binmax)
            
            # The ID's of the galaxies that lie in this field
            matched_galIDs = np.array(catmatch[kidscatname][0])
            
            # Find the selected lenses that lie in this KiDS field
            if 'catalog' in purpose:
                galIDmask = np.in1d(galIDlist, matched_galIDs)
            if 'covariance' in purpose:
                galIDmask = np.in1d(galIDlist, matched_galIDs) & lenssel
            [galIDs, galRAs, galDECs, galweights, galZs, Dcls, Dals] = \
            [gallist[galIDmask] for gallist in [galIDlist, galRAlist, \
                                                galDEClist, galweightlist, \
                                                galZlist, Dcllist, Dallist]]
            
            print 'Analysing part %i/%i, process %i: %s (contains %i objects)'\
                            %(kidscatN, len(splitkidscats[Nsplit]), \
                              Nsplit+1, kidscatname, len(galIDs))
            if ('random' in purpose):
                print '	of catalog:', Ncat+1
            
            # Split the list of lenses into chunks of 100 max.
            lenssplits = np.append(np.arange(0, len(galIDs), 100), len(galIDs))
            
            # Import and mask all used data from the sources in this KiDS field
            
            srcNr = srcNr_varlist[(index)]
            srcRA = srcRA_varlist[(index)]
            srcDEC = srcDEC_varlist[(index)]
            w = w_varlist[:,index][[0,1,2,3],:].T
            e1 = e1_varlist[:,index][[0,1,2,3],:].T
            e2 = e2_varlist[:,index][[0,1,2,3],:].T
            srcm = srcm_varlist[(index)]
            tile = tile_varlist[(index)]
            
            if len(srcNr) != 0:
                srcPZ = np.array([srcPZ_a,]  *len(srcNr))
            else:
                srcPZ = np.zeros((0, len(srcPZ_a)), dtype=np.float64)
            
            if 'covariance' in purpose:
                # These lists will contain the final
                # covariance output [Noutput, Nsrc, NRbins]
                outputnames = ['Cs', 'Ss', 'Zs']
                output = [Cs_out, Ss_out, Zs_out] = \
                        np.zeros([len(outputnames), len(w.T[0]), nRbins])
        
            for l in xrange(len(lenssplits)-1):
                print 'Lens split %i/%i:'%(l+1, len(lenssplits)-1), lenssplits[l], '-', lenssplits[l+1]
                
                # Select all the lens properties that are in this lens split
                galID_split, galRA_split, galDEC_split, galZ_split, Dcl_split, \
                Dal_split, galweights_split = [galIDs[lenssplits[l] : lenssplits[l+1]], \
                                               galRAs[lenssplits[l] : lenssplits[l+1]], \
                                               galDECs[lenssplits[l] : lenssplits[l+1]], \
                                               galZs[lenssplits[l] : lenssplits[l+1]], \
                                               Dcls[lenssplits[l] : lenssplits[l+1]], \
                                               Dals[lenssplits[l] : lenssplits[l+1]], \
                                               galweights[lenssplits[l] : lenssplits[l+1]]]

                # Create a mask for the complete list of lenses,
                # that only highlights the lenses in this lens split
                galIDmask_split = np.in1d(galIDlist, galID_split)
                        
                # Calculate the projected distance (srcR) and the shear
                # (gamma_t and gamma_x) of all lens-source pairs.
                srcR, incosphi, insinphi = shear.calc_shear(Dal_split, \
                                                            galRA_split, \
                                                            galDEC_split, \
                                                            srcRA, srcDEC, \
                                                            e1, e2, Rmin, Rmax)
                                               
                # Calculate k (=1/Sigma_crit) and the weight-mask
                # of every lens-source pair
                k, kmask = shear.calc_Sigmacrit(Dcl_split, Dal_split, \
                                                Dcsbins, srcPZ, cat_version)
                Nsrc = np.ones(np.shape(k))
                # Mask all invalid lens-source pairs using
                # the value of the radius
                srcR = np.ma.filled(np.ma.array(srcR, mask = kmask, \
                                                fill_value = 0))
                
                # Create an lfweight matrix that can be
                # masked according to lens-source distance
                w_meshed_A, foo = np.meshgrid(w.T[0],np.zeros(len(k)))
                w_meshed_B, foo = np.meshgrid(w.T[1],np.zeros(len(k)))
                w_meshed_C, foo = np.meshgrid(w.T[2],np.zeros(len(k)))
                w_meshed_D, foo = np.meshgrid(w.T[3],np.zeros(len(k)))
                w_meshed = [w_meshed_A, w_meshed_B, w_meshed_C, w_meshed_D]
                w_meshed_A, w_meshed_B, w_meshed_C, w_meshed_D = [], [], [], []
                foo = [] # Remove unused lists
                
                if 'catalog' in purpose:
                    # For each radial bin of each lens we calculate
                    # the weights and weighted shears
                    
                    k_tot, k2_tot, wk2_tot_A, wk2_tot_B, wk2_tot_C, wk2_tot_D, \
                    w2k2_tot_A, w2k2_tot_B, w2k2_tot_C, w2k2_tot_D, Nsrc_tot, \
                    srcm_tot_A, srcm_tot_B, srcm_tot_C, srcm_tot_D = \
                        shear.calc_shear_output_tree(e1, e2, k,\
                                                np.array(w_meshed), Nsrc, srcm)
                    
                    gammat_tot_A = np.zeros((len(galID_split), nRbins))
                    gammax_tot_A = np.zeros((len(galID_split), nRbins))
                    gammat_tot_B = np.zeros((len(galID_split), nRbins))
                    gammax_tot_B = np.zeros((len(galID_split), nRbins))
                    gammat_tot_C = np.zeros((len(galID_split), nRbins))
                    gammax_tot_C = np.zeros((len(galID_split), nRbins))
                    gammat_tot_D = np.zeros((len(galID_split), nRbins))
                    gammax_tot_D = np.zeros((len(galID_split), nRbins))
                    
                    
                    # THIS ALL WON'T WORK. Calculating shears is fine, just that all the weighting comming in is impossible to account for. We can still do something for randoms, other stuff is actually reasonably fine in terms of speed.
                    for i in xrange(len(galID_split)):
                        
                        lens_cat = treecorr.Catalog(ra=[galRA_split[i]], dec=[galDEC_split[i]], ra_units='degrees', dec_units='degrees')
                        src_cat_A = treecorr.Catalog(ra=srcRA, dec=srcDEC, g1=e1.T[0], g2=e2.T[0], w=w.T[0]*k[i,:]**2.0, ra_units='degrees', dec_units='degrees', flip_g2=True)
                        #src_cat_B = treecorr.Catalog(ra=srcRA, dec=srcDEC, g1=e1.T[1], g2=e2.T[1], w=w.T[1]*k[i,:]**2.0, ra_units='degrees', dec_units='degrees', flip_g2=True)
                        #src_cat_C = treecorr.Catalog(ra=srcRA, dec=srcDEC, g1=e1.T[2], g2=e2.T[2], w=w.T[2]*k[i,:]**2.0, ra_units='degrees', dec_units='degrees', flip_g2=True)
                        # Setting up the tree
                        gglensA = treecorr.NGCorrelation(nbins=nRbins, min_sep=Rmin/Dal_split[i], max_sep=Rmax/Dal_split[i], sep_units='radians')
                        #gglensB = treecorr.NGCorrelation(nbins=nRbins, min_sep=Rmin/Dal_split[i], max_sep=Rmax/Dal_split[i], sep_units='radians')
                        #gglensC = treecorr.NGCorrelation(nbins=nRbins, min_sep=Rmin/Dal_split[i], max_sep=Rmax/Dal_split[i], sep_units='radians')
                        # Calculating shear signal
                        gglensA.process(lens_cat, src_cat_A, num_threads=1)
                        #gglensB.process(lens_cat, src_cat_B, num_threads=1)
                        #gglensC.process(lens_cat, src_cat_C, num_threads=1)
                        
                        gammat_tot_A[i,:], gammax_tot_A[i,:] = gglensA.xi/k[i,0], gglensA.xi_im/k[i,0]
                        
                        #gammat_tot_B[i,:], gammax_tot_B[i,:] = gglensB.xi/k[i,0], gglensB.xi_im/k[i,0]
                        #gammat_tot_C[i,:], gammax_tot_C[i,:] = gglensC.xi/k[i,0], gglensC.xi_im/k[i,0]
                        #gammat_tot_D[i,:], gammax_tot_D[i,:] = gglensC.xi/k[i,0], gglensC.xi_im/k[i,0]
                    """
                    output_onebin = [gammat_tot_A, gammax_tot_A, \
                                     gammat_tot_B, gammax_tot_B, \
                                     gammat_tot_C, gammax_tot_C, \
                                     gammat_tot_D, gammax_tot_D, \
                                     k_tot*np.ones((len(galID_split), nRbins)), k2_tot*np.ones((len(galID_split), nRbins)), \
                                     wk2_tot_A*np.ones((len(galID_split), nRbins)), wk2_tot_B*np.ones((len(galID_split), nRbins)), \
                                     wk2_tot_C*np.ones((len(galID_split), nRbins)), wk2_tot_D*np.ones((len(galID_split), nRbins)), \
                                     w2k2_tot_A*np.ones((len(galID_split), nRbins)), w2k2_tot_B*np.ones((len(galID_split), nRbins)), \
                                     w2k2_tot_C*np.ones((len(galID_split), nRbins)), w2k2_tot_D*np.ones((len(galID_split), nRbins)), Nsrc_tot*np.ones((len(galID_split), nRbins)), \
                                     srcm_tot_A*np.ones((len(galID_split), nRbins)), srcm_tot_B*np.ones((len(galID_split), nRbins)), \
                                     srcm_tot_C*np.ones((len(galID_split), nRbins)), srcm_tot_D*np.ones((len(galID_split), nRbins))]
                    # Writing the lenssplit list to the complete
                    # output lists: [galIDs, Rbins] for every variable
                    for o in xrange(len(output)):
                        output[o, : ,:][galIDmask_split] = \
                        output[o, : ,:][galIDmask_split] + \
                            output_onebin[o]
                    """
                if 'covariance' in purpose:
                    # Start the reduction of one radial bin
                    for r in xrange(nRbins):
                        # Masking the data of all lens-source pairs according
                        # to the radial binning
                        Rmask = np.logical_not((Rbins[r] < srcR) \
                                        & (srcR < Rbins[r+1]))
                        kmask = [] # Remove unused lists
                    
                        if np.sum(Rmask) != np.size(Rmask):
                        
                            incosphilist = np.ma.filled(np.ma.array(incosphi, \
                                                                mask = Rmask, \
                                                                fill_value = 0))
                            insinphilist = np.ma.filled(np.ma.array(insinphi, \
                                                                mask = Rmask, \
                                                                fill_value = 0))
                            klist = np.ma.filled(np.ma.array(k, mask = Rmask, \
                                                         fill_value = 0))
                            Nsrclist = np.ma.filled(np.ma.array(Nsrc, mask = Rmask,\
                                                            fill_value = 0))
                            wlist = np.array([np.ma.filled(np.ma.array(u, \
                                                            mask = Rmask, \
                                                            fill_value = 0)) \
                                          for u in w_meshed])
                    
                    
                            # For each radial bin of each lens we calculate
                            # the weighted Cs, Ss and Zs
                            output_onebin = [C_tot, S_tot, Z_tot] = \
                            shear.calc_covariance_output(incosphilist, \
                                                         insinphilist, klist, \
                                                         galweights_split)
                    
                            # Writing the complete output to the output lists
                            for o in xrange(len(output)):
                                output[o, : ,r] = output[o, : ,r] + \
                                    output_onebin[o]
        
            # Write the final output of this split to a fits file
            if 'covariance' in purpose:
                filename = shear.define_filename_splits(path_splits, purpose, \
                                                        filename_var, \
                                                        kidscatname, 0, \
                                                        filename_addition, \
                                                        blindcat)
                shear.write_catalog(filename, srcNr, Rbins, Rcenters, nRbins, \
                                    Rconst, output, outputnames, variance, \
                                    purpose, e1, e2, w, srcm)
            end = time.time()
            print end-start
            
        quit()
        if ('random' in purpose):
	    
            if os.path.isfile(filename):
                os.remove(filename)
                print 'Placeholder:', filename, 'is removed.'
    	    
        if 'catalog' in purpose:
            filename = shear.define_filename_splits(path_splits, purpose, \
                                                    filename_var, Nsplit+1, \
                                                    Nsplits, filename_addition,\
                                                    blindcat)
            shear.write_catalog(filename, galIDlist, Rbins, Rcenters, nRbins, \
                                Rconst, output, outputnames, variance, \
                                purpose, e1, e2, w, srcm)
            print 'Written:', filename

    return


if __name__ == '__main__':

    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
    lens_selection, lens_weights, binname, Nobsbins, src_selection, \
    cat_version, path_Rbins, name_Rbins, Runit, path_output, path_splits, \
    path_results, purpose, O_matter, O_lambda, Ok, h, filename_addition, Ncat, \
    splitslist, blindcats, blindcat, blindcatnum, path_kidscats, \
    path_gamacat = shear.input_variables()

    print 'Step 1: Create split catalogues in parallel'
    print


    if 'bootstrap' in purpose:
        purpose = purpose.replace('bootstrap', 'catalog')

        path_catalogs = '%s/catalogs'%(path_output.rsplit('/',1)[0])
        path_splits = '%s/splits_%s'%(path_catalogs, purpose)
        path_results = '%s/results_%s'%(path_catalogs, purpose)

        if (Nsplit==0) and (blindcat==blindcats[0]):
            if not os.path.isdir(path_splits):
                os.makedirs(path_splits)
            if not os.path.isdir(path_results):
                os.makedirs(path_results)

    if 'catalog' in purpose:

        binname = 'None'
        lens_binning = {'None': ['self', np.array([])]}
        #if Nsplits < Nobsbins:
        #    Nsplits = Nobsbins
        #    Nsplit = binnum-1
        Nobsbins = 1
        binnum = 1

        if centering == 'Cen':
            lens_selection = {'rank%s'%centering: ['self', np.array([1])]}
        else:
            lens_selection = {}

        lens_weights = {'None': ''}

    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(purpose, centering, binname, \
    binnum, Nobsbins, lens_selection, src_selection, lens_weights, \
    name_Rbins, O_matter, O_lambda, Ok, h)

    if ('random' in purpose):
        filename_var = '%i_%s'%(Ncat+1, filename_var)
        # Ncat is the number of existing catalogs, we want to go one beyond
        print 'Number of new catalog:', Ncat+1
    #		print 'Splits already written: \n', splitslist

    # Stop if the catalog already exists.
    outname = shear.define_filename_results(path_results, purpose, \
                                            filename_var, filename_addition, \
                                            Nsplit, blindcat)
    print 'Requested file:', outname
    print

    if os.path.isfile(outname):
        print 'This output already exists:', outname
        print
        quit()

    
    # Printing a placeholder file, that tells other codes \
    #    that this catalog is being written.
    if ('random' in purpose):
        #filename = 0
        for i in xrange(Nsplits):
            filename = shear.define_filename_splits(path_splits, purpose, \
                                                filename_var, i+1, \
                                                Nsplits, filename_addition, \
                                                blindcat)
    	
            with open(filename, 'w') as file:
                print >>file, ''
            print 'Placeholder:', filename, 'is written.'

    else:
        filename = 0
    # Importing all GAMA data, and the information
    # on radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, \
    Rcenters, nRbins, Rconst, gamacat, galIDlist, galRAlist, galDEClist, \
    galweightlist, galZlist, Dcllist, Dallist = \
    shear.import_data(path_Rbins, Runit, path_gamacat, path_kidscats, \
                      centering, purpose, Ncat, O_matter, O_lambda, Ok, h, \
                      lens_weights, filename_addition, cat_version)

    # Calculate the source variance

    # These lists will contain all ellipticities for the variance calculation
    w_varlist = np.array([[]]*4)
    e1_varlist = np.array([[]]*4)
    e2_varlist = np.array([[]]*4)

    if cat_version == 2:
        print 'Importing KiDS catalogs from: %s'%path_kidscats
        i = 0
        for kidscatname in kidscats:
            i += 1
            print '	%i/%i:'%(i, len(kidscats)), kidscatname
        
            # Import and mask all used data from the sources in this KiDS field
            srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm, tile = \
            shear.import_kidscat(path_kidscats, kidscatname, kidscat_end, \
                                 src_selection, cat_version)

            # Make ellipticity- and lfweight-lists for the variance calculation
            w_varlist = np.hstack([w_varlist, w.T])
            e1_varlist = np.hstack([e1_varlist, e1.T])
            e2_varlist = np.hstack([e2_varlist, e2.T])

    if cat_version == 3:
        srcNr_varlist = np.array([])
        srcRA_varlist = np.array([])
        srcDEC_varlist = np.array([])
        srcm_varlist = np.array([])
        tile_varlist = np.array([])
        
        kidscatname2 = np.array([])
        for i in xrange(len(kidscats)):
            kidscatname2 = np.append(kidscatname2, \
                                     kidscats[i].rsplit('-', 1)[0])
        
        kidscatname2 = np.unique(kidscatname2)

        print 'Importing KiDS catalogs from: %s'%path_kidscats
        i = 0
        for kidscatname in kidscatname2:
            i += 1
            print '	%i/%i:'%(i, len(kidscatname2)), kidscatname
            
            # Import and mask all used data from the sources in this KiDS field
            srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm, tile = \
            shear.import_kidscat(path_kidscats, kidscatname, \
                                 kidscat_end, src_selection, cat_version)
            
            srcNr_varlist = np.append(srcNr_varlist, srcNr)
            srcRA_varlist = np.append(srcRA_varlist, srcRA)
            srcDEC_varlist = np.append(srcDEC_varlist, srcDEC)
            srcm_varlist = np.append(srcm_varlist, srcm)
            tile_varlist = np.append(tile_varlist, tile)
            
            # Make ellipticity- and lfweight-lists for the variance calculation
            w_varlist = np.hstack([w_varlist, w.T])
            e1_varlist = np.hstack([e1_varlist, e1.T])
            e2_varlist = np.hstack([e2_varlist, e2.T])
            srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm, tile = [], [], [], \
                                                                [], [], [], \
                                                                [], [], []

    # Calculating the variance of the ellipticity for this source selection
    variance = shear.calc_variance(e1_varlist, e2_varlist, w_varlist)

    # Binnning information of the groups
    lenssel_binning = shear.define_lenssel(gamacat, centering, lens_selection, \
                                           'None', 'None', 0, -inf, inf)

    # Mask the galaxies in the shear catalog, WITHOUT binning
    binname, lens_binning, Nobsbins, binmin, binmax = \
    shear.define_obsbins(binnum, lens_binning, lenssel_binning, gamacat)


    # We translate the range in source redshifts
    # to a range in source distances Ds
    zsrcbins = np.arange(0.025,3.5,0.05)
    Dcsbins = np.array([distance.comoving(y, O_matter, O_lambda, h) \
                        for y in zsrcbins])

    if cat_version == 3:
        srcNZ, spec_weight = shear.import_spec_cat(path_kidscats, kidscatname2,\
                                                   kidscat_end, src_selection, \
                                                   cat_version)
        srcPZ_a, bins = np.histogram(srcNZ, range=[0.025, 3.5], bins=70, \
                                     weights=spec_weight, density=1)
        srcPZ_a = srcPZ_a/srcPZ_a.sum()

    # Printing the made choices

    print
    print 'Using %s cores to create split catalogues'%Nsplits, \
            ', - Center definition = %s'%centering
    print

    output = 0
    outputnames = 0
                          
    # Split up the list of KiDS fields, for parallel obsbins
    splitkidscats = np.array(shear.split(kidscats, Nsplits))
    
    if 'catalog' in purpose:
        calculation = loop_multi(Nsplits, output, outputnames, gamacat, \
                                 centering, lens_selection, lens_binning, \
                                 binname, binnum, binmin, binmax, filename_var, filename)
    if 'covariance' in purpose:
        for i in xrange(Nobsbins):
            filename_var = shear.define_filename_var(purpose, centering, \
                                                     binname, i+1, Nobsbins, \
                                                     lens_selection, \
                                                     src_selection, \
                                                     lens_weights, name_Rbins, \
                                                     O_matter, O_lambda, Ok, h)
                                                     
            binname, lens_binning, Nobsbins, binmin, binmax = \
            shear.define_obsbins(i+1, lens_binning, lenssel_binning, gamacat)
            calculation = loop_multi(Nsplits, output, outputnames, gamacat, \
                                     centering, lens_selection, lens_binning, \
                                     binname, i, binmin, binmax, filename_var, filename)



