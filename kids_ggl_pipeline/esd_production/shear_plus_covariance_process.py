#!/usr/bin/python

"""
# Part of the module to determine the shear
# as a function of radius from a galaxy.
"""
# Import the necessary libraries
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import astropy.io.fits as pyfits
import multiprocessing as mp
from glob import glob
import numpy as np
import sys
import os
import time
from scipy.interpolate import interp1d

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import gc

from . import distance, memory_test as memory, shearcode_modules as shear

if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
    xrange = range


# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number

def loop_multi(purpose, Nsplits, Nsplit, output, outputnames, gamacat,
               colnames, centering, gallists, lens_selection, lens_binning,
               binname, binnum, binmin, binmax, Rbins, Rcenters, Rmin, Rmax,
               Runit, nRbins, Rconst, com, filename_var,
               filename, cat_version, blindcat, blindcats, srclists, path_splits,
               splitkidscats, catmatch, Dcsbins, zlensbins, Dclbins, Dc_epsilon, filename_addition,
               variance, h, path_kidscats, kidscatname, kidscolnames, kidscat_end, src_selection):
    
    q1 = mp.Queue()
    procs = []
    #chunk = int(np.ceil(len(R)/float(nprocs)))
    
    for j in xrange(Nsplits):
        
        work = mp.Process(
            target=loop,
            args=(purpose, Nsplits, j, output, outputnames, gamacat, colnames,
            centering, gallists, lens_selection, lens_binning, binname,
            binnum, binmin, binmax, Rbins, Rcenters, Rmin, Rmax, Runit,
            nRbins, Rconst, com, filename_var, filename, cat_version, blindcat, blindcats,
            srclists, path_splits, splitkidscats, catmatch, Dcsbins, zlensbins, Dclbins,
            Dc_epsilon, filename_addition, variance, h, path_kidscats,
             kidscatname, kidscolnames, kidscat_end, src_selection, q1))
        procs.append(work)
        work.start()
        #work.join()
    
    #for j in xrange(Nsplits):
    work.join()
    
    result = np.zeros(Nsplits)
    for j in xrange(Nsplits):
        result[j] = q1.get()
    q1.close()
    q1.join_thread()

    if len(np.unique(result)) != len(result):
        return 1
    else:
        return 0


def loop(purpose, Nsplits, Nsplit, output, outputnames, gamacat, colnames,
         centering, gallists, lens_selection, lens_binning, binname, binnum,
         binmin, binmax, Rbins, Rcenters, Rmin, Rmax, Runit, nRbins, Rconst, com,
         filename_var,
         filename, cat_version, blindcat, blindcats, srclists, path_splits,
         splitkidscats, catmatch, Dcsbins, zlensbins, Dclbins, Dc_epsilon, filename_addition,
         variance, h, path_kidscats, kidscatname, kidscolnames, kidscat_end, src_selection, q1):

    # galaxy information
    galIDlist, galRAlist, galDEClist, galZlist, galweightlist, \
        Dallist, Dcllist, z_epsilon = gallists
    # source information
    srcNr_varlist, srcRA_varlist, srcDEC_varlist, w_varlist, \
        e1_varlist, e2_varlist, srcm_varlist, tile_varlist, srcPZ_varlist, k_interpolated = srclists

    gallists, srclists = [], []
    gc.collect()

    if 'catalog' in purpose:
        # These lists will contain the final output
        outputnames = ['gammat_'+blind for blind in blindcats] + \
                      ['gammax_'+blind for blind in blindcats] + \
                      ['k', 'k^2'] + \
                      ['lfweight_'+blind+'*k^2' for blind in blindcats] + \
                      ['lfweight_'+blind+'^2*k^2' for blind in blindcats] + \
                      ['Nsources'] + \
                      ['bias_m_'+blind for blind in blindcats]
        output = np.zeros([len(outputnames), len(galIDlist), nRbins])

    # Start of the reduction of one KiDS field
    kidscatN = 0

    if cat_version == 2:
        print('KiDS-DR2 is no longer supported, please use v1.7')
        raise SystemExit()

    if cat_version == 3 or cat_version == 0:

        for kidscatname in splitkidscats[Nsplit]:
            
            index = np.array(np.where(tile_varlist == catmatch[kidscatname][1]))[0]
            """
            memfrac = memory.test() # Check which fraction of the memory is full
            while memfrac > 90: # If it is too high...
                print 'Waiting: More memory required'
                time.sleep(30) # wait before continuing the calculation
            """
            kidscatN = kidscatN+1
            lenssel = shear.define_lenssel(
                gamacat, colnames, centering, lens_selection, lens_binning,
                binname, binnum, binmin, binmax, Dcllist, galZlist, h)

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
            
            if len(galIDs)==0:
                print('NOT analysing part {0}/{1}, process {2}: {3} (contains {4} objects)'\
                      .format(kidscatN, len(splitkidscats[Nsplit]), \
                              Nsplit+1, kidscatname, len(galIDs)))
                continue
            print('Analysing part {0}/{1}, process {2}: {3} (contains {4} objects)'\
                  .format(kidscatN, len(splitkidscats[Nsplit]), \
                          Nsplit+1, kidscatname, len(galIDs)))
            
            # Split the list of lenses into chunks of 100 max.
            # why 100? trying larger numbers (CS)
            lenssplits = np.append(np.arange(0, len(galIDs), 100), len(galIDs))

            # Import and mask all used data from the sources in this KiDS field

            srcNr = srcNr_varlist[(index)]
            srcRA = srcRA_varlist[(index)]
            srcDEC = srcDEC_varlist[(index)]
            w = w_varlist[:,index][[b for b, blind in enumerate(blindcats)],:].T
            e1 = e1_varlist[:,index][[b for b, blind in enumerate(blindcats)],:].T
            e2 = e2_varlist[:,index][[b for b, blind in enumerate(blindcats)],:].T
            srcm = srcm_varlist[(index)]
            tile = tile_varlist[(index)]
            srcZB = srcPZ_varlist[(index)]

            """
            if len(srcNr) != 0:
                srcPZ = np.array([srcPZ_a,]  *len(srcNr))
            else:
                srcPZ = np.zeros((0, len(srcPZ_a)), dtype=np.float64)
            """
            
            #srcPZ = shear.import_spec_cat_pz(catmatch[kidscatname][1], catmatch, srcNr)

            if 'covariance' in purpose:
                # These lists will contain the final
                # covariance output [Noutput, Nsrc, NRbins]
                outputnames = ['Cs', 'Ss', 'Zs']
                output = [Cs_out, Ss_out, Zs_out] = \
                        np.zeros([len(outputnames), len(w.T[0]), nRbins])

            for l in xrange(len(lenssplits)-1):
                print('Lens split {0}/{1}:'.format(l+1, len(lenssplits)-1), \
                      lenssplits[l], '-', lenssplits[l+1])
                # Select all the lens properties that are in this lens split
                galID_split, galRA_split, galDEC_split, galZ_split, Dcl_split, \
                Dal_split, galweights_split = [galIDs[lenssplits[l] : lenssplits[l+1]], \
                                               galRAs[lenssplits[l] : lenssplits[l+1]], \
                                               galDECs[lenssplits[l] : lenssplits[l+1]], \
                                               galZs[lenssplits[l] : lenssplits[l+1]], \
                                               Dcls[lenssplits[l] : lenssplits[l+1]], \
                                               Dals[lenssplits[l] : lenssplits[l+1]], \
                                               galweights[lenssplits[l] : lenssplits[l+1]]]

                galZ_split_2, srcZB_2 = np.meshgrid(galZ_split, srcZB)
                src_mask = np.logical_not(srcZB_2 >= galZ_split_2+z_epsilon).T
                
                
                # Create a mask for the complete list of lenses,
                # that only highlights the lenses in this lens split
                galIDmask_split = np.in1d(galIDlist, galID_split)

                # Calculate the projected distance (srcR) and the shear
                # (gamma_t and gamma_x) of all lens-source pairs.
                srcR, incosphi, insinphi = shear.calc_shear(Dal_split, Dcl_split, \
                                                            galRA_split, \
                                                            galDEC_split, \
                                                            srcRA, srcDEC, \
                                                            e1, e2, Rmin, Rmax, com, \
                                                            Runit, galweights_split)

                # Calculate k (=1/Sigma_crit) and the weight-mask
                # of every lens-source pair
                #k, kmask = shear.calc_Sigmacrit(Dcl_split, Dal_split, \
                #                                Dcsbins, srcPZ, cat_version, Dc_epsilon)
                k = k_interpolated(galZ_split_2).T
                
                Nsrc = np.ones(np.shape(k))
                # Mask all invalid lens-source pairs using
                # the value of the radius
                #new_mask = np.logical_or(kmask,src_mask)
                
                srcR = np.ma.filled(np.ma.array(srcR, mask = src_mask, \
                                                fill_value = 0))

                # Create an lfweight matrix that can be
                # masked according to lens-source distance
                w_meshed = [np.meshgrid(w.T[b],np.zeros(len(k)))[0] for b in xrange(len(blindcats))]
            

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

                        if 'catalog' in purpose:
                            # For each radial bin of each lens we calculate
                            # the weights and weighted shears
                            
                            output_onebin = shear.calc_shear_output(incosphilist, insinphilist,\
                                                    e1, e2, Rmask, klist,\
                                                     wlist, Nsrclist, srcm, Runit, blindcats)
                            
                            # Writing the lenssplit list to the complete
                            # output lists: [galIDs, Rbins] for every variable
                            for o in xrange(len(output)):
                                output[o,:,r][galIDmask_split] = \
                                output[o,:,r][galIDmask_split] + \
                                output_onebin[:,o]

                        if 'covariance' in purpose:
                            # For each radial bin of each lens we calculate
                            # the weighted Cs, Ss and Zs
                            if ('pc' in Runit) or ('mps2' in Runit):
                                output_onebin = [C_tot, S_tot, Z_tot] = \
                                shear.calc_covariance_output(incosphilist, \
                                                             insinphilist, \
                                                             klist, \
                                                             galweights_split, \
                                                             Runit)
                            else:
                                output_onebin = [C_tot, S_tot, Z_tot] = \
                                shear.calc_covariance_output(incosphilist, \
                                                        insinphilist, \
                                                        Nsrclist, \
                                                        galweights_split, \
                                                        Runit)

                            # Writing the complete output to the output lists
                            for o in xrange(len(output)):
                                output[o, : ,r] = output[o,:,r] + output_onebin[o]

            # Write the final output of this split to a fits file
            if 'covariance' in purpose:
                filename = shear.define_filename_splits(path_splits, purpose, \
                                                        filename_var, \
                                                        kidscatname, 0, \
                                                        filename_addition, \
                                                        blindcat)
                if len(srcNr)==0:
                    print('\033[91m' + 'No sources in this tile ...' + '\033[0m')
                    continue
                else:
                    shear.write_catalog(filename, srcNr, Rbins, Rcenters, nRbins, \
                                    Rconst, output, outputnames, variance, \
                                    purpose, e1, e2, w, srcm, blindcats)

        if 'catalog' in purpose:
            filename = shear.define_filename_splits(path_splits, purpose, \
                                                    filename_var, Nsplit+1, \
                                                    Nsplits, filename_addition,\
                                                    blindcat)
            shear.write_catalog(filename, galIDlist, Rbins, Rcenters, nRbins, \
                                Rconst, output, outputnames, variance, \
                                purpose, e1, e2, w, srcm, blindcats)
            print('Written:', filename)

    q1.put(Nsplit)

    return


#if __name__ == '__main__':

def main(nsplit, nsplits, nobsbin, blindcat, config_file, fn):

    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
        lens_selection, lens_weights, binname, Nobsbins, src_selection, \
        cat_version, path_Rbins, name_Rbins, Runit, path_output, \
        path_splits, path_results, purpose, O_matter, O_lambda, Ok, h, \
        filename_addition, Ncat, splitslist, blindcats, blindcat, \
        blindcatnum, path_kidscats, path_gamacat, colnames, kidscolnames, specz_file, m_corr_file,\
        z_epsilon, n_boot, cross_cov, com = \
            shear.input_variables(
                nsplit, nsplits, nobsbin, blindcat, config_file)

    print('Step 1: Create split catalogues in parallel')
    print()


    if 'bootstrap' in purpose:
        purpose = purpose.replace('bootstrap', 'catalog')

        path_catalogs = '{}/catalogs'.format(path_output.rsplit('/',1)[0])
        path_splits = '{0}/splits_{1}'.format(path_catalogs, purpose)
        path_results = '{0}/results_{1}'.format(path_catalogs, purpose)

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
            lens_selection = {'rank{}'.format(centering): ['self', np.array([1])]}
        else:
            lens_selection = {}

        lens_weights = {'None': ''}

    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(purpose, centering, binname, \
    binnum, Nobsbins, lens_selection, lens_binning, src_selection, lens_weights, \
    name_Rbins, O_matter, O_lambda, Ok, h)

    # Stop if the catalog already exists.
    outname = shear.define_filename_results(path_results, purpose, \
                                            filename_var, filename_addition, \
                                            Nsplit, blindcat)
    print('Requested file:', outname)
    print()

    if os.path.isfile(outname):
        print('(in shear_plus_covariance)')
        print('This output already exists:', outname)
        print()
        return


    # Printing a placeholder file, that tells other codes \
    #    that this catalog is being written.
    filename = 0 # Leftover from random.


    # Importing all GAMA data, and the information
    # on radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, \
        Rcenters, nRbins, Rconst, gamacat, galIDlist, galRAlist, galDEClist, \
        galweightlist, galZlist, Dcllist, Dallist = \
        shear.import_data(
            path_Rbins, Runit, path_gamacat, colnames, kidscolnames, path_kidscats,
            centering, purpose, Ncat, O_matter, O_lambda, Ok, h,
            lens_weights, filename_addition, cat_version, com)

    # Calculate the source variance

    # These lists will contain all ellipticities for the variance calculation
    w_varlist = np.array([[]]*np.array(blindcats).shape[0])
    e1_varlist = np.array([[]]*np.array(blindcats).shape[0])
    e2_varlist = np.array([[]]*np.array(blindcats).shape[0])

    if cat_version == 2:
        print('KiDS-DR2 is no longer supported, please use v1.7')
        raise SystemExit()

    # these are dummy variables if cat_version==2
    srcNr_varlist = np.array([])
    srcRA_varlist = np.array([])
    srcDEC_varlist = np.array([])
    srcm_varlist = np.array([])
    tile_varlist = np.array([])
    srcPZ_a = np.array([])
    srcPZ_varlist = np.array([])

    if cat_version in (0,3):
        kidscatname2 = np.array([])
        for cat in kidscats:
            kidscatname2 = np.append(kidscatname2, cat.rsplit('-', 1)[0])
        kidscatname2 = np.unique(kidscatname2)

        print('Importing KiDS catalogs from: {}'.format(path_kidscats))
        i = 0
        for kidscatname in kidscatname2:
            i += 1
            print('	{0}/{1}:'.format(i, len(kidscatname2)), kidscatname)

            # Import and mask all used data from the sources in this
            # KiDS field
            srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm, tile = \
                shear.import_kidscat(path_kidscats, kidscatname, kidscolnames,
                                     kidscat_end, src_selection, cat_version, blindcats)

            srcNr_varlist = np.append(srcNr_varlist, srcNr)
            srcRA_varlist = np.append(srcRA_varlist, srcRA)
            srcDEC_varlist = np.append(srcDEC_varlist, srcDEC)
            srcm_varlist = np.append(srcm_varlist, srcm)
            tile_varlist = np.append(tile_varlist, tile)
            srcPZ_varlist = np.append(srcPZ_varlist, srcPZ)

            # Make ellipticity- and lfweight-lists for the variance calculation
            w_varlist = np.hstack([w_varlist, w.T])
            e1_varlist = np.hstack([e1_varlist, e1.T])
            e2_varlist = np.hstack([e2_varlist, e2.T])
            srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm, tile = \
                [[] for j in range(9)]

    # Calculating the variance of the ellipticity for this source selection
    variance = shear.calc_variance(e1_varlist, e2_varlist, w_varlist)

    # Binnning information of the groups
    lenssel_binning = shear.define_lenssel(
        gamacat, colnames, centering, lens_selection, 'None', 'None',
        0, -inf, inf, Dcllist, galZlist, h)

    # Mask the galaxies in the shear catalog, WITHOUT binning
    binname, lens_binning, Nobsbins, binmin, binmax = \
        shear.define_obsbins(binnum, lens_binning, lenssel_binning, gamacat,
                             Dcllist, galZlist)


    # We translate the range in source and lens redshifts
    # to ranges in distances Ds (in pc/h)
    zsrcbins = np.arange(0.025,3.5,0.05) # Source redshift bins
    zlensbins =  np.linspace(0.025, 0.7, 100) # Lens redshift bins
        
    #Dcsbins = np.array([distance.comoving(y, O_matter, O_lambda, h) \
    #                    for y in zsrcbins])
    #Dc_epsilon = distance.comoving(z_epsilon, O_matter, O_lambda, h)

    # New method
    cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

    Dcsbins = np.array((cosmo.comoving_distance(zsrcbins).to('pc')).value)
    Dclbins = np.array((cosmo.comoving_distance(zlensbins).to('pc')).value)
    
    Dc_epsilon = (cosmo.comoving_distance(z_epsilon).to('pc')).value
    
    if cat_version == 3:
        print('\nCalculating the lensing efficiency ...')

        srclims = src_selection['Z_B'][1]
        sigma_selection = {}
        # 10 lens redshifts for calculation of Sigma_crit
        lens_redshifts = np.arange(0.001, srclims[1]-z_epsilon, 0.05)

        cosmo_eff = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)
        lens_comoving = np.array((cosmo_eff.comoving_distance(lens_redshifts).to('pc')).value)
    
            
        lens_angular = lens_comoving/(1.0+lens_redshifts)
        k = np.zeros_like(lens_redshifts)
            
        for i in xrange(lens_redshifts.size):
            sigma_selection['Z_B'] = ['self', np.array([lens_redshifts[i]+z_epsilon, srclims[1]])]
            srcNZ_k, spec_weight_k = shear.import_spec_cat(path_kidscats, kidscatname2,\
                                        kidscat_end, specz_file, sigma_selection, \
                                        cat_version)
                
            srcPZ_k, bins_k = np.histogram(srcNZ_k, range=[0.025, 3.5], bins=70, \
                                weights=spec_weight_k, density=1)
            srcPZ_k = srcPZ_k/srcPZ_k.sum()
            k[i], kmask = shear.calc_Sigmacrit(np.array([lens_comoving[i]]), np.array([lens_angular[i]]), \
                            Dcsbins, zlensbins, Dclbins, srcPZ_k, cat_version, Dc_epsilon, np.array([lens_redshifts[i]]), com)
            
        k_interpolated = interp1d(lens_redshifts, k, kind='cubic', bounds_error=False, fill_value=(0.0, 0.0))


    if cat_version == 0:
        print('\nCalculating the lensing efficiency ...')
            
        srclims = src_selection['z_photometric'][1]
        sigma_selection = {}
        # 10 lens redshifts for calculation of Sigma_crit
        lens_redshifts = np.arange(0.001, srclims[1]-z_epsilon, 0.05)
        
        cosmo_eff = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)
        lens_comoving = np.array((cosmo_eff.comoving_distance(lens_redshifts).to('pc')).value)
        
            
        lens_angular = lens_comoving/(1.0+lens_redshifts)
        k = np.zeros_like(lens_redshifts)
    
        for i in xrange(lens_redshifts.size):
            sigma_selection['z_photometric'] = ['self', np.array([lens_redshifts[i]+z_epsilon, srclims[1]])]
            srcNZ_k, spec_weight_k = shear.import_spec_cat_mocks(path_kidscats, kidscatname2,\
                                    kidscat_end, specz_file, sigma_selection, \
                                    cat_version)
                    
            srcPZ_k, bins_k = np.histogram(srcNZ_k, range=[0.025, 3.5], bins=70, \
                                            weights=spec_weight_k, density=1)
            srcPZ_k = srcPZ_k/srcPZ_k.sum()
            k[i], kmask = shear.calc_Sigmacrit(np.array([lens_comoving[i]]), np.array([lens_angular[i]]), \
                                                        Dcsbins, zlensbins, Dclbins, srcPZ_k, cat_version, Dc_epsilon, com)
            
        k_interpolated = interp1d(lens_redshifts, k, kind='cubic', bounds_error=False, fill_value=(0.0, 0.0))


    # Calculation of average m correctionf for KiDS-450
    if cat_version == 3:
        # Values are hardcoded, if image simulations give
        # different results this must be changed!
        print('\nCalculating average m correction...')

        #m_corr = np.unique(srcm_varlist)
        # Import the m-corr values from file...
        # This needs to be passed from config file!
        m_file = glob(m_corr_file)
        if len(m_file) == 0:
            msg = 'm-corr file not found.'
            raise IOError(msg)
        elif len(m_file) > 1:
            msg = 'm-corr file name {0} ambiguous. Using file {1}.'.format(
                                                            filename, m_file[0])
        m_corr = np.genfromtxt(m_file[0])
        m_selection = {}
        dlsods = np.zeros(m_corr[:,2].shape)
        
        for i, m in enumerate(m_corr):
            #srcm_i = srcm_varlist[srcm_varlist == m] ??
            m_selection['Z_B'] = ['self', np.array([m[0], m[1]])]
        
            srcNZ_m, spec_weight_m = shear.import_spec_cat(path_kidscats, kidscatname2,\
                                kidscat_end, specz_file, m_selection, \
                                cat_version)

            srcPZ_m, bins_m = np.histogram(srcNZ_m, range=[0.025, 3.5], bins=70, \
                                                weights=spec_weight_m, density=1)

            srcPZ_m = srcPZ_m/srcPZ_m.sum()
            dlsods[i] = np.mean(shear.calc_mcorr_weight(Dcllist, Dallist, \
                                Dcsbins, zlensbins, Dclbins, srcPZ_m, cat_version, Dc_epsilon, galZlist, com))
            m_selection = {}

        srcm_varlist = np.average(m_corr[:,2], weights=dlsods)
        srcm_varlist = srcm_varlist * np.ones(srcNr_varlist.shape)


    # Printing the made choices

    print()
    print('Using {} cores to create split catalogues'.format(Nsplits), \
          ', - Center definition = {}'.format(centering))
    print()

    output = 0
    outputnames = 0

    # Split up the list of KiDS fields, for parallel obsbins
    splitkidscats = np.array(shear.split(kidscats, Nsplits))

    # galaxy information
    gallists = (galIDlist, galRAlist, galDEClist, galZlist, galweightlist,
                Dallist, Dcllist, z_epsilon)
    # source information
    srclists = (srcNr_varlist, srcRA_varlist, srcDEC_varlist, w_varlist,
              e1_varlist, e2_varlist, srcm_varlist, tile_varlist, srcPZ_varlist, k_interpolated)


    galIDlist, galRAlist, galDEClist, galZlist, galweightlist, \
        Dallist, Dcllist = [], [], [], [], [], [], []
    srcNr_varlist, srcRA_varlist, srcDEC_varlist, w_varlist, \
        e1_varlist, e2_varlist, srcm_varlist, tile_varlist, srcPZ_a = [], \
                                            [], [], [], [], [], [], [], []
    gc.collect()

    if 'catalog' in purpose:
        out = loop_multi(purpose, Nsplits, Nsplits, output, outputnames, gamacat,
                 colnames, centering, gallists, lens_selection, lens_binning,
                 binname, Nobsbins, binmin, binmax, Rbins, Rcenters, Rmin, Rmax,
                 Runit, nRbins, Rconst, com, filename_var, filename, cat_version,
                 blindcat, blindcats, srclists, path_splits, splitkidscats, catmatch,
                 Dcsbins, zlensbins, Dclbins, Dc_epsilon, filename_addition, variance, h,
                path_kidscats, kidscatname, kidscolnames, kidscat_end, src_selection)

    if 'covariance' in purpose:
        for i in xrange(Nobsbins):
            
            filename_var = shear.define_filename_var(
                    purpose, centering, binname, i+1, Nobsbins, lens_selection,
                    lens_binning, src_selection, lens_weights, name_Rbins,
                    O_matter, O_lambda, Ok, h)

            binname, lens_binning, Nobsbins, binmin, binmax = \
                    shear.define_obsbins(i+1, lens_binning, lenssel_binning,
                                         gamacat)
            
            out = loop_multi(purpose, Nsplits, Nsplits, output, outputnames, gamacat,
                             colnames, centering, gallists, lens_selection, lens_binning,
                             binname, i, binmin, binmax, Rbins, Rcenters, Rmin, Rmax,
                             Runit, nRbins, Rconst, com, filename_var, filename, cat_version,
                             blindcat, blindcats, srclists, path_splits, splitkidscats, catmatch,
                             Dcsbins, zlensbins, Dclbins, Dc_epsilon, filename_addition, variance, h,
                             path_kidscats, kidscatname, kidscolnames, kidscat_end, src_selection)

    if out == 0:
        print('Splits finished properly.')
    else:
        print('Splits not finished properly, please restart the pipeline.')
        raise SystemExit()
    return 


## END OF FILE ##
