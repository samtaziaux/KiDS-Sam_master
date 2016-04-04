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
import shearcode_modules as shear
from astropy import constants as const, units as u

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number

def main():

    # Input parameters
    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
            lens_selection, lens_weights, binname, Nobsbins, src_selection, \
            cat_version, path_Rbins, name_Rbins, Runit, path_output, \
            path_splits, path_results, purpose, O_matter, O_lambda, Ok, h, \
            filename_addition, Ncat, splitslist, blindcats, blindcat, \
            blindcatnum, path_kidscats, path_gamacat = shear.input_variables()

    
    if 'bootstrap' in purpose:
        print('Step 4: Combine the bootstrap samples into the ESD '\
              'profiles and bootstrap covariance matrix')
    else:
        print('Step 2: Combine the splits into the ESD profiles '\
              'and analytical covariance matrix')
    print

    # Define the list of variables for the output filename
    filename_var = shear.define_filename_var(purpose, centering, binname, \
                    'binnum', Nobsbins, lens_selection, src_selection, \
                    lens_weights, name_Rbins, O_matter, O_lambda, Ok, h)
    if ('random' or 'star') in purpose:
        filename_var = '%i_%s'%(Ncat, filename_var)
        # Ncat is the number of existing randoms
        print 'Number of existing random catalogs:', Ncat

    # Paths to the resulting files
    filename_N1 = filename_var.replace('binnum', '%i'%binnum)
    filenameESD = shear.define_filename_results(path_results, \
                    purpose, filename_N1, filename_addition, Nsplit, blindcat)


    # Printing the covariance matrix to a text file
    filename_N1 = filename_var.replace('binnumof', 's')
    filenamecov = '%s/%s_matrix_%s%s_%s.txt'%(path_results, purpose, \
                            filename_N1, filename_addition, blindcat)


    # Stop if the output already exists.
    if os.path.isfile(filenamecov):
        print 'This output already exists:', filenameESD
        print 'This output already exists:', filenamecov
        print
        quit()


    # Importing all GAMA data, and the information
    # on radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, \
    Rcenters, nRbins, Rconst, gamacat, galIDlist, galRAlist, galDEClist, \
    galweightlist, galZlist, Dcllist, Dallist = shear.import_data(path_Rbins, \
                                                Runit, path_gamacat, \
                                                path_kidscats, centering, \
                                                purpose, Ncat, O_matter, \
                                                O_lambda, Ok, h, lens_weights, \
                                                filename_addition, cat_version)

    galIDlist_matched = np.array([], dtype=np.int32)
    for kidscatname in kidscats:
        galIDlist_matched = np.append(galIDlist_matched, \
                                      catmatch[kidscatname][0])
    galIDlist_matched = np.unique(galIDlist_matched)

    # Binnning information of the groups
    lenssel_binning = shear.define_lenssel(gamacat, centering, \
                        lens_selection, 'None', 'None', 0, -inf, inf) # Mask
                        # the galaxies in the shear catalog, WITHOUT binning
                        # (for the bin creation)

    binname, lens_binning, Nobsbins, \
    binmin, binmax = shear.define_obsbins(binnum, lens_binning, \
                                          lenssel_binning, gamacat)


    # These lists will contain the final ESD profile
    if 'covariance' in purpose:
        gammat = np.zeros([Nobsbins,nRbins])
        gammax = np.zeros([Nobsbins,nRbins])
        wk2 = np.zeros([Nobsbins,nRbins])
        srcm = np.zeros([Nobsbins,nRbins])

    ESDt_tot = np.zeros([Nobsbins,nRbins])
    ESDx_tot = np.zeros([Nobsbins,nRbins])
    error_tot = np.zeros([Nobsbins,nRbins])
    bias_tot = np.zeros([Nobsbins,nRbins])

    # These lists will contain the final covariance matrix
    radius1 = np.zeros([Nobsbins,Nobsbins,nRbins,nRbins])
    radius2 = np.zeros([Nobsbins,Nobsbins,nRbins,nRbins])
    bin1 = np.zeros([Nobsbins,Nobsbins,nRbins,nRbins])
    bin2 = np.zeros([Nobsbins,Nobsbins,nRbins,nRbins])
    cov = np.zeros([Nobsbins,Nobsbins,nRbins,nRbins])
    cor = np.zeros([Nobsbins,Nobsbins,nRbins,nRbins])
    covbias = np.zeros([Nobsbins,Nobsbins,nRbins,nRbins])


    # The calculation of the covariance starts

    if 'bootstrap' in purpose: # Calculating the bootstrap covariance
        for N1 in xrange(Nobsbins):

            filename_N1 = shear.define_filename_var(purpose, centering, \
                                                    binname, N1+1, Nobsbins, \
                                                    lens_selection, \
                                                    src_selection, \
                                                    lens_weights, name_Rbins, \
                                                    O_matter, O_lambda, Ok, h)
                                                    
            if ('random' or 'star') in purpose:
                filename_N1 = '%i_%s'%(Ncat, filename_N1)
                # Ncat is the number of existing randoms
            
            filenameESD = shear.define_filename_results(path_results, purpose, \
                                                        filename_N1, \
                                                        filename_addition, \
                                                        Nsplit, blindcat)

            # Paths to the bootstrap catalog of Obsbin 1
            shearcatname_N1 = shear.define_filename_splits(path_splits, \
                                                        purpose, \
                                                        filename_N1, 0, 0, \
                                                        filename_addition, \
                                                        blindcat) # Paths to
                                                        # the resulting files
                                                           
            sheardat_N1 = pyfits.open(shearcatname_N1)[1].data

            # Import the ESD profiles from the bootstrap catalog
            ESDt_N1 = sheardat_N1['ESDt']
            ESDx_N1 = sheardat_N1['ESDx']
            error_N1 = sheardat_N1['ESD(error)']
            bias_N1 = sheardat_N1['bias']
            variance = sheardat_N1['variance(e[A,B,C,D])'][0]
            
            # Uploading the shear profile
            ESD = np.loadtxt(filenameESD).T
            
            ESDt_tot[N1] = ESD[1]
            ESDx_tot[N1] = ESD[2]
            error_tot[N1] = ESD[3]
            bias_tot[N1] = ESD[4]
            
            for N2 in xrange(Nobsbins):

                filename_N2 = shear.define_filename_var(purpose, centering, \
                                                        binname, N2+1, \
                                                        Nobsbins, \
                                                        lens_selection, \
                                                        src_selection, \
                                                        lens_weights, \
                                                        name_Rbins, \
                                                        O_matter, \
                                                        O_lambda, Ok, h)
                                                        
                if ('random' or 'star') in purpose:
                    filename_N2 = '%i_%s'%(Ncat, filename_N2)
                    # Ncat is the number of existing randoms
                
                shearcatname_N2 = shear.define_filename_splits(path_splits, \
                                                        purpose, filename_N2, \
                                                        0, 0, \
                                                        filename_addition, \
                                                        blindcat) # Paths
                                                        # to the resulting files
                                                        
                sheardat_N2 = pyfits.open(shearcatname_N2)[1].data

                # Import the ESD profiles from the bootstrap catalog of Obsbin 2
                ESDt_N2 = sheardat_N2['ESDt']
                ESDx_N2 = sheardat_N2['ESDx']
                error_N2 = sheardat_N2['ESD(error)']
                bias_N2 = sheardat_N2['bias']

                for R1 in xrange(nRbins):
                    for R2 in range(nRbins):

                        ESD_mask = (0 < error_N1[:, R1]) \
                                    &(error_N1[:, R1] < inf) \
                                    & (0 < error_N2[:, R2]) \
                                    &(error_N2[:, R2] < inf)
                                    # Mask values that do not
                                    # have finite values (inf or nan)

                        if sum(ESD_mask) > 0: # If there are any values
                                              # that are not masked
                            
                            cov[N1,N2,R1,R2] = np.cov((ESDt_N1[:, R1])[ESD_mask], \
                                        (ESDt_N2[:, R2])[ESD_mask], bias=1)[0,1]


    if 'covariance' in purpose: # Calculating the covariance

        for N1 in xrange(Nobsbins):

            filename_N1 = shear.define_filename_var(purpose, centering, \
                                                    binname, \
                                                    N1+1, Nobsbins, \
                                                    lens_selection, \
                                                    src_selection, \
                                                    lens_weights, name_Rbins, \
                                                    O_matter, O_lambda, Ok, h)
                                                    
            if ('random' or 'star') in purpose:
                filename_N1 = '%i_%s'%(Ncat, filename_N1)
                # Ncat is the number of existing randoms
            
            filenameESD = shear.define_filename_results(path_results, \
                                                        purpose, \
                                                        filename_N1, \
                                                        filename_addition, \
                                                        Nsplit, blindcat)

            for x in xrange(len(kidscats)):

               # Loading the covariance data file of each KiDS field
                shearcatname_N1 = shear.define_filename_splits(path_splits, \
                                                            purpose, \
                                                            filename_N1, \
                                                            kidscats[x], 0, \
                                                            filename_addition, \
                                                            blindcat)
               
                if os.path.isfile(shearcatname_N1):
                    print '	Combining field', x+1, '/', len(kidscats), \
                            ':', kidscats[x]

                    sheardat_N1 = pyfits.open(shearcatname_N1)[1].data
                    
                    Cs_N1 = sheardat_N1['Cs']
                    Ss_N1 = sheardat_N1['Ss']
                    Zs_N1 = sheardat_N1['Zs']
                    
                    e1 = sheardat_N1['e1'][:,blindcatnum]
                    e2 = sheardat_N1['e2'][:,blindcatnum]
                    lfweight = sheardat_N1['lfweight'][:,blindcatnum]
                    srcmlist = sheardat_N1['bias_m']
                    variance = sheardat_N1['variance(e[A,B,C,D])'][0]

                    e1 = np.reshape(e1,[len(e1),1])
                    e2 = np.reshape(e2,[len(e2),1])
                    lfweights = np.reshape(lfweight,[len(lfweight),1])
                    srcmlists = np.reshape(srcmlist,[len(srcmlist),1])

                    # Calculating the relevant quantities for each field
                    
                    # The tangential shear
                    gammat[N1] = gammat[N1] + \
                                    sum(lfweights*(Cs_N1*e1+Ss_N1*e2),0)
                     # The cross shear
                    gammax[N1] = gammax[N1] + \
                                    sum(lfweights*(-Ss_N1*e1+Cs_N1*e2),0)
                    wk2[N1] = wk2[N1] + sum(lfweights*Zs_N1,0)
                    # The total weight (lensfit weight + lensing efficiency)
                    srcm[N1] = srcm[N1] + sum(lfweights*Zs_N1*srcmlists,0)
                    
                    for N2 in xrange(Nobsbins):
                        
                        filename_N2 = shear.define_filename_var(purpose, \
                                                            centering, \
                                                            binname, \
                                                            N2+1, Nobsbins, \
                                                            lens_selection, \
                                                            src_selection, \
                                                            lens_weights, \
                                                            name_Rbins, \
                                                            O_matter, \
                                                            O_lambda, Ok, h)
                                                                
                        if ('random' or 'star') in purpose:
                            filename_N2 = '%i_%s'%(Ncat, filename_N2)
                            # Ncat is the number of existing randoms

                        shearcatname_N2 = shear.define_filename_splits(path_splits, \
                                                            purpose, \
                                                            filename_N2, \
                                                            kidscats[x], 0,\
                                                            filename_addition,\
                                                            blindcat)

                        sheardat_N2 = pyfits.open(shearcatname_N2)[1].data

                        # Importing the relevant data from each file
                        Cs_N2 = np.array(sheardat_N2['Cs'])
                        Ss_N2 = np.array(sheardat_N2['Ss'])
                        Zs_N2 = np.array(sheardat_N2['Zs'])
                    
                        
                        for R1 in xrange(nRbins):
                            for R2 in range(nRbins):

                                if 'covariance' in purpose:
                                    cov[N1,N2,R1,R2] = cov[N1,N2,R1,R2] + \
                                    sum(variance[blindcatnum]*(lfweight**2)* \
                                    (Cs_N1[:,R1]*Cs_N2[:,R2]+Ss_N1[:,R1]* \
                                     Ss_N2[:,R2])) # The new covariance matrix
                else:
                    print('ERROR: Not all fields are analysed! '\
                                  'Please restart shear code!')
                    quit()
            
            # Calculating the final output values of the accompanying shear data
            ESDt_tot[N1], ESDx_tot[N1], error_tot[N1], \
            bias_tot[N1] = shear.calc_stack(gammat[N1], \
                                            gammax[N1], \
                                            wk2[N1], \
                                            np.diagonal(cov[N1,N1,:,:]), \
                                            srcm[N1], [1,1,1,1], blindcatnum)
            
            # Determine the stacked galIDs
            binname, lens_binning, Nobsbins, \
            binmin, binmax = shear.define_obsbins(N1+1, lens_binning, \
                                                lenssel_binning, gamacat)

            lenssel = shear.define_lenssel(gamacat, centering, \
                                           lens_selection, lens_binning, \
                                           binname, N1+1, binmin, binmax)

            galIDs = galIDlist[lenssel] # Mask all quantities
            galIDs_matched = galIDs[np.in1d(galIDs, galIDlist_matched)]
            galIDs_matched_infield = galIDs[np.in1d(galIDs, galIDs_infield)]

            print 'galIDs_matched:', len(galIDs_matched)

            # Printing stacked shear profile to a text file
            shear.write_stack(filenameESD, Rcenters, Runit, \
                              ESDt_tot[N1], ESDx_tot[N1], \
                              error_tot[N1], bias_tot[N1], h, \
                              variance, wk2[N1], np.diagonal(cov[N1,N1,:,:]), \
                              blindcat, blindcats, blindcatnum, \
                              galIDs_matched, galIDs_matched_infield)

        for N1 in xrange(Nobsbins):
            for N2 in xrange(Nobsbins):
                for R1 in xrange(nRbins):
                    for R2 in xrange(nRbins):
                        cov[N1,N2,R1,R2] = cov[N1,N2,R1,R2]/ \
                                            (wk2[N1,R1]*wk2[N2,R2])
                        # The covariance matrix


    with open(filenamecov, 'w') as file:
        print >>file, '#', '%s_min[m]'%binname, '	', '%s_min[n]'%binname,'	','Radius[i](kpc/h%g)'%(h*100), '	','Radius[j](kpc/h%g)'%(h*100), '   ','covariance(h%g*M_sun/pc^2)^2'%(h*100), '	', 'correlation','	', 'bias(1+K[m,i])(1+K[n,j])'

    with open(filenamecov, 'a') as file:

        # Calculating the correlation
        for N1 in xrange(Nobsbins):
            for N2 in xrange(Nobsbins):
                for R1 in xrange(nRbins):
                    for R2 in xrange(nRbins):
                        
                        radius1[N1,N2,R1,R2] = Rcenters[R1]
                        radius2[N1,N2,R1,R2] = Rcenters[R2]
                        bin1[N1,N2,R1,R2] = (lens_binning.values()[0])[1][N1]
                        bin2[N1,N2,R1,R2] = (lens_binning.values()[0])[1][N2]
                        
                        if (0. < error_tot[N1,R1])&(error_tot[N1,R1] < inf) \
                            and (0. < error_tot[N2,R2])&(error_tot[N2,R2] < inf):
                            
                            cor[N1,N2,R1,R2] = cov[N1,N2,R1,R2]/ \
                                    ((cov[N1,N1,R1,R1]*cov[N2,N2,R2,R2])**0.5)
                            
                            covbias[N1,N2,R1,R2] = bias_tot[N1,R1]*bias_tot[N2,R2]
                            
                        else:
                            cov[N1,N2,R1,R2] = -999
                            cor[N1,N2,R1,R2] = -999
                            covbias[N1,N2,R1,R2] = -999

                        print >>file, bin1[N1,N2,R1,R2], '	',bin2[N1,N2,R1,R2], '	',radius1[N1,N2,R1,R2], '	',radius2[N1,N2,R1,R2], '	',cov[N1,N2,R1,R2], '	',cor[N1,N2,R1,R2], '	',covbias[N1,N2,R1,R2]

    print 'Written: Covariance matrix data:', filenamecov

    return

main()
