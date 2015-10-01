#!/usr/bin/python

# Import the necessary libraries
import pyfits
import numpy as np
import distance
import sys
import os
import time
import shearcode_modules as shear
from astropy import constants as const, units as u
import numpy.core._dotblas

from matplotlib import pylab as plt
from matplotlib import rc, rcParams


# Plotting the data for the separate observable bins
plotstyle = 'log' # What plotting style is used (lin, log, errorlin, errorlog)
subplots = False # Are there subplots?
Nrows = 1 # If so, how into many rows will the subplots be devided?
extra = '' # difference/ratio as an extra plot

h = 1

xlabel = r'radius R [h$^{-1}$ kpc]'
ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h M$_{\odot}/pc^2$]'

filenames = []
#filenames = np.append(filenames, [''])


# Pipeline tests
filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG-999--999_logmstarbin1of8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_D.txt'])
filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/pipeline_testresults/output_logmstarbins_oldcatmatch/results_shearcatalog/shearcatalog_logmstarbin1of8_RankBCGm999_Z_B0p005-1p2_Rbins10-20-2000kpc_Om0p315_Ol0p685_Ok0_h1_oldcatmatch_D.txt'])
filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/pipeline_testresults/output_logmstarbins_oldcatmatch/results_shearcovariance/shearcovariance_logmstarbin1of8_RankBCGm999_Z_B0p005-1p2_Rbins10-20-2000kpc_Om0p315_Ol0p685_Ok0_h1_oldcatmatch_D.txt'])


"""
filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_LumBbins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG1-1_Nfof5-inf_LumBbin1of6_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_D.txt'])
filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/pipeline_testresults/output_LumBbins_oldcatmatch/results_shearcatalog/shearcatalog_LumBbin1of6_Nfof5-inf_RankBCG1_Z_B0p005-1p2_Rbins10-20-2000kpc_Om0p315_Ol0p685_Ok0_h1_oldcatmatch_D.txt'])
"""

# Environment
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_envS8bins/results_shearcatalog/shearcatalog_rankBCG-999-inf_envS8bin1of4_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_envS8bins/results_shearcatalog/shearcatalog_rankBCG-999-inf_envS8bin2of4_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_envS8bins/results_shearcatalog/shearcatalog_rankBCG-999-inf_envS8bin3of4_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_envS8bins/results_shearcatalog/shearcatalog_rankBCG-999-inf_envS8bin4of4_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])

# 10 vs 16 bins
# Cristobal

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_logRbins12:20:2000kpc_Om0.3_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_logRbins12:20:2000kpc_Om0.3_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins_R10000/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_logRbins12:20:2000kpc_Om0.3_h100_R10000_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_logRbins20:10:10000kpc_Om0.3_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_shearcatalog/shearcatalog_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_logRbins20:10:10000kpc_Om0.3_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_20bins_Om0.3_h100_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_logRbins20:10:10000kpc_Om0.3_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_shearcatalog/shearcatalog_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_logRbins20:10:10000kpc_Om0.3_h100_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_shearcatalog/shearcatalog_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_AngSepBCGbins/results_shearcatalog/shearcatalog_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of1_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])

#Groups

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearcatalog/shearcatalog_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearcatalog/shearcatalog_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])


#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])


# Edo

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG-999-inf_logmstarbin7of8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_oldcatmatch/results_covariance/covariance_rankBCG-999-inf_logmstarbin5of8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG-999-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins_KiDSv0.5/results_shearcatalog/shearcatalog_rankBCG-999-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_KiDSv0.5_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG-999-inf_logmstarbin5of8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_KiDSv0.5/results_shearcatalog/shearcatalog_rankBCG-999-inf_logmstarbin5of8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_KiDSv0.5_A.txt'])

# Aaron

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearcatalog/shearcatalog_rankBCG-999-inf_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer/Aaron/output_Nobins/results_shearcatalog/shearcatalog_rankBCG-999-inf_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])

"""
filenames1 = []
for i in np.arange(1,9):
    filenames1 = np.append(filenames1, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_KiDSv0.5/results_shearcatalog/shearcatalog_rankBCG-999-inf_logmstarbin%iof8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_KiDSv0.5_A.txt'%i])


Nobsbins = 8

filenames1 = []
for i in np.arange(1, Nobsbins+1):
    filenames1 = np.append(filenames1, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG2-inf_Nfof5-inf_logmstarbin%iof8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_A.txt'%i])

filenames2 = []
for i in np.arange(1, Nobsbins+1):
    filenames2 = np.append(filenames2, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_oldcatmatch/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_logmstarbin%iof8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_A.txt'%i])


filenames3 = []
for i in np.arange(1, Nobsbins+1):
    filenames3 = np.append(filenames3, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins_oldcatmatch_fluxscale+h-corr/results_shearcatalog/shearcatalog_rankBCG-999-inf_logmstarbin%iof8_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_oldcatmatch_fluxscale+h-corr_A.txt'%i])



Nobsbins = 3

filenames1 = []
for i in np.arange(1, Nobsbins+1):
    filenames1 = np.append(filenames1, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_corr-AngSepBCGbins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG2-inf_Nfof5-inf_corr-AngSepBCGbin%iof3_ZB0.005-1.2_8bins_Om0.315_h100_oldcatmatch_A.txt'%i])

filenames2 = []
for i in np.arange(1, Nobsbins+1):
    filenames2 = np.append(filenames2, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_corr-AngSepBCGbins_oldcatmatch/results_shearcatalog/shearcatalog_rankBCG2-inf_Nfof5-inf_corr-AngSepBCGbin%iof3_ZB0.005-1.2_logRbins12:20:2000kpc_Om0.315_h100_oldcatmatch_A.txt'%i])
"""

filelists = np.vstack([filenames])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins/results_shearcatalog/shearcatalog_rankBCG-999-inf_logmstarbin4of7_ZB0.005-1_logRbins12:30:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins/results_covariance/covariance_rankBCG-999-inf_logmstarbin4of7_ZB0.005-1_logRbins12:30:2000kpc_Om0.315_h100_A.txt'])

# Covariance tests
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins_lenssplits10/results_shearcatalog/shearcatalog_rankBCG1-inf_Nfof2-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_lenssplits10_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearcatalog/shearcatalog_rankBCG1-inf_Nfof2-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearcatalog/shearcatalog_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearbootstrap/shearbootstrap_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins_test/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_test_A.txt'])
#filenames = np.append(filenames, ['/data2/brouwer/shearprofile/output_Nobins/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearcatalog/shearcatalog_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearbootstrap/shearbootstrap_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins_test/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_16bins_Om0.315_h100_test_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins/results_covariance/covariance_rankBCG-999-inf_logmstarbin1of1_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_logmstarbins/results_shearcatalog/shearcatalog_rankBCG-999-inf_logmstarbin1of1_ZB0.005-1.2_16bins_Om0.315_h100_A.txt'])

#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_shearcatalog/shearcatalog_rankBCG1-inf_Nfof2-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['/disks/shear10/brouwer_veersemeer/shearcode_output/output_Nobins/results_covariance/covariance_rankBCG1-inf_Nfof2-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])


#filenames = np.append(filenames, ['../../Cristobal/output_AngSepBCGbins/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin1of3_ZB0.005-1.2_logRbins12:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['../../Cristobal/output_AngSepBCGbins/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin2of3_ZB0.005-1.2_logRbins12:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['../../Cristobal/output_AngSepBCGbins/results_shearcatalog/shearcatalog_rankBCG2-inf_Nfof5-inf_AngSepBCGbin2of3_ZB0.005-1.2_logRbins12:20:2000kpc_Om0.315_h100_A.txt'])

#filenames = np.append(filenames, ['../../Cristobal/output_AngSepBCGbins/results_covariance/covariance_rankBCG2-inf_Nfof5-inf_AngSepBCGbin3of3_ZB0.005-1.2_logRbins12:20:2000kpc_Om0.315_h100_A.txt'])


#filenames = np.append(filenames, ['../../shearcode_output/Edo/output_logmstarbins/results_covariance/covariance_rankBCG-999-inf_logmstarbin7of7_ZB0.005-1_logRbins12:30:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['../../shearcode_output/Edo/output_logmstarbins/results_shearcatalog/shearcatalog_rankBCG-999-inf_logmstarbin7of7_ZB0.005-1_logRbins12:30:2000kpc_Om0.315_h100_A.txt'])

# Bootrap error difference
#filenames = np.append(filenames, ['shear_output/results_bootstrap/bootstrap_ESD_rankIterCen1-1_Nfof5-inf_LumBbin1of6_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A_difference.txt'])
#filenames = np.append(filenames, ['shear_output/results_bootstrap/bootstrap_ESD_rankIterCen1-1_Nfof5-inf_LumBbin2of6_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A_difference.txt'])
#filenames = np.append(filenames, ['shear_output/results_bootstrap/bootstrap_ESD_rankIterCen1-1_Nfof5-inf_LumBbin3of6_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A_difference.txt'])
#filenames = np.append(filenames, ['shear_output/results_bootstrap/bootstrap_ESD_rankIterCen1-1_Nfof5-inf_LumBbin4of6_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A_difference.txt'])
#filenames = np.append(filenames, ['shear_output/results_bootstrap/bootstrap_ESD_rankIterCen1-1_Nfof5-inf_LumBbin5of6_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A_difference.txt'])
#filenames = np.append(filenames, ['shear_output/results_bootstrap/bootstrap_ESD_rankIterCen1-1_Nfof5-inf_LumBbin6of6_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A_difference.txt'])

# Normal vs bootstrap
#filenames = np.append(filenames, ['groups_Nfof5-inf/results_shearbootstrap/shearbootstrap_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins21_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['groups_Nfof5-inf/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins21_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['groups_Nfof5-inf/results_shearcatalog/shearcatalog_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins21_Om0.315_h100_A.txt'])

# Centrals vs satellites
#filenames = np.append(filenames, ['shearcode_output/results_randombootstrap/randombootstrap_26_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['shearcode_output/results_shearbootstrap/shearbootstrap_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['shearcode_output/results_shearbootstrap/shearbootstrap_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A_error.txt'])

# Random vs real
#filenames = np.append(filenames, ['shearcode_output/results_randomcatalog/randomcatalog_26_rankBCG1-inf_Nfof2-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['shearcode_output/results_shearcatalog/shearcatalog_rankBCG1-1_Nfof5-inf_ZB0.005-0.9_21bins_Om0.315_h100_A.txt'])

#filenames = np.append(filenames, ['shearcode_output/results_randombootstrap/randombootstrap_ESD_20_rankBCG1-inf_Nfof2-inf_ZB0.005-0.9_21bins_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['shearcode_output/results_randomcatalog/randomcatalog_20_rankBCG1-1_Nfof5-inf_ZB0.005-0.9_21bins_Om0.315_h100_A.txt'])

# Manmask vs no manmask
#filenames = np.append(filenames, ['shearcode_output/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'])
#filenames = np.append(filenames, ['shearcode_output/results_covariance/covariance_rankBCG1-1_Nfof5-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_no-manmask_A.txt'])
#filenames = np.append(filenames, ['shearcode_output/results_covariance/covariance_rankBCG-999-inf_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_manmask1_A.txt'])

Nfilelists = len(filelists)
Nfiles = len(filenames)


if extra == 'difference' or extra == 'ratio':
    for i in xrange(Nfiles/2):

        filename1 = filenames[i*2+0]
        filename2 = filenames[i*2+1]
        filename3 = filenames[i*2+2]
        
        file_ext = filename1.split('.')[-1]
        filename_difference = filename1.replace('.%s'%file_ext,'_%s.%s'%(extra, file_ext))
        
        # Load the text file containing the stacked profile
        data1 = np.loadtxt(filename1).T
        data2 = np.loadtxt(filename2).T

    # These values do not change
        Rcenters = data1[0]
        bias_tot = data1[4]
        variance = [data1[5,0],0,0,0]

    # Change
        data1_ESDt = data1[1]
        data2_ESDt = data2[1]
        
        data1_ESDx = data1[2]
        data2_ESDx = data2[2]
        
        data1_error = data1[3]
        data2_error = data2[3]

        if extra == 'difference':
            ESDt_tot = abs((data1_ESDt - data2_ESDt)/((data1_ESDt + data2_ESDt)/2))
            ESDx_tot = abs((data1_ESDx - data2_ESDx)/((data1_ESDx + data2_ESDx)/2))
            error_tot = abs((data1_error - data2_error)/((data1_error + data2_error)/2))

        if extra == 'ratio':
            ESDt_tot = (data1_ESDt/data2_ESDt)
            ESDx_tot = (data1_ESDx/data2_ESDx)
            error_tot = (data1_error/data2_error)

        ESDt_tot[data1_ESDt==-999.] = -999.
        ESDt_tot[data2_ESDt==-999.] = -999.
        
        ESDx_tot[data1_ESDx==-999.] = -999.
        ESDx_tot[data2_ESDx==-999.] = -999.

        error_tot[data1_error==-999.] = -999.
        error_tot[data2_error==-999.] = -999.

        shear.write_stack(filename_difference, Rcenters, ESDt_tot, ESDx_tot, error_tot, bias_tot, h, variance, 0, [], []) # Printing stacked shear profile to a file

        filenames = np.append(filenames, [filename_difference])

for i in xrange(Nfiles):

    n = 0
    subplots = False
    if subplots:
        subplots = Nfiles
        n = i+1
    
#	n = n+1
    filename = filenames[i]

    plotlabel = filename.split('/')[-1]
    plotlabel = (plotlabel.rsplit('.',1)[-2])
    plotlabel = np.array(plotlabel.split('_'))
    plotlabel = ' '.join(np.hstack([plotlabel[0:4], plotlabel[-2]]))
    plotlabel = r'%s'%plotlabel
    plottitle = ''


    shear.define_plot(filename, plotlabel, plottitle, plotstyle, subplots, xlabel, ylabel, n)

# Plot the ESD profile into a file
file_ext = filename.split('.')[-1]
plotname = filename.replace('.%s'%file_ext, '_combined.png')

# Writing and showing the plot
shear.write_plot(plotname, plotstyle)
