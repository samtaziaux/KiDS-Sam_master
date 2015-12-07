#!/usr/bin/python

"This modules will compute the 2-halo term of the halo model."

import pyfits
import numpy as np
import sys
import os
from astropy import constants as const, units as u
import glob
import gc
import subprocess as sub
import shlex


sys.path.insert(0, '../environment_project/')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../halomodel/')
sys.path.insert(0, '../../esd_production/')

import twohalo_mm
import environment_utils as envutils
import esd_utils
import shearcode_modules as shear

inf = np.inf
sigma_8 = 0.829
omegab_h2 = 0.02205
n = 0.9603

# Read the config file
def read_config(dataname, blindcat, binnum):

    lensIDname = dataname.replace('_%s.'%blindcat, '_lensIDs.')

    data = np.loadtxt(dataname).T
    R = data[0]
    lensIDs = np.loadtxt(lensIDname)
    
    print lensIDname
    
    return data, R, lensIDname, lensIDs


def import_gamacat(gamacat, lens_binning, binname, centering, rankmin, rankmax):

    # Importing angular seperation
    angseplist = gamacat['AngSep%s'%centering]
    angseplist[angseplist<=0] = 0.

    # Importing and correcting log(Mstar)
    logmstarlist = gamacat['logmstar']
    fluxscalelist = gamacat['fluxscale'] # Fluxscale, needed for stellar mass correction
    corr_list = np.log10(fluxscalelist)# - 2*np.log10(h/0.7)
    logmstarlist = logmstarlist + corr_list
    mstarlist = 10**logmstarlist
    nQlist = gamacat['nQ']

    ranklist = gamacat['rank%s'%centering]

    if 'envS4' in binname:
        # Importing the real environment
        envlist = gamacat['envS4']
    else:
        # Importing the shuffled environment
        shuffledcatname = lens_binning[binname][0]
        print shuffledcatname
        
        shuffledcat = pyfits.open(shuffledcatname)[1].data
        envlist = shuffledcat[lens_binning.keys()[0]]
        
    # Applying a mask to the galaxies
    obsmask = (fluxscalelist<500)&(logmstarlist>5) & (0 <= envlist)&(envlist < 4) & \
                                                        (rankmin <= ranklist)&(ranklist < rankmax) & (nQlist >= 3.)

    return angseplist, mstarlist, ranklist, envlist, obsmask
    
def calc_angsephist(angseplist, ranklist, galweightlist, binname, centering, rankmin, rankmax, envnames, envlist):

    # Creating and printing the angular seperation histogram (needed for the halo model)

    nbins = 100
    angsepmask = ranklist>1
    path_results = '../environment_project/results'
    
    filename = '%s/Angsephist_%s_rank%g-%g.txt'%(path_results, binname, rankmin, rankmax)

    datanames = ['AngSep%s center'%centering, 'Ngals (Voids)', 'Ngals (Sheets)', 'Ngals (Filaments)', 'Ngals (Knots)']

    if rankmax <= 2:
        data = np.ones([5, nbins])
    else:
        weight = galweightlist[angsepmask]
        
        angsepbins, angsephists, angsephistcens = \
        envutils.create_histogram('Angular separation (kpc)', \
                                  angseplist[angsepmask], nbins, envnames, envlist[angsepmask], 'log', False, weight, False)
        
        # Printing the angular separation histogram to a file
        data = np.vstack([angsephistcens, angsephists])
    
    envutils.write_textfile(filename, datanames, data)
    
    return filename
    
    

def main():
    
    config_file = str(sys.argv[1]) # The path to the configuration file
    envnames = ['Void', 'Sheet', 'Filament', 'Knot']
    
    findlist = np.array([])
    replacelist = np.array([])

    
    # Input parameters
    Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, lens_selection, \
            lens_weights, binname, Nobsbins, src_selection, path_Rbins, name_Rbins, Runit, \
            path_output, path_splits, path_results, purpose, O_matter, O_lambda, Ok, h, \
            filename_addition, Ncat, splitslist, blindcats, blindcat, blindcatnum, \
            path_kidscats, path_gamacat = shear.input_variables()

    if centering == 'None':
        centering = 'BCG'
        rankmin = -999
        rankmax = inf
    else:
        rankmin = lens_selection['rank%s'%centering][1][0]
        rankmax = lens_selection['rank%s'%centering][1][1]

    # Importing all GAMA data, and the information on radial bins and lens-field matching.
    catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, Rcenters, nRbins, Rconst, \
    gamacat, galIDlist, galRAlist, galDEClist, galweightlist, galZlist, Dcllist, Dallist = \
    shear.import_data(path_Rbins, Runit, path_gamacat, path_kidscats, centering, \
    purpose, Ncat, O_matter, O_lambda, Ok, h, lens_weights, filename_addition)

    # Importing lists that are used for the halomodel input (mstar, z, AngSep, env, rank)
    angseplist, mstarlist, ranklist, envlist, obsmask = import_gamacat(gamacat, lens_binning, binname, centering, rankmin, rankmax)


    # Paths to the data files that should be fitted
    filename_var = shear.define_filename_var(purpose, centering, binname, \
        '*', Nobsbins, lens_selection, src_selection, lens_weights, name_Rbins, O_matter, O_lambda, Ok, h)
    datafilename = shear.define_filename_results(path_results, purpose, filename_var, filename_addition, Nsplit, blindcat)
    filename_N1 = filename_var.replace('*of', 's')
    covfilename = '%s/%s_matrix_%s%s_%s.txt'%(path_results, purpose, filename_N1, filename_addition, blindcat)

    # Adding the filenames of the data to the config file
    findlist = np.append(findlist, ['datafilename', 'covfilename'])
    replacelist = np.append(replacelist, [datafilename, covfilename])
    print 'Data filename:', datafilename
    print 'Covariance filename:', covfilename
    print
    
    # Calculating average redshift, log(M*) and satellite fraction of the lens samples (needed for halo model)
    zaverage, mstaraverage, fsatmin, fsatmax = envutils.calc_halomodel_input(envnames, envlist, ranklist, galZlist, mstarlist, galweightlist)
    
    obsnames = np.array(['fsatvalues', 'zgalvalues', 'Mstarvalues'])
    obs = np.array([fsatmin, zaverage, mstaraverage])
    
    for o in xrange(len(obs)):
        findlist = np.append(findlist, obsnames[o])
        printline = ''
        for e in xrange(len(envnames)):
            printline = '%s%s,'%(printline, obs[o, e])
        printline = printline.rsplit(',', 1)[0]
        replacelist = np.append(replacelist, printline)

    # Creating the Angular Separation histogram for the halo model
    Rsatfilename = calc_angsephist(angseplist, ranklist, galweightlist, binname, centering, rankmin, rankmax, envnames, envlist)
    findlist = np.append(findlist, 'Rsatfilename')
    replacelist = np.append(replacelist, Rsatfilename)

    # Creating the two halo term
    dsigmas = np.zeros([len(zaverage), nRbins])
    for i in xrange(len(zaverage)):
        
        dsigma = twohalo_mm.dsigma_mm(sigma_8, h, omegab_h2, O_matter, O_lambda, n, zaverage[i], Rcenters/1.e3)[0]
        print 'dSigma:', dsigma, 'at z =', zaverage[i]
        dsigmas[i] = dsigma
    
    path_results = '../environment_project/results'
    twohalofilename = '%s/twohalo_%s_rank%g-%g.txt'%(path_results, binname, rankmin, rankmax)
    
    envutils.write_textfile(twohalofilename, envnames, dsigmas)
    findlist = np.append(findlist, 'twohalofilename')
    replacelist = np.append(replacelist, twohalofilename)
    
    # Adding the calculated input to the config file
    newconfigname = '%s_rank%g-%g'%(binname, rankmin, rankmax)
    
    config_files = envutils.create_config(config_file, findlist, replacelist, newconfigname)

    ps = []
    codename = '../../kids-ggl.py'
    runname = 'python %s'%codename
    runname += ' -c %s --sampler'%config_files[0]
    print
    print 'Running: %s'%runname

    p = sub.Popen(shlex.split(runname))
    ps.append(p)
    for p in ps:
        p.wait()
    
    return
    
main()


