#!/usr/bin/python


import pyfits
import numpy as np
import sys
import os

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams


# Importing mcmc results
def import_mcmc(filename, burn):

    mcmcname = '%s.fits'%filename
    headername = '%s.hdr'%filename
    
    print 'Importing MCMC results:', mcmcname
    print '                   and:', headername
    mcmccat = pyfits.open(mcmcname)[1]
    mcmcnames = mcmccat.columns.names
    mcmcdata = mcmccat.data
    
    mcmcmask = (mcmcdata['Mcen1'] != 0)
    end = np.sum(mcmcmask)
    
    mcmc = dict()
    for colname in mcmcnames:
        mcmc[colname] = (mcmcdata[colname])[burn:end]

    return mcmc, end, mcmcname, headername, mcmcmask
    
    
# Import header file
def read_header(headername):

    inputparams = dict()

    header = open(headername, 'r').read().split('\n')
    header = [header[i].split(' ') for i in xrange(len(header))]

    for line in header:
        if len(line)>1:
#            print line
            if line[0] == 'datafile':
                esdfiles = line[1].split(',')
            elif line[0] == 'covfile':
                covfile = line[1]
            elif 'fixed' in line:
                inputvalues = line[4].split(',')
                inputvalues = [float(inputvalues[p]) for p in xrange(len(inputvalues))]
                inputparams[line[0]] = inputvalues
    
    return esdfiles, covfile, inputparams


# Importing the ESD profiles
def read_esdfiles(esdfiles):
    
    data = np.loadtxt(esdfiles[0]).T
    data_x = data[0]

    data_y = np.zeros(len(data_x))
    error_h = np.zeros(len(data_x))
    error_l = np.zeros(len(data_x))
    
    print 'Imported ESD profiles: %i'%len(esdfiles)
    
    for f in xrange(len(esdfiles)):
        # Load the text file containing the stacked profile
        data = np.loadtxt(esdfiles[f]).T
    
        bias = data[4]
        bias[bias==-999] = 1
    
        datax = data[0]
        datay = data[1]/bias
        datay[datay==-999] = np.nan
    
        errorh = (data[3])/bias # covariance error
        errorl = (data[3])/bias # covariance error
        errorh[errorh==-999] = np.nan
        errorl[errorl==-999] = np.nan
        
        
        data_y = np.vstack([data_y, datay])
        error_h = np.vstack([error_h, errorh])
        error_l = np.vstack([error_l, errorl])
        
    data_y = np.delete(data_y, 0, 0)
    error_h = np.delete(error_h, 0, 0)
    error_l = np.delete(error_l, 0, 0)
    
    return data_x, data_y, error_h, error_l


# Taking the median result of all samples after the burn-in phase
def calc_esds_masses(inputparams, mcmc, esdnames, massnames, envnames):
        
    # Calculating the satellite fraction
    fsat = inputparams['fsat']
    
    esd_fracs = np.array([[1, 1-fsat[esd], fsat[esd], fsat[esd]] for esd in xrange(len(esdnames))])
    
    print 'ESD fractions:'
    print esd_fracs
    print
    
    # Creating the full names of the masses
    masses = np.array([['%s%i'%(m, env+1) \
                        for env in xrange(len(envnames))] \
                        for m in massnames])
   
    # Creating the full names of the ESDs                        
    esds = np.array([['%s%i'%(esd, env+1) \
                    for env in xrange(len(envnames))] \
                    for esd in esdnames])
    
    
    #"""
    # Calculating the median ESD fits
    masslist = mcmc['Mavg1']
    index = len(masslist)/2
    sorts = np.argsort(masslist)

    #"""
    
    """
    # Selecting the result with the lowest chi2
    chi2list = mcmc['chi2']
    index = np.where(chi2list == np.amin(chi2list))[0][-1]
    esds_med = np.array([[(mcmc[esds[e,i]])[index] \
                        for e in xrange(len(esdnames))] \
                        for i in xrange(len(envnames))])
    """
        
    esds_med = np.array((([[(mcmc[esds[e,i]][sorts])[index] \
                            for e in xrange(len(esdnames))] \
                            for i in xrange(len(envnames))])))

    # Calculating the median masses
    masses_med = np.array([[(mcmc[masses[m, env]])[index] \
                           for env in xrange(len(envnames))] \
                           for m in xrange(len(massnames))])


    r=1
    print 'Median ESDs for radial bin %i:'%r
    tot = esds_med[0,:,r]
    print tot
    print
    print 'Weighted ESDs for radial bin %i:'%r
    tot = esd_fracs[0]*esds_med[0,:,r]
    print tot
    print
    print 'Total ESD:'
    print np.sum(tot[1:])

    return fsat, esd_fracs, esds_med, masses_med, esds, masses


# Create any histogram for the four environments
def create_histogram(obsname, obslist, nbins, envnames, envlist, style, norm, weightlist):

    obsmin = np.amin(obslist)
    obsmax = np.amax(obslist)

    if 'log' in style:    
        if obsmin <= 0.:
            obsmin = np.amin(obslist[obslist > 0.])
        bins = np.logspace(np.log10(obsmin), np.log10(obsmax), nbins)
        bins = np.append(bins, obsmax)
        
        plt.xscale('log')
        plt.yscale('log')

    if 'lin' in style:
        bins = np.arange(obsmin, obsmax, abs(obsmax-obsmin)/nbins)
        bins = np.append(bins, obsmax)
    
    histvals, histbins = np.histogram(obslist, bins)
    hists = np.zeros([len(envnames), nbins])
    
    for env in xrange(len(envnames)):
        envmask = (envlist == env)
       
        if type(weightlist) == np.ndarray:
            weight = weightlist[envmask]
            print 'The weights are:', weight
        else:
            weight = np.ones(len(obslist[envmask]))
       
        histvals, histbins = np.histogram(obslist[envmask], histbins, normed = norm, weights = weight)
        histcens = np.array([(histbins[i]+histbins[i+1])/2 for i in xrange(len(histvals))])
        hists[env] = histvals
 
        # Show results
        plt.plot(histcens, histvals, '', ls='-', label=envnames[env])
    
    plt.legend(fontsize=13, loc='best')
    
    plt.xlabel(obsname, size=15)
    
    #plt.show()
    plt.clf()
    
    return histbins, hists, histcens


# Calculating average redshift, log(M*) and satellite fraction of the lens samples (needed for halo model)
def calc_halomodel_input(envnames, envlist, ranklist, zlist, mstarlist, weightlist):
    
    if type(weightlist) != np.ndarray:
        weightlist = np.ones(len(zlist))
    
    zaverage = []
    mstaraverage = []
    fsatmin = []
    fsatmax = []
    
    for env in xrange(len(envnames)):
        
        envmask = (envlist == env)
        
        zaverage = np.append(zaverage, np.average(zlist[envmask], weights=weightlist[envmask]))
        mstaraverage = np.append(mstaraverage, np.average(mstarlist[envmask], weights=weightlist[envmask]))
    
        satmask = (ranklist >= 2)
        cenmask = (ranklist == 1)
        isomask = (ranklist == -999.)
        
        lenses = float(sum(np.ones(len(ranklist[envmask]))*weightlist[envmask]))
        sats = float(sum(np.ones(len(ranklist[satmask*envmask]))*weightlist[satmask*envmask]))
        cens = float(sum(np.ones(len(ranklist[cenmask*envmask]))*weightlist[cenmask*envmask]))
        isos = float(sum(np.ones(len(ranklist[isomask*envmask]))*weightlist[isomask*envmask]))
        
        fsatmin = np.append(fsatmin, sats/lenses)
        fsatmax = np.append(fsatmax, (sats+isos)/lenses)
    
    obsnames = ['average(Z)', 'average(M*)', 'minimum fsat', 'maximum fsat']
    obs = np.array([zaverage, mstaraverage, fsatmin, fsatmax])
        
    print '              ', envnames
    for o in xrange(len(obs)):
        printline = '%s: '%obsnames[o]
        
        for e in xrange(len(envnames)):
            printline = '%s%s,'%(printline, obs[o, e])
        print printline
        
    return zaverage, mstaraverage, fsatmin, fsatmax

    
# Write the results to a fits catalogue
def write_catalog(filename, galIDlist, outputnames, output):

    fitscols = []

    # Adding the lens IDs
    fitscols.append(pyfits.Column(name = 'ID', format='J', array = galIDlist))

    # Adding the output
    [fitscols.append(pyfits.Column(name = outputnames[c], format = '1D', array = output[c])) for c in xrange(len(outputnames))]

    cols = pyfits.ColDefs(fitscols)
    tbhdu = pyfits.new_table(cols)

    #	print
    if os.path.isfile(filename):
        os.remove(filename)
        print 'Old catalog overwritten:', filename
    else:
        print 'New catalog written:', filename
    print

    tbhdu.writeto(filename)


# Printing data to a text file
def write_textfile(filename, datanames, data):

    # Create the file with header    
    with open(filename, 'w') as file:
        
        printline = '# '
        for n in xrange(len(datanames)):
            printline = '%s	%s'%(printline, datanames[n])
        print >>file, printline

    # Write the data to the file
    with open(filename, 'a') as file:
        
        for i in xrange(len(data[0])): # For each element ...
            printline = ''
            for n in xrange(len(datanames)): # ... in each row:
                printline = '%s	%s'%(printline, data[n, i]) # Put it in the row ...
            
            print >>file, printline # ... and print it
    
    print 'Written:', filename
