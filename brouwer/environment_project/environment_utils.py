#!/usr/bin/python


import pyfits
import numpy as np
import sys
import os

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams




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
