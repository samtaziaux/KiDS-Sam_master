#!/usr/bin/python

""" Create a histogram of the number of lenses from "real environments" that appear in the separate "shuffled environments" """

import pyfits
import numpy as np
import sys
import os

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['Computer Modern']})


deltanames = ['deltaR1', 'deltaR2', 'deltaR3', 'deltaR4', 'deltaR5', 'deltaR6', 'deltaR7', 'deltaR8']

mergedcatdir = '/data2/brouwer/MergedCatalogues'
gamacatname = '%s/GAMACatalogue_1.0.fits'%mergedcatdir
shuffcatname = '%s/shuffled_environment_S4_deltaR.fits'%mergedcatdir

gamacat = pyfits.open(gamacatname)[1].data
shuffcat = pyfits.open(shuffcatname)[1].data

galIDlist = gamacat['ID'] # IDs of all galaxies
galZlist = gamacat['Z'] # Central Z of the galaxy
envlist = gamacat['envS4'] # Environment of the galaxy (0 void, 1 sheet, 2 filament, 3 knot)

envnames = ['Void', 'Sheet', 'Filament', 'Knot']
envcolors = ['red', 'green', 'blue', 'orange']
envnumbers = np.arange(4)
envbins = np.append(envnumbers, 4)

#for R in np.arange(len(deltanames))+1:
for R in [4]:
    
    shuffenvlist =  shuffcat['shuffenvR%i'%R] # Shuffled environment of the galaxy (0 void, 1 sheet, 2 filament, 3 knot)
    
    print
    print 'shuffenvR%i'%(R)

    fig, ax = plt.subplots(figsize=(6,6))

    for env in envnumbers:

        plt.subplot()
        shuffenvmask = (shuffenvlist == env)
        realenvlist = envlist[shuffenvmask]

        x, histbins, patches = plt.hist(realenvlist, envbins, normed=1, label = 'Shuffled %s'%envnames[env], color = envcolors[env], linewidth = 2, histtype='step')

    ax.set_xticks(envnumbers+0.5)
    ax.set_xticklabels(envnames)

    print x

    plt.ylabel(r'P(galaxy in environment)', fontsize=15)
    plt.xlabel(r'True cosmic environment', fontsize=15)
    
    plt.legend(loc='upper right',ncol=1, prop={'size':12})
    
    plotname = 'shuffenv_plots/shuffled_proportion_histogram_S4_%s.png'%deltanames[R-1]
    plt.savefig(plotname,format='png')

    plt.show()

    print 'Written:', plotname
