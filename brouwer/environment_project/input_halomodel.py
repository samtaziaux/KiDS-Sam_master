
#!/usr/bin/python

import numpy as np
import pyfits
import os
from matplotlib import pyplot as plt
import delta_r_utils as utils
import environment_utils as envutils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['Computer Modern']})


centering = 'BCG'
envnames = ['Void', 'Sheet', 'Filament', 'Knot']

path_gamacats = '/data2/brouwer/MergedCatalogues'
gamacatname = '%s/GAMACatalogue_1.0.fits'%path_gamacats
shuffledcatname = '%s/shuffled_environment_S4_deltaR.fits'%path_gamacats

shuffled = False

# Importing GAMA catalogue
print 'Importing GAMA catalogue:', gamacatname
gamacat = pyfits.open(gamacatname)[1].data
shuffledcat = pyfits.open(shuffledcatname)[1].data

galIDlist = gamacat['ID']

# Importing angular seperation
angseplist = gamacat['AngSep%s'%centering]
angseplist[angseplist<=0] = 0.

# Importing and correcting log(Mstar)
logmstarlist = gamacat['logmstar']
fluxscalelist = gamacat['fluxscale'] # Fluxscale, needed for stellar mass correction
corr_list = np.log10(fluxscalelist)# - 2*np.log10(h/0.7)
logmstarlist = logmstarlist + corr_list

# Applying a mask to the galaxies
obsmask = (fluxscalelist<500)&(logmstarlist>5)

logmstarlist = logmstarlist[obsmask]
mstarlist = 10**logmstarlist
angseplist = angseplist[obsmask]

zlist = gamacat['Z'][obsmask]
ranklist = gamacat['rank%s'%centering][obsmask]

if not shuffled:
    # Importing the real environment
    envlist = gamacat['envS4'][obsmask]
else:
    # Importing the shuffled environment
    envlist = shuffledcat['shuffenvR4'][obsmask]
    
print 'Imported: %i of %i galaxies'%(len(logmstarlist), len(galIDlist))

# Importing mstar weights
if shuffled:
    weightcatname = 'mstarweight_shuffled.fits'
else:
    weightcatname = 'mstarweight.fits'

weightcat = pyfits.open(weightcatname)[1].data
weightlist = weightcat['mstarweight']
weightlist = weightlist[obsmask]

# Calculating average redshift, log(M*) and satellite fraction of the lens samples (needed for halo model)

if shuffled:
    print 'For the shuffled environments:'
else:
    print 'For the cosmic environments:'
print 

print 'Without logmstarweight:'
zaverage, mstaraverage, fsatmin, fsatmax = envutils.calc_halomodel_input(envnames, envlist, ranklist, zlist, mstarlist, False)


print
print 'With logmstarweight:'
zaverage, mstaraverage, fsatmin, fsatmax = envutils.calc_halomodel_input(envnames, envlist, ranklist, zlist, mstarlist, weightlist)



