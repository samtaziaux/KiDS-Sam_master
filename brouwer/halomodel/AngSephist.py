#!/usr/bin/python

import numpy as np
import pyfits

centering = 'BCG'

path_gamacats = '/data2/brouwer/MergedCatalogues'
gamacatname = '%s/ShearMergedCatalogueAll_sv0.8_shuffdeltaR.fits'%path_gamacats

print 'Importing GAMA catalogue:', gamacatname
gamacat = pyfits.open(gamacatname)[1].data

angseplist = gamacat['AngSep%s'%centering] # IDs of all galaxies

