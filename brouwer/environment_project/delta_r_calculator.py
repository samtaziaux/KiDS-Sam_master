#!/usr/bin/python

"Module to determine the local overdensity delta_r within a sphere of radius r."

# Import the necessary libraries
import sys
sys.path.insert(0, '/data2/brouwer/shearprofile/KiDS-GGL/esd_production/')

import pyfits
import numpy as np
import distance
import os
import time
import shearcode_modules as shear
import delta_r_utils as utils
import gc
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number
pi = np.pi # Pi

purpose = 'shearcatalog'
centering = 'None'

O_matter = 0.25
O_lambda = 1.-O_matter
Ok = 0.
h = 1.



# Importing the GAMA catalogue
path_gamacat = '/data2/brouwer/MergedCatalogues/GAMACatalogue_1.0.fits'
centering = 'None'
purpose = 'shearcatalog'
Ncat = 1
Runit = 'kpc'
lens_weights = {'None': ''}

gamacat, galIDlist, galRAlist, galDEClist, galweightlist, galZlist, Dcllist, Dallist = \
shear.import_gamacat(path_gamacat, centering, purpose, Ncat, \
O_matter, O_lambda, Ok, h, Runit, lens_weights)

# Conversion from pc to Mpc
Dcllist = Dcllist/1e6
Dallist = Dallist/1e6

petrolist = gamacat['Rpetro2'] # petrosian magnitude of all galaxies
nQlist = gamacat['nQ'] # nQ of all galaxies
delta8list = gamacat['delta8']# delta8 of all galaxies
complist = gamacat['Completeness'] # Completeness around all galaxies
gmaglist = gamacat['absmag_g'] # g magnitude of all galaxies
rmaglist = gamacat['absmag_r'] # r magnitude of all galaxies

# Defining the color (g-r)0
colorlist = gmaglist-rmaglist

maglist = utils.calc_magnitude(galZlist, Dcllist, colorlist, petrolist)

# Masking of unused lenses
zmin = 0.039
zmax = 0.263

lensmask = (zmin<galZlist)&(galZlist<zmax)&(nQlist>=3)&(complist>0.8)#&(np.isfinite(complist))
DDPmask = (lensmask)&(-21.8<maglist)&(maglist<-20.1)

galIDs, galRAs, galDECs, galZs, Dcls, Dals, delta8s, comps = [lenslist[lensmask] \
for lenslist in [galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, delta8list, complist]] # Mask all non-selected GAMA galaxies

DDP_IDs, DDP_RAs, DDP_DECs, DDP_Zs, DDP_Dcls, DDP_Dals = [lenslist[DDPmask] \
for lenslist in [galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist]] # Mask all non-DDP GAMA galaxies

galcoords =  SkyCoord(ra=galRAs*u.degree, dec=galDECs*u.degree, frame='icrs')

print
print 'Number of galaxies:'
print 'All (no cut):', len(galIDlist)
print 'Used lenses:', len(galIDs)
print 'Density Defining Population:', len(DDP_IDs)
print



# Import GAMA masks for completeness calculation
path_gamamasks = ['/disks/shear10/GamaMasks/%smask08000.fits'%g for g in ['g09', 'g12', 'g15']]
gamamasks = np.array([pyfits.open(path_gamamask, ignore_missing_end=True)['PRIMARY'].data for path_gamamask in path_gamamasks])
gamamasks[gamamasks ==-2.] = 0.
print gamamasks

# Creating the RA and DEC coordinates of each GAMA field
pixsize = 0.1

gama1 = [[129., 141.], [-2.,3.]]
gama2 = [[174., 186.], [-3.,2.]]
gama3 = [[211.5, 223.5], [-2.,3.]]

gamalims = np.array([gama1, gama2, gama3])

# Matching the GAMA coordinates to the lenses within Rmax
for f in xrange(len(gamalims)): # For each GAMA field ...
    
    print 'GAMA field %i:'%(f+1)
    
    gamamask = gamamasks[f]
    gamacoords = utils.create_fieldcoords(gamalims, pixsize, f)

    Ngals = 10
    for g in xrange(Ngals):

        print float(g+1)/float(Ngals)*100., '%'
        Rmax = 1*u.deg

        idxlenses, idxgama, sep2d, dist3d = search_around_sky(galcoords, gamacoords, Rmax)

        plt.plot(idxlenses)
        plt.plot(gamaRAs, gamaDECs, '.')
        plt.show()

        #print np.amax(idxlenses), np.amax(idxgama), np.amax(sep2d), np.amax(dist3d)

quit()


# Calculating the volume and mean DDP density of GAMA
gamavol = utils.calc_rho_mean(gamalims, O_matter, O_lambda, h)
rho_mean = len(DDP_IDs)/gamavol
print 'mean density', rho_mean


# Calculating the number density rho around each lens, for different sphere sizes
Rmax = np.arange(1,9) # Maximum radius around each lens
print 'Rmax:', Rmax

# For each Rmax, this list will contain the number density rho around each lens
rholist = np.zeros([len(galIDs), len(Rmax)])

# For each lens ...
#Ngals=len(galIDs)
Ngals = 10
for g in xrange(Ngals):

    print float(g+1)/float(Ngals)*100., '%'
    
    # ... select all properties of that lens
    galID = galIDs[g]
    galRA = galRAs[g]
    galDEC = galDECs[g]
    galZ = galZs[g]
    Dcl = Dcls[g]
    Dal = Dals[g]
    comp = comps[g]
    
    # ... calculate rho around this lens
    rho_r = utils.calc_rho(Rmax, galRA, DDP_RAs, galDEC, DDP_DECs, Dcl, DDP_Dcls, comp)

    rholist[g] = rho_r

# Calculate the density delta for each lens and Rmax
delta_r = (rholist-rho_mean)/rho_mean
    

print 'Density rho:', rholist
print
print 'Mean density:', rho_mean
print
print 'delta_r (Margot):', delta_r[:,7][0:Ngals]
print 'delta_r (Tamsyn):', delta8s[0:Ngals]

print 'difference', np.sum((delta_r[:,7][0:Ngals]-delta8s[0:Ngals])**2)


#Ngal = 88.1e3
#V_DDP = 6.75e6
#rho_DDP = 5.35e-3
#print rho_DDP*V_DDP


filename = 'delta_r_new.txt'
with open(filename, 'w') as file:
    print >>file, '# galID     delta(R=1Mpc)     delta(R=2Mpc)     delta(R=3Mpc)     delta(R=4Mpc)     delta(R=5Mpc)     delta(R=6Mpc)     delta(R=7Mpc)     delta(R=8Mpc)'

with open(filename, 'a') as file:
    for g in xrange(len(galIDs)):
        print >>file, '%.12g	%.12g	%.12g	%.12g	%.12g	%.12g	%.12g	%.12g	%.12g'%(galIDs[g], delta_r[g,0], delta_r[g,1], delta_r[g,2], delta_r[g,3], delta_r[g,4], delta_r[g,5], delta_r[g,6], delta_r[g,7])

print 'Written: ESD profile data:', filename




