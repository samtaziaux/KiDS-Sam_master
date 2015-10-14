# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

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

import astropy.units as u
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.coordinates import search_around_sky

#%matplotlib inline

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number
pi = np.pi # Pi

purpose = 'shearcatalog'
centering = 'None'

O_matter = 0.315
O_lambda = 1.-O_matter
Ok = 0.
h = 1.

# <codecell>

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

galRAlist *= u.degree
galDEClist *= u.degree
Dcllist *= u.pc
Dallist *= u.pc

petrolist = gamacat['Rpetro'] # petrosian magnitude of all galaxies
nQlist = gamacat['nQ'] # nQ of all galaxies
delta8list = gamacat['delta8']# delta8 of all galaxies
complist = gamacat['Completeness'] # Completeness around all galaxies
gmaglist = gamacat['absmag_g'] # g magnitude of all galaxies
rmaglist = gamacat['absmag_r'] # r magnitude of all galaxies

# Defining the color (g-r)0
colorlist = gmaglist-rmaglist
maglist = utils.calc_magnitude(galZlist, Dcllist, colorlist, petrolist)

# <codecell>

# Masking of unused lenses

zmin = 0.039
zmax = 0.263
Dcmin = distance.comoving(zmin, O_matter, O_lambda, h)
Dcmax = distance.comoving(zmax, O_matter, O_lambda, h)
print 'Distance: Min    Max'
print Dcmin, '  ', Dcmax, 'pc'

lensmask = (zmin<galZlist)&(galZlist<zmax)#&(nQlist>=3)&(complist>0.8)
DDPmask = (lensmask)&(-21.8<maglist)&(maglist<-20.1)

# Mask all non-selected GAMA galaxies
galIDs, galRAs, galDECs, galZs, Dcls, Dals, delta8s, comps = [lenslist[lensmask] \
for lenslist in [galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, delta8list, complist]] 

# Mask all non-DDP GAMA galaxies
DDP_IDs, DDP_RAs, DDP_DECs, DDP_Zs, DDP_Dcls, DDP_Dals = [lenslist[DDPmask] \
for lenslist in [galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist]] 

galcoords =  SkyCoord(ra=galRAs, dec=galDECs, distance=Dcls, frame='icrs')
DDPcoords =  SkyCoord(ra=DDP_RAs, dec=DDP_DECs, distance=DDP_Dcls, frame='icrs')

print
print 'Number of galaxies:'
print 'All (no cut):', len(galIDlist)
print 'Used lenses:', len(galIDs)
print 'Density Defining Population:', len(DDP_IDs)
print

# <codecell>

# Import GAMA masks for completeness calculation
path_gamamasks = ['/disks/shear10/GamaMasks/%smask08000.fits'%g for g in ['g09', 'g12', 'g15']]
gamamasks = np.array([pyfits.open(path_gamamask, ignore_missing_end=True)['PRIMARY'].data for path_gamamask in path_gamamasks])
gamamasks[gamamasks ==-2.] = 0.

# Creating the RA and DEC coordinates of each GAMA field
pixsize_coords = 0.01

print 'Importing GAMAmask:'
print 'Old size:', np.shape(gamamasks)
gapsize = pixsize_coords/0.001

gamaRAnums = np.arange(int(gapsize/2.), int(len(gamamasks[0])+gapsize/2), int(gapsize))
gamaDECnums = np.arange(int(gapsize/2.), int(len(gamamasks[0,0])+gapsize/2), int(gapsize))
#print gamaRAnums
#print gamaDECnums
                     
gama1 = [[129., 141.], [-2.,3.]]
gama2 = [[174., 186.], [-3.,2.]]
gama3 = [[211.5, 223.5], [-2.,3.]]
gamalims = np.array([gama1, gama2, gama3])

gamamasks_small = np.zeros([len(gamalims), len(gamaRAnums), len(gamaDECnums)])

for f in xrange(len(gamalims)):
    for i in xrange(len(gamaRAnums)):
        gamaRAnum = gamaRAnums[i]
        gamamasks_small[f, i, :] = gamamasks[f, gamaRAnum, :][gamaDECnums]

print 'New size:', np.shape(gamamasks_small)

gamamasks = gamamasks_small
gamamasks = np.reshape(gamamasks, [len(gamalims), np.size(gamamasks[0])])

print 'Final shape:', np.shape(gamamasks)

# <codecell>

# Calculating the sizes of the circle (in degree) for each lens, from Rmax (in Mpc)
Rmax = np.arange(1,9)*u.Mpc # Maximum radius (in Mpc) around each lens
x, y = np.meshgrid(Rmax, Dals.to(u.Mpc)) # Maximum angular separation (in degree) around each lens
sepmax = np.degrees((x/y).value)
x, y = [[],[]]

# The complete size of the circle, to calculate the completeness
complete = pi*sepmax**2*(1/pixsize_coords**2)

# This array will contain the value delta_r for each lens
delta_rs = np.zeros(np.shape(sepmax))
delta_r = np.ones([len(galIDlist), len(Rmax)])*-999

print np.shape(delta_rs)
print np.shape(delta_r)

# <codecell>

# Calculating the volume and mean DDP density of GAMA
gamavol = utils.calc_rho_mean(gamalims, O_matter, O_lambda, h)
rho_mean = len(DDP_IDs)/gamavol

print 'mean density', rho_mean

# <codecell>

# This array will contain the value delta_r for each lens
delta_rs = np.zeros(np.shape(sepmax))
delta_r = np.ones([len(galIDlist), len(Rmax)])*-999

comp_rs = np.zeros(np.shape(sepmax))
comp_r = np.ones([len(galIDlist), len(Rmax)])*-999

print np.shape(delta_rs)
print np.shape(delta_r)

for f in xrange(len(gamalims)): # For each GAMA field ...
    
    print
    print 'GAMA field %i:'%(f+1)
    
    gamacoords_field = utils.create_fieldcoords(gamalims, pixsize_coords, f) # Create the coordinates of the GAMA mask

    # Masking the lenses that are outside the field
    galfieldmask = (gamalims[f, 0, 0] < galRAs.value) & (galRAs.value < gamalims[f, 0, 1]) \
    & (gamalims[f, 1, 0] < galDECs.value) & (galDECs.value < gamalims[f, 1, 1])
    
    # Masking the DDP galaxies that are outside the field
    DDPfieldmask = (gamalims[f, 0, 0] < DDP_RAs.value) & (DDP_RAs.value < gamalims[f, 0, 1]) \
    & (gamalims[f, 1, 0] < DDP_DECs.value) & (DDP_DECs.value < gamalims[f, 1, 1])
  
    galcoords_field = galcoords[galfieldmask]
    DDPcoords_field = DDPcoords[DDPfieldmask]
    rho_field = np.zeros([np.sum(galfieldmask), len(Rmax)])
    comp_field = np.zeros([np.sum(galfieldmask), len(Rmax)])
    
    print '    Contains:', len(galIDs[galfieldmask]), 'galaxies'
    
    #print '    Galaxy coordinates:', galcoords_field
    #print '    GAMA coordinates:', gamacoords_field

#    plt.plot(galcoords[fieldmask].ra.deg, galcoords[fieldmask].dec.deg, '.')
#    plt.show()
    
    #For each lens in this field ...
    Ngals = len(galcoords_field)
    #Ngals = 10
    for g in xrange(Ngals):
        
        if g % 100 == 0:
            print 'Field #%i:'%(f+1), float(g+1)/float(Ngals)*100., '%'
        
        # ... find the seperation with all GAMA coordinates and DDP galaxies
        gamasep = (galcoords_field[g]).separation(gamacoords_field).deg
        DDPsep = (galcoords_field[g]).separation_3d(DDPcoords_field).to('Mpc')
        
        # for each Rmax...
        for R in xrange(len(Rmax)):
    
            # ... determine the gama coordinates that lie within the circle
            gama_Rmask = gamasep < (sepmax[galfieldmask])[g, R]
            comp_field[g, R] = np.sum((gamamasks[f])[gama_Rmask])/((complete[galfieldmask])[g, R])
                        
            # ... calculate the number density of DDP galaxies inside the sphere
            sphere_size = (4./3.*pi*Rmax[R]**3)*comp_field[g, R]
            DDP_Rmask = DDPsep < Rmax[R]
            
 #           print comp_field[g, R]
            
            # Calculate rho_r
            rho_field[g, R] = (np.sum(DDP_Rmask)/sphere_size).value
                        
    delta_r_field = (rho_field-rho_mean)/rho_mean
    delta_rs[galfieldmask] = delta_r_field
    comp_rs[galfieldmask] = comp_field
    
delta_r[lensmask] = delta_rs
comp_r[lensmask] = comp_rs

# <codecell>

print 'delta_r (Margot/Tamsyn), difference, completeness:'
for g in xrange(Ngals):
    print '%.5g      %.5g:     %.5g     %.5g'\
    %(delta_rs[:,7][g], delta8s[g], abs((delta_rs[:,7][g]-delta8s[g])/delta8s[g]), comp_rs[g, 7])

nanmask = np.isfinite(delta8s[0:Ngals])
difference = (((delta_rs[:,7]-delta8s)/delta8s)[0:Ngals])[nanmask]

print 'Mean difference:', np.mean(abs(difference)),' Bias:', np.mean(difference)

# <codecell>

#Ngal = 88.1e3
#V_DDP = 6.75e6
#rho_DDP = 5.35e-3
#print rho_DDP*V_DDP

filename = 'delta_r_new.fits'
deltanames = ['delta_R%i'%i for i in Rmax.value]
compnames = ['compl_R%i'%i for i in Rmax.value]

outputnames = np.hstack([deltanames, compnames])
output = np.hstack([delta_r, comp_r]).T

utils.write_catalog(filename, galIDlist, outputnames, output)

# <codecell>


