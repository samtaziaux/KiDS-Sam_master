# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#!/usr/bin/python

"Module to determine the local overdensity delta_r within a sphere of radius r."

# Import the necessary libraries
import sys
sys.path.insert(0, '../../esd_production/')

import pyfits
import numpy as np
import distance
import os
import time
import gc
from matplotlib import pyplot as plt

import delta_r_utils as utils
import shearcode_modules as shear
import environment_utils as envutils

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
#path_gamacat = '/data2/brouwer/MergedCatalogues/GAMACatalogue_1.0.fits'
path_gamacat = '/disks/shear10/brouwer_veersemeer/MergedCatalogues/GAMACatalogue_1.0.fits'

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
Dcmin = distance.comoving(zmin, O_matter, O_lambda, h)*u.pc
Dcmax = distance.comoving(zmax, O_matter, O_lambda, h)*u.pc

print 'Redshift: Min    Max'
print zmin, '    ', zmax
print 'Distance: Min    Max'
print Dcmin, '  ', Dcmax

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
pixsize_coords = 0.01
path_gamamasks = ['/disks/shear10/GamaMasks/%smask08000.fits'%g for g in ['g09', 'g12', 'g15']]
gamalims, gamamasks = utils.import_gamamasks(path_gamamasks, pixsize_coords)

# <codecell>

# Calculating the sizes of the circle (in degree) for each lens, from Rmax (in Mpc)
Rmax = np.arange(1,9)*u.Mpc # Maximum radius (in Mpc) around each lens
Rmaxmesh, Dalsmesh = np.meshgrid(Rmax, Dals.to(u.Mpc)) # Maximum angular separation (in degree) around each lens
sepmax = np.degrees((Rmaxmesh/Dalsmesh).value)

# This array will contain the value delta_r for each lens
delta_rs = np.zeros(np.shape(sepmax))
delta_r = np.ones([len(galIDlist), len(Rmax)])*-999


# <codecell>

# The part of each sphere that lies above/below the redshift maximum/minimum, to calculate the completeness

dDmax = (Dcmax - Dcls).to(u.Mpc)
dDmin = (Dcls - Dcmin).to(u.Mpc)

Rmaxmesh, dDmaxmesh = np.meshgrid(Rmax, dDmax)
Rmaxmesh, dDminmesh = np.meshgrid(Rmax, dDmin)

Dmaxmask = dDmaxmesh < Rmaxmesh
Dminmask = dDminmesh < Rmaxmesh

dDmesh = np.ones(np.shape(Dmaxmask)) * Rmaxmesh

dDmesh[Dmaxmask] = dDmaxmesh[Dmaxmask]
dDmesh[Dminmask] = dDminmesh[Dminmask]

height = Rmaxmesh + dDmesh

sphere = 4./3. * pi * Rmaxmesh**3.
Vinside = (pi/3. * height**2.) * (3.*Rmaxmesh - height)
distcomp = Vinside/sphere


# <codecell>

# Calculating the volume and mean DDP density of GAMA
gamavol = utils.calc_rho_mean(gamalims, gamamasks, O_matter, O_lambda, h)
rho_mean = len(DDP_IDs)/gamavol

print 'mean density', rho_mean

# <codecell>

# This array will contain the value delta_r for each lens
delta_rs = np.zeros(np.shape(sepmax))
delta_r = np.ones([len(galIDlist), len(Rmax)])*-999

comp_rs = np.zeros(np.shape(sepmax))
comp_r = np.ones([len(galIDlist), len(Rmax)])*-999


for f in xrange(len(gamalims)): # For each GAMA field ...
    
    print
    print 'GAMA field %i:'%(f+1)
    
    gridRAlims = gamalims[f, 0]
    gridDEClims = gamalims[f, 1]
    gamacoords_field = utils.create_gridcoords(gridRAlims, gridDEClims, pixsize_coords) # Create the coordinates of the GAMA mask

    # Masking the lenses that are outside the field
    galfieldmask = (gamalims[f, 0, 0] < galRAs.value) & (galRAs.value < gamalims[f, 0, 1]) \
    & (gamalims[f, 1, 0] < galDECs.value) & (galDECs.value < gamalims[f, 1, 1])
    
    # Masking the DDP galaxies that are outside the field
    DDPfieldmask = (gamalims[f, 0, 0] < DDP_RAs.value) & (DDP_RAs.value < gamalims[f, 0, 1]) \
    & (gamalims[f, 1, 0] < DDP_DECs.value) & (DDP_DECs.value < gamalims[f, 1, 1])
  
    galcoords_field = galcoords[galfieldmask]
    DDPcoords_field = DDPcoords[DDPfieldmask]
    distcomp_field = distcomp[galfieldmask]
    
    rho_field = np.zeros([np.sum(galfieldmask), len(Rmax)])
    comp_field = np.zeros([np.sum(galfieldmask), len(Rmax)])
    
    print '    Contains:', len(galIDs[galfieldmask]), 'galaxies'
    
    #print '    Galaxy coordinates:', galcoords_field
    #print '    GAMA coordinates:', gamacoords_field

#    plt.plot(galcoords[fieldmask].ra.deg, galcoords[fieldmask].dec.deg, '.')
#    plt.show()
    
    #For each lens in this field ...
    Ngals = len(galcoords_field)
    #Ngals = 20
    for g in xrange(Ngals):
        
        if g % 1000 == 0:
            print 'Galaxy #%i:'%(g+1), float(g+1)/float(Ngals)*100., '%'
        
        # ... find the seperation with all GAMA coordinates and DDP galaxies
        gamasep = (galcoords_field[g]).separation(gamacoords_field).deg
        DDPsep = (galcoords_field[g]).separation_3d(DDPcoords_field).to('Mpc')
        
        # Create a test-grid to calculates the completeness
        compgridRAlims = [galcoords_field[g].ra.deg - 5, galcoords_field[g].ra.deg + 5]
        compgridDEClims = [galcoords_field[g].dec.deg - 5, galcoords_field[g].dec.deg + 5]
        compgrid = utils.create_gridcoords(compgridRAlims, compgridDEClims, pixsize_coords)
        
        compgridsep = (galcoords_field[g]).separation(compgrid).deg
        
        # for each Rmax...
        for R in xrange(len(Rmax)):
    
            # ... determine the gama coordinates that lie within the circle
            gama_Rmask = gamasep < (sepmax[galfieldmask])[g, R]
            compgrid_Rmask = compgridsep < (sepmax[galfieldmask])[g, R]

            complete = np.sum(2*((sepmax[galfieldmask])[g, R]**2 - compgridsep[compgrid_Rmask]**2)**0.5)
            
            comp_field[g, R] = np.sum((gamamasks[f])[gama_Rmask] * \
                    2*((sepmax[galfieldmask])[g, R]**2 - gamasep[gama_Rmask]**2)**0.5) \
                    * distcomp_field[g, R] / complete
            
            #if R==7:
            #    print comps[g], comp_field[g, R]
            
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

# Comparing delta_r histograms

nanmask = np.isfinite(delta8s[0:Ngals])

deltamin = -1
deltamax = 6

delta8bins = np.arange(deltamin, deltamax, 0.4)
histbins = np.append(delta8bins, deltamax)

# the histogram of the data
#delta_margot_hist, histbins, patches = plt.hist(delta_rs[:,7][nanmask], histbins, histtype='step', color = 'blue')
#delta_tamsyn_hist, histbins, patches = plt.hist(delta8s[nanmask], histbins, histtype='step', color = 'red')
#plt.ylabel(r'Number of galaxies', fontsize=15)
#plt.xlabel(r'Local overdensity $\delta_8$', fontsize=15)

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

filename = '/disks/shear10/brouwer_veersemeer/MergedCatalogues/delta_r_catalog_test.fits'
deltanames = ['delta_R%i'%i for i in Rmax.value]
compnames = ['comp_R%i'%i for i in Rmax.value]

outputnames = np.hstack([deltanames, compnames])
output = np.hstack([delta_r, comp_r]).T

envutils.write_catalog(filename, galIDlist, outputnames, output)

# <codecell>


