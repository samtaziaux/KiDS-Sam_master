#!/usr/bin/python

"Module to determine the local overdensity delta_r within a sphere of radius r."

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
import memory_test as memory
import time
import gc


# Important constants
G = const.G.to('pc3/Msun s2') # Gravitational constant
c = const.c.to('pc/s') # Speed of light
inf = np.inf # Infinity
nan = np.nan # Not a number
pi = np.pi

purpose = 'shearcatalog'
centering = 'BCG'


O_matter = 0.25
O_lambda = 1. - O_matter
h = 1.0

galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, petrolist, nQlist = shear.import_gamacat(centering, purpose, 1, -999, inf, O_matter, O_lambda, h, 'Rpetro2', 'nQ')
galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, delta8list, complist = shear.import_gamacat(centering, purpose, 1, -999, inf, O_matter, O_lambda, h, 'delta8', 'Completeness')
galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, gmaglist, rmaglist = shear.import_gamacat(centering, purpose, 1, -999, inf, O_matter, O_lambda, h, 'absmag_g', 'absmag_r')

# Defining the color (g-r)0
colorlist = gmaglist-rmaglist

# Conversion from kpc to Mpc
Dcllist = Dcllist/1e3
Dallist = Dallist/1e3


# Calculation the k(z)-correction
colorbins = [-1.344980, 0.2, 0.34, 0.48, 0.62, 0.76, 0.9, 16.99882]
Ncolorbins = len(colorbins)-1
kcor_table = np.array([[-31.36066, 38.63440, -14.79088, 1.427126, 1.3007671E-03], [-17.76654, 25.49545, -10.79095, 1.365723, 6.2354435E-03], [-12.94193, 21.43714, -9.825956, 1.683059, -1.9719140E-03], [-6.399204, 14.76284, -7.473374, 1.847070, -6.8006814E-03], [9.017070, -1.390476, -0.9145135, 1.375556, -4.7244146E-03], [14.78212, -6.591838, 0.9443237, 1.357406, -5.1314551E-03], [15.09393, -5.729835, -0.2097371, 1.859484, -1.2495981E-02]])

klist = np.zeros(len(galIDlist))
for c in xrange(Ncolorbins):
	
	colormask = (colorbins[c]<colorlist)&(colorlist<colorbins[c+1])

	acol = kcor_table[c]
	for a in xrange(len(acol)):
		klist[colormask] = klist[colormask] + acol[a]*galZlist[colormask]**(4-a)

#	print c+1
#	print np.median(colorlist[colormask])
#	print

# Calculating the E(z)-correction
Q0all = 0.97
Q0list = -Q0all*galZlist

print 'petromax', np.amax(petrolist[np.isfinite(petrolist)])

# Defining the magnitude
maglist = petrolist - 5*np.log10((1+galZlist)*Dcllist) - 25. - klist - Q0list

# Masking of unused lenses
Rmax = np.arange(1,9)
zmin = 0.039
zmax = 0.263

lensmask = (zmin<galZlist)&(galZlist<zmax)&(nQlist>=3)&(complist>0.8)#&(np.isfinite(complist))
DDPmask = (lensmask)&(-21.8<maglist)&(maglist<-20.1)

galIDs, galRAs, galDECs, galZs, Dcls, Dals, galIDmask = shear.mask_gamacat(purpose, galIDlist, lensmask, centering, galranklist, galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist) # Mask all GAMA galaxies that are not in this field
DDP_IDs, DDP_RAs, DDP_DECs, DDP_Zs, DDP_Dcls, DDP_Dals, DDP_IDmask = shear.mask_gamacat(purpose, galIDlist, DDPmask, centering, galranklist, galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist) # Mask all GAMA galaxies that are not in this field

delta8 = delta8list[galIDmask]
comps = complist[galIDmask]

rholist = np.zeros([len(galIDs), 8])

print len(galIDlist)
print len(galIDs)
print len(DDP_IDs)
print 'Rmax:', Rmax

# Calculating the volume and mean density of GAMA
"""
gama1 = [[129., 141.],[-2.,3.]]
gama2 = [[174., 186.],[-3.,2.]]
gama3 = [[211.5, 223.5],[-2.,3.]]

r1 = distance.comoving(0.039, O_matter, O_lambda, h)/1e3
r2 = distance.comoving(0.263, O_matter, O_lambda, h)/1e3

gamacoords = np.array([gama1,gama2,gama3])

gamavol = 0.
for i in xrange(len(gamacoords)):
    theta1 = np.radians(-gamacoords[i,0,0])
    theta2 = np.radians(-gamacoords[i,0,1])
    phi1 = np.radians(90.-gamacoords[i,1,0])
    phi2 = np.radians(90.-gamacoords[i,1,1])
    
    gamavol = gamavol + 1./3. * (theta1 - theta2) * (r1**3 - r2**3) * (np.cos(phi1) - np.cos(phi2))
"""
gamavol = 6.75e6

print 'GAMA volume:', gamavol, '(Mpc/h)^3'

rho_mean = len(DDP_IDs)/gamavol
print 'mean density', rho_mean

def calc_rho(Rmax, galRA, DDP_RAs, galDEC, DDP_DECs, Dcl, DDP_Dcls, comp):

    rho_r = np.array([])
    
    # Calculate the distance between two galaxies (in kpc)
    theta = (np.arccos(np.cos(np.radians(DDP_DECs))*np.cos(np.radians(galDEC))*np.cos(np.radians(DDP_RAs-galRA)) + np.sin(np.radians(DDP_DECs))*np.sin(np.radians(galDEC))))
    galdist = (Dcl**2 + DDP_Dcls**2 - 2*Dcl*DDP_Dcls*np.cos(theta))**0.5
    
    galdist = galdist[np.isfinite(galdist)]
    
    for R in Rmax:
        rhomask = galdist<R
        result = sum(rhomask)/(4./3.*pi*R**3)/comp
        rho_r = np.append(rho_r, result)      

    return rho_r


Ngals=len(galIDs)
for g in xrange(Ngals):

    print float(g)/float(Ngals)*100., '%'
    
    galID = galIDs[g]
    galRA = galRAs[g]
    galDEC = galDECs[g]
    galZ = galZs[g]
    Dcl = Dcls[g]
    Dal = Dals[g]
    comp = comps[g]
    
    rho_r = calc_rho(Rmax, galRA, DDP_RAs, galDEC, DDP_DECs, Dcl, DDP_Dcls, comp)

    rholist[g] = rho_r

delta_r = (rholist-rho_mean)/rho_mean
    
print 'mean density', rho_mean
print

print 'density rho', rholist
print

print 'delta_r (Margot):', delta_r[:,7][0:Ngals]
print 'delta_r (Tamsyn):', delta8[0:Ngals]

print 'difference', np.sum((delta_r[:,7][0:Ngals]-delta8[0:Ngals])**2)


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




