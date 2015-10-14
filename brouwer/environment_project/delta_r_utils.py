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
import gc
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky

pi = np.pi


# Creating the RA and DEC coordinates of each GAMA field
def create_fieldcoords(gamalims, pixsize, field):
    # Creating all coordinates within this GAMA field

    gamaRAlist = np.arange(gamalims[field, 0, 0], gamalims[field, 0, 1], pixsize)+pixsize/2
    gamaDEClist = np.arange(gamalims[field, 1, 0], gamalims[field, 1, 1], pixsize)+pixsize/2

    gamaRAs, gamaDECs = np.meshgrid(gamaRAlist, gamaDEClist)

    gamaRAs = np.reshape(gamaRAs, [np.size(gamaRAs)])
    gamaDECs = np.reshape(gamaDECs, [np.size(gamaDECs)])

    gamacoords = SkyCoord(ra=gamaRAs*u.degree, dec=gamaDECs*u.degree, frame='icrs')
    
    return gamacoords
    

# Calculating the galaxy magnitudes (following McNaught-Roberts et al.)
def calc_magnitude(galZlist, Dcllist, colorlist, petrolist):
    
    # Calculation the k(z)-correction
    colorbins = [-1.344980, 0.2, 0.34, 0.48, 0.62, 0.76, 0.9, 16.99882]
    Ncolorbins = len(colorbins)-1
    kcor_table = \
    np.array([[-31.36066, 38.63440, -14.79088, 1.427126, 1.3007671E-03], \
    [-17.76654, 25.49545, -10.79095, 1.365723, 6.2354435E-03], \
    [-12.94193, 21.43714, -9.825956, 1.683059, -1.9719140E-03], \
    [-6.399204, 14.76284, -7.473374, 1.847070, -6.8006814E-03], \
    [9.017070, -1.390476, -0.9145135, 1.375556, -4.7244146E-03], \
    [14.78212, -6.591838, 0.9443237, 1.357406, -5.1314551E-03], \
    [15.09393, -5.729835, -0.2097371, 1.859484, -1.2495981E-02]])

    klist = np.zeros(len(galZlist))
    for c in xrange(Ncolorbins):
        
        colormask = (colorbins[c]<colorlist)&(colorlist<colorbins[c+1])

        acol = kcor_table[c]
        for a in xrange(len(acol)):
            klist[colormask] = klist[colormask] + acol[a]*galZlist[colormask]**(4-a)

    # Calculating the E(z)-correction
    Q0all = 0.97
    Q0list = -Q0all*galZlist

    # Defining the magnitude
    maglist = petrolist - 5*np.log10( (1+galZlist) * (Dcllist.to(u.Mpc)).value ) - 25. - klist - Q0list
    
    return maglist


# Calculate the number density rho around one lens
def calc_rho(Rmax, galRA, DDP_RAs, galDEC, DDP_DECs, Dcl, DDP_Dcls, comp):

    rho_r = np.array([])
    
    # Calculate the distance between two galaxies (in Mpc)
    theta = (np.arccos(np.cos(np.radians(DDP_DECs))*np.cos(np.radians(galDEC))*np.cos(np.radians(DDP_RAs-galRA)) + np.sin(np.radians(DDP_DECs))*np.sin(np.radians(galDEC))))
    galdist = (Dcl**2 + DDP_Dcls**2 - 2*Dcl*DDP_Dcls*np.cos(theta))**0.5
    galdist = galdist[np.isfinite(galdist)]
    
    for R in Rmax:
        rhomask = galdist<R
        result = sum(rhomask)/(4./3.*pi*R**3)/comp
        rho_r = np.append(rho_r, result)

    return rho_r


# Calculating the volume of GAMA
def calc_rho_mean(gamalims, O_matter, O_lambda, h):
    
    """
    r1 = distance.comoving(0.039, O_matter, O_lambda, h)/1e3
    r2 = distance.comoving(0.263, O_matter, O_lambda, h)/1e3

    gamacoords = np.array([gama1,gama2,gama3])

    gamavol = 0.
    for i in xrange(len(gamacoords)):
        theta1 = np.radians(-gamalims[i,0,0])
        theta2 = np.radians(-gamalims[i,0,1])
        phi1 = np.radians(90.-gamalims[i,1,0])
        phi2 = np.radians(90.-gamalims[i,1,1])
        
        gamavol = gamavol + 1./3. * (theta1 - theta2) * (r1**3 - r2**3) * (np.cos(phi1) - np.cos(phi2))
    """
    
    gamavol = 6.75e6
    print 'GAMA volume:', gamavol, '(Mpc/h)^3'

    return gamavol
    
    
    
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
