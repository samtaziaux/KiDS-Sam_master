#!/usr/bin/python

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

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

n = 8

#DEClim = [125,145]
DEClim = [170,190]
#DEClim = [210,225]
#DEClim = [125,225]

purpose = 'shearcatalog'
Rmax = 2000

paths = ['/data2/brouwer/ManMask', '/data2/brouwer/LF_cat_DR2_v2']
colors = ['yellow', 'g']
plotlabels = ['DR1 KiDS fields', 'DR2 KiDS fields']


for p in xrange(len(paths)):
	kidscoord, kidscat_end = shear.run_kidscoord(paths[p])
	
	kidsfields = kidscoord.values()

	for k in xrange(len(kidsfields)):
		
		RA = kidsfields[k][0]
		DEC = kidsfields[k][1]-0.2+(p*0.4)
		
		if k == 0:
			plt.plot(RA, DEC, marker='s', markersize=20, color='%s'%colors[p], label=plotlabels[p])
			
		else:
			plt.plot(RA, DEC, marker='s', markersize=20, color='%s'%colors[p])

# GAMA galaxies
galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, obslist, obslimlist = shear.import_gamacat('BCG', purpose, 0, -999, np.inf, 0.315, 1-0.315, 1, 'logmstar', 'No')

# Catmatch for the old and new fields
kidscoord1, kidscat_end = shear.run_kidscoord(paths[0])
kidscoord2, kidscat_end = shear.run_kidscoord(paths[1])

catmatch1, kidscats1, galIDs_infield1 = shear.run_catmatch(kidscoord1, galIDlist, galRAlist, galDEClist, Dallist, Rmax, purpose)
catmatch2, kidscats2, galIDs_infield2 = shear.run_catmatch(kidscoord2, galIDlist, galRAlist, galDEClist, Dallist, Rmax, purpose)

#galIDs_matched1 = np.unique(np.hstack(catmatch1.values()))
#galIDs_matched2 = np.unique(np.hstack(catmatch2.values()))

galIDs_matched1 = galIDs_infield1
galIDs_matched2 = galIDs_infield2

galIDs_diff = galIDs_matched1[np.logical_not(np.in1d(galIDs_matched1, galIDs_matched2))]
catmatchmask = np.in1d(galIDlist, galIDs_diff)

print 'DR1', len(galIDs_matched1)
print 'DR2', len(galIDs_matched2)
print 'Total differing lenses:', len(galIDs_diff)

# Masking according to the logmstar bins

bins = np.loadtxt('binlimits/logmstarbins_8_Edo.txt').T
obsmask = (bins[n-1] < obslist) & (obslist < bins[n])

galIDlist_obs = galIDlist[obsmask]
galRAlist_obs = galRAlist[obsmask]
galDEClist_obs = galDEClist[obsmask]
print 'Galaxies in this bin:', len(galIDlist[obsmask])

for g in np.arange(0, len(galIDlist_obs)):
	
	RA = galRAlist_obs[g]
	DEC = galDEClist_obs[g]
	
	if g == 0:
		plt.plot(RA, DEC, marker='x', color='r', linestyle='', label='Lenses in logmstar-bin %i'%(n))
	else:
		plt.plot(RA, DEC, marker='x', color='r', linestyle='')


galIDlist_diff = galIDlist[obsmask & catmatchmask]
galRAlist_diff = galRAlist[obsmask & catmatchmask]
galDEClist_diff = galDEClist[obsmask & catmatchmask]

print 'DR1 lenses in this bin:', len(galIDlist[obsmask & np.in1d(galIDlist, galIDs_matched1)])
print 'DR2 lenses in this bin:', len(galIDlist[obsmask & np.in1d(galIDlist, galIDs_matched2)])

print 'Differing lenses in this bin:', len(galIDlist_diff)

for c in np.arange(0, len(galIDlist_diff)):
	
	RA = galRAlist_diff[c]
	DEC = galDEClist_diff[c]
	
	if c == 0:
		plt.plot(RA, DEC, marker='x', color='b', linestyle='', label='Differing lenses in logmstar-bin %i'%(n))
	else:
		plt.plot(RA, DEC, marker='x', color='b', linestyle='')

plt.xlabel('RA',fontsize=15)
plt.ylabel('DEC',fontsize=15)


plt.autoscale(enable=False, axis='both', tight=None)
plt.axis([DEClim[0],DEClim[1],-3,4])

plt.xlim(DEClim[0],DEClim[1])
plt.ylim(-3,4)

plt.legend(loc='upper right',ncol=2, prop={'size':12})

plotname = 'old_vs_new_kidscoords_DEC-%g-%g_logmstarbin-%i.png'%(DEClim[0], DEClim[1], n)

plt.savefig(plotname, format='png')
print 'Written: ESD profile plot:', plotname
#	plt.show()
plt.close()
