#!/usr/bin/python

""" Create a radial bin file """

import numpy as np

# Creating the radial bins

Rbins_tot = np.array([])
"""
# Start, End, number of steps and step size of the radius R (logarithmic 10^x)
nRbins = 0
Rmin = 10.
Rmax = 20.
Rbins = np.logspace(np.log10(Rmin), np.log10(Rmax), nRbins+1)
Rbins = Rbins[0:-1]

print np.log10(Rbins[1]) - np.log10(Rbins[0])
Rbins_tot = np.append(Rbins_tot, Rbins)
"""
# Start, End, number of steps and step size of the radius R (logarithmic 10^x)
nRbins = 12
Rmin = 20.
Rmax = 2000.
Rbins = np.logspace(np.log10(Rmin), np.log10(Rmax), nRbins+1)
Rbins = Rbins[0:-1]

print np.log10(Rbins[1]) - np.log10(Rbins[0])
Rbins_tot = np.append(Rbins_tot, Rbins)
Rbins_tot = np.append(Rbins_tot, Rmax)

"""
# Start, End, number of steps and step size of the radius R (logarithmic 10^x)
nRbins = 0
Rmin = 2000.
Rmax = 10000.
Rbins = np.logspace(np.log10(Rmin), np.log10(Rmax), nRbins+1)


print np.log10(Rbins[1]) - np.log10(Rbins[0])
Rbins_tot = np.append(Rbins_tot, Rbins)
"""

nRbins_tot = len(Rbins_tot)-1
Rbins_center = np.array([(Rbins_tot[r]+Rbins_tot[r+1])/2 for r in xrange(nRbins_tot)])

Rbins_min = Rbins_tot[0:-1]
Rbins_max = Rbins_tot[1:nRbins_tot+1]

print
print 'Rbins:', (Rbins_tot)
print 'Number:', nRbins_tot
print
print 'Min:', (Rbins_min)
print 'Center:', (Rbins_center)
print 'Max:', (Rbins_max)
print

# Printing the radial bins to a text file

Rbinsname = '%sbins.txt'%nRbins_tot

with open(Rbinsname, 'w') as file:
	print >>file, "# Radial bin: min	center	max"

with open(Rbinsname, 'a') as file:
	for i in xrange(nRbins_tot):
		print >>file, Rbins_min[i], '	', Rbins_center[i], '	', Rbins_max[i]
		

print "Written: Radial bin file:", Rbinsname
