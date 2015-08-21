#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['Computer Modern']})

# Start, End, number of steps and step size of the radius R (logarithmic 10^x)

Rrangefile = np.loadtxt('11bins.dat').T
Rcenters = Rrangefile[1]
nRbins = len(Rcenters)


#filename_A = 'shear_output/results_covariance/covariance_matrix_rankIterCen1-1_Nfof5-inf_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A.txt'
#filename_B = 'shear_output/results_bootstrap/bootstrap_matrix_rankIterCen1-1_Nfof5-inf_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A.txt'

filename_A = 'shear_output/results_covariance/covariance_matrix_rankIterCen1-1_Nfof5-inf_LumBbin6_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A.txt'
filename_B = 'shear_output/results_bootstrap/bootstrap_matrix_rankIterCen1-1_Nfof5-inf_LumBbin6_ZB0.005-1.3_11bins_Om0.27_h100_Edo_A.txt'


data_A = np.loadtxt(filename_A).T
data_B = np.loadtxt(filename_B).T

binname = 'LumB'
binnum = 1
nObsbins = 6


frac_low = 0.7
n_low =  int(float(nRbins) * float(frac_low))

print
print '%s bin: %g/%g, Radial bins: %g'%(binname, binnum, nObsbins, nRbins)

correlation_A = data_A[5].reshape([nObsbins, nObsbins, nRbins,nRbins])
correlation_B = data_B[5].reshape([nObsbins, nObsbins, nRbins,nRbins])

correlation_A = correlation_A[binnum-1, binnum-1]
correlation_B = correlation_B[binnum-1, binnum-1]

correlation_difference = correlation_B-correlation_A

difference_low = correlation_difference[0:n_low, 0:n_low]
difference_high = correlation_difference[n_low:nRbins, n_low:nRbins]

difference_low = np.ravel(difference_low)
difference_high = np.ravel(difference_high)
correlation_difference = np.ravel(correlation_difference)

difference_low = difference_low[difference_low != -999]
difference_high = difference_high[difference_high != -999]
correlation_difference = correlation_difference[correlation_difference != -999]

median_difference = np.median(correlation_difference)
median_low = np.median(difference_low)
median_high = np.median(difference_high)

print
print 'Radial distances:'
for r in xrange(len(Rcenters)):
	if r == n_low:
		print Rcenters[r], 'kpc/h', '<-- Radial distance cut'
	else:
		print Rcenters[r], 'kpc/h'
print

print 'Median (total):', median_difference
print 'Median (low):', median_low
print 'Median (high):', median_high
print

# Plot Histogram
plottitle = '%s bin: %g of %g, Radial bins: %g'%(binname, binnum, nObsbins, nRbins)

plt.figure()

plt.xlabel(r'ESD $\langle\Delta\Sigma\rangle$ correlation [h70 M$_{\odot}/pc^2$]')
plt.ylabel('Number of radial bins')

plt.hist([correlation_difference, difference_low, difference_high] , color=['c', 'r', 'y'], alpha = 0.5, histtype='stepfilled', label=[r'bootstrap-analytical (all radii): %g $<$ R $<$ %g kpc/h'%(Rcenters[0], Rcenters[-1]), r'bootstrap-analytical (small radii): R $<$ %g kpc/h'%(Rcenters[n_low]), r'bootstrap-analytical (large radii): %g $<$ R kpc/h'%(Rcenters[n_low])])

plt.axvline(median_difference, color='c', linestyle='dashed', linewidth=2)
plt.axvline(median_high, color='y', linestyle='dashed', linewidth=2)
plt.axvline(median_low, color='r', linestyle='dashed', linewidth=2)

plt.plot([median_difference - np.std(correlation_difference), median_difference + np.std(correlation_difference)], [2] * 2, "c", linewidth=2);
plt.plot([median_low - np.std(difference_low), median_low + np.std(difference_low)], [4] * 2, "r", linewidth=2);
plt.plot([median_high - np.std(difference_high), median_high + np.std(difference_high)], [6] * 2, "y", linewidth=2);

#plt.axvline(0, color='b', linestyle='dashed', linewidth=2)

plt.axis([-0.4, 0.4, 0, 50])

plt.legend(loc='upper right',ncol=1)

# Plot the ESD profile into a file
file_ext = filename_A.split('.')[-1]

filename_A = filename_A.replace('%sbin%i'%(binname, nObsbins), '%sbin%iof%i'%(binname, binnum, nObsbins))

plotname_hist = filename_A.replace('.%s'%file_ext, '_%gkpc-limit.png'%Rcenters[n_low])
plotname_hist = plotname_hist.replace('matrix', 'histogram')

plt.savefig(plotname_hist,format='png')
print 'Written:', plotname_hist
plt.show()

plt.close()

# Plot eigenvalues

plt.figure()

plt.xlabel('Radius (kpc/h70)')
plt.ylabel(r'ESD Eigenvalues')


plt.plot();
plt.xscale('log')
plt.yscale('log')

eigenvalues_A, eigenvectors_A = np.linalg.eig(correlation_A)
eigenvalues_B, eigenvectors_B = np.linalg.eig(correlation_B)

plt.plot(Rcenters, eigenvalues_A, label = 'Correlation (Analytical)')
plt.plot(Rcenters, eigenvalues_B, label = 'Correlation (Bootstrap)')

#plt.axis([-0.4, 0.4, 0, 50])
plt.legend(loc='upper right',ncol=1)

plotname_ev = filename_A.replace('.%s'%file_ext, '.png')
plotname_ev = plotname_ev.replace('matrix', 'eigenvalues')

plt.savefig(plotname_ev,format='png')
print 'Written:', plotname_ev
plt.show()

plt.close()


