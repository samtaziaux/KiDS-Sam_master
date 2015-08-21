#!/usr/bin/python

"This contains all the modules that are needed to calculate the shear profile catalog and the covariance."

import pyfits
import numpy as np
import distance
import sys
import os
import time
from astropy import constants as const, units as u
import numpy.core._dotblas
import glob
import gc
import scipy.optimize as opt
import shearcode_modules as shear

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['Computer Modern']})


# Important constants
G = const.G.to('km2 pc/Msun s2').value # Gravitational constant
pi = np.pi


def import_data(filename):
	
	data = np.loadtxt(filename).T

	bias = data[4]

	data_x = data[0]
	data_y = data[1]/bias

	ymask = (np.isfinite(data_y))

	data_x = data_x[ymask]
	data_y = data_y[ymask]

	errorh = (data[3])/bias # covariance error
	errorl = (data[3])/bias # covariance error
	errorh = errorh[ymask]
	errorl = errorl[ymask]
	
	return data_x, data_y, errorh, errorl, bias

def NFW(r, rho_0, Rs):
    return rho_0/((r/Rs)*(1+r/Rs)**2)

def SIS(r, sigma):
    return sigma**2/(2*G*r) / 1e3


def fit_func(xdata, ydata, yerror, func):

	"""
	# NFW profile
	def x2(params, r, ydata, yerror):
		return np.sum(((func(r, params[0], params[1])-ydata)**2)/yerror)

	fit = opt.fmin_slsqp(x2, [6, 1e4], bounds=[(0,9),(0,None)], args=(xdata, ydata, yerror), iprint=0)
	fitdata = func(xdata, fit[0], fit[1])

	"""
	# SIS profile:
	def x2(params, r, ydata, yerror):
		return np.sum(((func(r,params[0])-ydata)/yerror)**2)
	
#	fit = opt.fmin_slsqp(x2, [1e2], bounds=[(0,None)], args=(xdata, ydata, yerror), iprint=1)
	fit = opt.curve_fit(func, xdata, ydata, 1e2, sigma=yerror, absolute_sigma=True)

	value = fit[0][0]
	error = fit[1][0,0]**0.5
	fitdata = func(xdata, value)
	
	return fitdata, value, error

def print_masses(filename, masses, masserrors, Nobsbins1, Nobsbins2):

	with open(filename, 'w') as file:
		print >>file, '# sigma:	error:	[%i, %i]'%(Nobsbins1, Nobsbins2)

	with open(filename, 'a') as file:
		for i in xrange(Nobsbins1*Nobsbins2):
			print >>file, '%g	%g'%(masses[i], masserrors[i])
	
	return

def plot_masses(filenames1, filenames2, binname1, binrange1, Nobsbins1, binname2, binrange2, Nobsbins2):
	
	filename1 = filenames1[-1,-1]
	file_ext = filename1.split('.')[-1]
	plotname = filename1.replace('.%s'%file_ext,'_%s_matrix.png'%plotstyle)
	textname = plotname.replace('log_matrix.png', 'masses.txt')

	data1 = np.loadtxt(textname).T

	ydata1 = np.reshape(data1[0], [Nobsbins1, Nobsbins2]).T
	yerror1 = np.reshape(data1[1], [Nobsbins1, Nobsbins2]).T
	
	if extra == 'ratio':
		filename2 = filenames2[-1,-1]
		file_ext = filename2.split('.')[-1]
		plotname = filename2.replace('.%s'%file_ext,'_%s_matrix.png'%plotstyle)
		textname = plotname.replace('log_matrix.png', 'masses.txt')

		data2 = np.loadtxt(textname).T

		ydata2 = np.reshape(data2[0], [Nobsbins1, Nobsbins2]).T
		yerror2 = np.reshape(data2[1], [Nobsbins1, Nobsbins2]).T
	
		ydata = ydata1/ydata2
		yerror = ydata*((yerror1/ydata1)**2 + (yerror2/ydata2)**2)**0.5
	else:
		ydata = ydata1
		yerror = yerror1
	
	xdata = [(binrange1[x]+binrange1[x+1])/2 for x in xrange(Nobsbins1)]


	print xdata
	print ydata
	print yerror

	# Plotting the ueber matrix
	fig = plt.figure(figsize=(12/2,12))
	canvas = FigureCanvas(fig)

	gs_full = gridspec.GridSpec(1,2, width_ratios=[20,1], wspace=0.1)
	gs = gridspec.GridSpecFromSubplotSpec(Nobsbins2, 1, wspace=0, hspace=0, subplot_spec=gs_full[0,0])

	ax = fig.add_subplot(gs_full[0,0])
	
	for N2 in xrange(Nobsbins2):
		
		ax_sub = fig.add_subplot(gs[N2, 0])
		ax_sub.errorbar(xdata, ydata[N2], yerr=yerror[N2], color='blue', ls='', marker='o')
		if extra == 'ratio':
			ax_sub.plot(xdata, [1]*len(xdata), color='red', ls='-', marker='')
			ax_sub.set_ylim(0, 3)
		else:
			ax_sub.set_ylim(50, 325)


		ax_sub.yaxis.set_label_position('right')

		if N2 == Nobsbins2-1:
			ax_sub.set_ylabel(r'%s = %s'%(binname2, binrange2[N2]), fontsize=17)
		else:
			ax_sub.set_ylabel(r'%s'%(binrange2[N2]), fontsize=17)
		
		ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off', left='off', right='off')

		if N2 != Nobsbins2-1:
			ax_sub.tick_params(axis='x', labelbottom='off')

	# Define the labels for the plot
	xlabel = r'log(M$_{*}$ [M$_\odot$]) '
	ylabel = r'Velocity dispersion $\sigma_v$ [km/s]'
	if extra == 'ratio':
		ylabel = r'Velocity dispersion ratio $\sigma_{env}/\sigma_{shuf}$'

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	ax.xaxis.set_label_coords(0.5, -0.05)
	ax.yaxis.set_label_coords(-0.1, 0.5)

	ax.xaxis.label.set_size(17)
	ax.yaxis.label.set_size(17)

	plt.text(0.5, 1.08, plottitle1, horizontalalignment='center', fontsize=17, transform = ax.transAxes)
	plt.text(0.5, 1.05, plottitle2, horizontalalignment='center', fontsize=17, transform = ax.transAxes)

	filename = filenames1[-1,-1]
	file_ext = filename.split('.')[-1]
	
	if extra == 'ratio':
		plotname = filename.replace('.%s'%file_ext,'_masses_ratio.png')
	else:
		plotname = filename.replace('.%s'%file_ext,'_masses.png')

	plt.savefig(plotname,format='png')
	
	print
	print 'Written: Fitted mass values plot:', plotname

def plot_shear_matrix(filenames, plottitle1, plottitle2, plotstyle, binname1, binrange1, Nobsbins1, binname2, binrange2, Nobsbins2, h):
	
	# Plotting the ueber matrix
	fig = plt.figure(figsize=(13.5/4*Nobsbins2,12/4*Nobsbins1))
	canvas = FigureCanvas(fig)

	gs_full = gridspec.GridSpec(1,2, width_ratios=[20,1], wspace=0.1)
	gs = gridspec.GridSpecFromSubplotSpec(Nobsbins1, Nobsbins2, wspace=0, hspace=0, subplot_spec=gs_full[0,0])

	ax = fig.add_subplot(gs_full[0,0])
	
	if fit:
		
		masses = np.array([])
		masserrors = np.array([])
		
		for N1 in xrange(Nobsbins1):
			for N2 in xrange(Nobsbins2):

				data_x, data_y, errorh, errorl, bias = import_data(filenames[N1,N2])
				
				fitdata, mass, masserror = fit_func(data_x, data_y, errorh, SIS)
				masses = np.append(masses, mass)
				masserrors = np.append(masserrors, masserror)
	
	#	massmax = np.amax(masses)
	#	massmin = np.amin(masses)
		massmax = 250
		massmin = 0
	
		colors = (1.-(1.0*(masses-massmin)/(massmax-massmin)))
		
	else:
		color = 1
		mass = 100
	
	for N1 in xrange(Nobsbins1):
		for N2 in xrange(Nobsbins2):
			
			print '#%i: N1=%i, N2=%i'%(N1*Nobsbins2+N2+1, N1+1, N2+1)

			data_x, data_y, errorh, errorl, bias = import_data(filenames[N1,N2])
			
			# Add subplots
			if fit:
				mass = masses[N1*Nobsbins2+N2]
				masserror = masserrors[N1*Nobsbins2+N2]
				color = colors[N1*Nobsbins2+N2]
				
#			if mass >= 10:
			ax_sub = fig.add_subplot(gs[Nobsbins1-N1-1,N2], axisbg=(color, color, color))

			if 'log' in plotstyle:
				ax_sub.set_yscale('log')
				errorl[errorl>=data_y] = ((data_y[errorl>=data_y])*0.9999999999)

				plt.autoscale(enable=False, axis='both', tight=None)
				ax_sub.set_ylim(1e-2, 1e4)
				ax_sub.set_yticks([1e-2, 1e-1,1e0,1e1,1e2,1e3])
				
			if 'lin' in plotstyle:
				plt.autoscale(enable=False, axis='both', tight=None)
				ax_sub.set_ylim(-3e1, 3e1)
				plt.plot(data_x, [1]*len(data_x), ls='-', marker='', color='red')
				
			if fit:
				fitdata, mass, masserror = fit_func(data_x, data_y, errorh, SIS)
				print 'sigma:', mass, ', error:', masserror
				
				plt.plot(data_x, fitdata, ls='-', label=r'$\sigma_v$ = %g km/s'%mass, color='red')

				plt.legend(loc='upper right',ncol=1, prop={'size':12})

			ax_sub.set_xscale('log')
			ax_sub.errorbar(data_x, data_y, yerr=[errorl,errorh], color='blue', ls='', marker='o')
			ax_sub.set_xlim(1e1,1e4)

			if N1 != 0:
				ax_sub.tick_params(axis='x', labelbottom='off')
			if N2 != 0:
				ax_sub.tick_params(axis='y', labelleft='off')
			
			if N1 == Nobsbins1 - 1:
				ax_sub.xaxis.set_label_position('top')
				if N2 == 0:
					ax_sub.set_xlabel(r'%s = %s'%(binname2, binrange2[N2]), fontsize=17)
				else:
					ax_sub.set_xlabel(r'%s'%(binrange2[N2]), fontsize=17)
				
			if N2 == Nobsbins2 - 1:
				ax_sub.yaxis.set_label_position('right')
				if N1 == 0:
					ax_sub.set_ylabel(r'%s = %.3g - %.3g'%(binname1, binrange1[N1], binrange1[N1+1]), fontsize=17)
				else:
					ax_sub.set_ylabel(r'%.3g - %.3g'%(binrange1[N1], binrange1[N1+1]), fontsize=17)
			
	#			ax_sub.set_xticks([1e2,1e3])
			print


	# Turn off axis lines and ticks of the big subplot
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off', left='off', right='off')

	# Define the labels for the plot
	xlabel = r'radius R [kpc/h$_{%g}$]'%(h*100)
	ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h$_{%g}$ M$_{\odot}$/pc$^2$]'%(h*100)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	ax.xaxis.set_label_coords(0.5, -0.05)
	ax.yaxis.set_label_coords(-0.05, 0.5)

	ax.xaxis.label.set_size(17)
	ax.yaxis.label.set_size(17)

	plt.text(0.5, 1.08, plottitle1, horizontalalignment='center', fontsize=17, transform = ax.transAxes)
	plt.text(0.5, 1.05, plottitle2, horizontalalignment='center', fontsize=17, transform = ax.transAxes)

	filename = filenames[-1,-1]
	file_ext = filename.split('.')[-1]
	plotname = filename.replace('.%s'%file_ext,'_%s_matrix.png'%plotstyle)
	textname = plotname.replace('log_matrix.png', 'masses.txt')

	plt.savefig(plotname,format='png')
	print_masses(textname, masses, masserrors, Nobsbins1, Nobsbins2)
	
	print
	print 'Written: Covariance matrix plot:', plotname
	print
	print 'Written: Fit results:', textname
	print

	plt.close()
	
	return masses, masserrors

binname1 = r'log(M$_{*}$)'
#binrange1 = np.array([6.0, 9.23128796, 10.23492718, 10.73139954, 13.0])
binrange1 = np.array([6.0, 10.23492718, 10.73139954, 13.0])
#binrange1 = np.array([6.0, 10.23492718, 13.0])
#binrange1 = np.array([6.0, 11.2, 13.0])
Nobsbins1 = len(binrange1)-1	

#"""
binnames = ['envS4', '(shuff-)envS4']
binrange2 = ['Void', 'Sheet', 'Filament', 'Knot']
Nobsbins2 = len(binrange2)

"""

delta8_bins = np.loadtxt('binlimits/binlimits_rankBCG-999--999_delta8bins4_delta8-1-inf_ZB0.005-1.2_16bins_Om0.315_h100.txt').T

binname2 = r'Overdensity $\delta_8$'
binrange2 = ['%.3g - %.3g'%(delta8_bins[i], delta8_bins[i+1]) for i in xrange(len(delta8_bins)-1)]
Nobsbins2 = len(binrange2)
"""

extra = 'ratio'
fit = True

plottitle1 = ''
plottitle2 = ''

h=1.0

plotstyle = 'log'
shuffname = 'shuffenvR4'

#filenames = np.array([['/disks/shear10/brouwer_veersemeer/shearcode_output/output_delta8bins/results_shearcatalog/shearcatalog_rankBCG-999--999_delta8bin%iof%i_logmstar%g-%g_ZB0.005-1.2_logRbins10:20:2000kpc_Om0.315_h100_A.txt'%(N2+1, Nobsbins2, binrange1[N1], binrange1[N1+1]) for N2 in xrange(Nobsbins2)] for N1 in xrange(Nobsbins1)])

filenames1 = np.array([['/disks/shear10/brouwer_veersemeer/shearcode_output/output_envS4bins/results_shearcatalog/shearcatalog_rankBCG-999--999_envS4bin%iof%i_corr-logmstar%g-%g_ZB0.005-1.2_logRbins5:20:2000kpc_Om0.315_h100_A.txt'%(N2+1, Nobsbins2, binrange1[N1], binrange1[N1+1]) for N2 in xrange(Nobsbins2)] for N1 in xrange(Nobsbins1)])
filenames2 = np.array([['/disks/shear10/brouwer_veersemeer/shearcode_output/output_%sbins/results_shearcatalog/shearcatalog_rankBCG-999--999_%sbin%iof%i_corr-logmstar%g-%g_ZB0.005-1.2_logRbins5:20:2000kpc_Om0.315_h100_A.txt'%(shuffname, shuffname, N2+1, Nobsbins2, binrange1[N1], binrange1[N1+1]) for N2 in xrange(Nobsbins2)] for N1 in xrange(Nobsbins1)])

filenames = [filenames1, filenames2]

for i in xrange(len(filenames)):
	
	binname2 = binnames[i]
	masses, masserrors = plot_shear_matrix(filenames[i], plottitle1, plottitle2, plotstyle, binname1, binrange1, Nobsbins1, binname2, binrange2, Nobsbins2, h)

if fit:
	plot_masses(filenames1, filenames2, binname1, binrange1, Nobsbins1, binname2, binrange2, Nobsbins2)
