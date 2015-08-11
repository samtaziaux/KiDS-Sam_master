#!/usr/bin/python

"This contains all the modules that are needed to calculate the shear profile catalog and the covariance."

import pyfits
import numpy as np
import distance
import sys
import os
import time
from astropy import constants as const, units as u
import glob
import gc


# Important constants
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
pix = 0.187 # Used to translate pixel to arcsec
alpha = 0.057 # Used to calculate m
beta = -0.37 # Used to calculate m
inf = np.inf


def input_variables():
	
	Nsplit = int(sys.argv[1])-1 # The number of this particular core/split
	Ncores = int(sys.argv[2]) # The number of cores/splits over which this calculation will be spread
	binnum = int(sys.argv[3]) # Number of the bin for which the covariance will be computed (when not binning use '0')
	blindcat = str(sys.argv[4]) # The blind catalog (A, B, C, D)
	centering = str(sys.argv[5]) # The center definition: Cen, IterCen or BCG
	rankmin = float(sys.argv[6]) # The minimum rank of galaxies (rankmin <= galaxy rank)
	rankmax = float(sys.argv[7]) # The maximum rank of galaxies (galaxy rank <= rankmax), for no boundery use: 'inf'
	Nfofmin = float(sys.argv[8]) # The minimum number of group members (Nfofmin <= group members)
	Nfofmax = float(sys.argv[9]) # The maximum number of group members (group members <= Nfofmax), for no boundery use: 'inf'
	ZBmin = float(sys.argv[10]) # The minimum ZB (best redshift) of the sources
	ZBmax = float(sys.argv[11]) # The maximum ZB (best redshift) of the sources
	binname = str(sys.argv[12]) # Name of the binning observable in the merged catalog (for no binning use: 'No')
	path_obsbins = str(sys.argv[13]) # The path to the limits of the observable bins
	Nobsbins = int(sys.argv[14]) # The total number of observable bins
	obslim = str(sys.argv[15]) # Name of the limit observable 
	obslim_min = str(sys.argv[16]) # The minimum value of the limit observable
	obslim_max = str(sys.argv[17]) # The maximum value of the limit observable
	path_Rbins = str(sys.argv[18]) # The path to the limits of the radial bins
	path_output = str(sys.argv[19]) # The path where the output will be written
	purpose = str(sys.argv[20]) # The type of output that will be generated (shearcatalog, randomcatalog, covariance)
	O_matter = float(sys.argv[21]) # The value of Omega(matter) (Omega(Lambda) = 1 - Omega(matter))
	h = float(sys.argv[22]) # The value of the reduced Hubble constant h (H = h*100km/s/Mpc)
	filename_addition = str(sys.argv[23]) # Additional information about the plot
	path_kidscats = str(sys.argv[24]) # The path to the KiDS catalogues

	if blindcat=='A':
		blindcatnum=0
	if blindcat=='B':
		blindcatnum=1
	if blindcat=='C':
		blindcatnum=2
	if blindcat=='D':
		blindcatnum=3

	# The values for Omega(matter) and Omega(lambda)
	O_lambda = 1-O_matter

	# Defining filename addition
	if filename_addition == 'No':
		filename_addition = ''
	else:
		filename_addition = '_%s'%filename_addition

	if purpose == 'binlimits':
		purpose = 'shearcatalog'

	# Create all necessary folders once

	# Path containing the output folders
	path_output = '%s/output_%sbins%s'%(path_output, binname, filename_addition)
	path_catalogs = '%s/catalogs'%(path_output.rsplit('/',1)[0])

	# Path to the output splits and results
	path_splits = '%s/splits_%s'%(path_output, purpose)
	path_results = '%s/results_%s'%(path_output, purpose)
	
	if (Nsplit==0) and (blindcat=='A') and (binnum<=1):
	
#		print 'Nsplit:', Nsplit
#		print 'blindcat:', blindcat
#		print 'binnum:', binnum
	
		if not os.path.isdir(path_output):
			os.makedirs(path_output)
			
		if not os.path.isdir(path_catalogs):
			os.makedirs(path_catalogs)
		
		if not os.path.isdir(path_splits):
			os.makedirs(path_splits)

		if not os.path.isdir(path_results):
			os.makedirs(path_results)
	
	
	# Check if the observable limits are files or floats
	# Obslim_min
	if os.path.isfile(obslim_min): # from a file
		obslim_min = np.loadtxt(obslim_min).T
		if len(obslim_min) != Nobsbins:
			print 'Number of minimum observable limits is not the same as number of observable bins!'
			quit()
	else:
		try:
			obslim_min = float(obslim_min)
			obslim_min = np.array([obslim_min]*Nobsbins)
		except:
			print 'Minimum observable limit "%s" is not a file or float!'%obslim_min
			quit()

	# Obslim_max
	if os.path.isfile(obslim_max): # from a file
		obslim_max = np.loadtxt(obslim_max).T
		if len(obslim_max) != Nobsbins:
			print 'Number of maximum observable limits is not the same as number of observable bins!'
			quit()
	else:
		try:
			obslim_max = float(obslim_max)
			obslim_max = np.array([obslim_max]*Nobsbins)
		except:
			print 'Maximum observable limit "%s" is not a file or float!'%obslim_max
			quit()
	print 'Observable limits: %s <= %s <= %s'%(obslim_min, obslim, obslim_max)
	
	if 'catalog' in purpose:
		
		# Path to the output splits and results
		path_splits = '%s/splits_%s'%(path_catalogs, purpose)
		path_results = '%s/results_%s'%(path_catalogs, purpose)

		if (Nsplit==0) and (blindcat=='A') and (binnum<=1):
			
			if not os.path.isdir(path_splits):
				os.makedirs(path_splits)
				
			if not os.path.isdir(path_results):
				os.makedirs(path_results)
	

	splitslist = [] # This list will contain all created splits
	# Determining Ncat, the number of existing catalogs
	if ('random' in purpose):

		# Defining the name of the output files
		if os.path.isfile(path_Rbins): # from a file
			name_Rbins = path_Rbins.split('.')[0]
			name_Rbins = name_Rbins.split('/')[-1]
		else:
			name_Rbins = 'logRbins%skpc'%path_Rbins
		
		if all(np.array([rankmin,rankmax]) > 0):
			filename_var = 'group_ZB%g-%g_%s_Om%g_h%g'%(ZBmin, ZBmax, name_Rbins, O_matter, h*100)
		else:
			filename_var = 'all_ZB%g-%g_%s_Om%g_h%g'%(ZBmin, ZBmax, name_Rbins, O_matter, h*100)
		
		for Ncat in xrange(100):
			outname = '%s/%s_%i_%s%s_split%iof*.fits'%(path_splits.replace('bootstrap', 'catalog'), purpose.replace('bootstrap', 'catalog'), Ncat+1, filename_var, filename_addition, Nsplit+1)
			splitfiles = glob.glob(outname)
			splitslist = np.append(splitslist, splitfiles)
			
			if len(splitfiles) == 0:
				break
	else:
		Ncat = 1
	
	print
	print 'Running:', purpose
	
	return Nsplit, Ncores, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, binnum, path_obsbins, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, path_output, path_catalogs, path_splits, path_results, purpose, O_matter, O_lambda, h, filename_addition, Ncat, splitslist, blindcat, blindcatnum, path_kidscats


def define_filename_var(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, ZBmin, ZBmax, binname, binnum, Nobsbins, obslim, obslim_min, obslim_max, path_Rbins, O_matter, h): # Define the list of variables for the output filename
	
	
	# Defining the name of the output files
	if os.path.isfile(path_Rbins): # from a file
		name_Rbins = path_Rbins.split('.')[0]
		name_Rbins = name_Rbins.split('/')[-1]
	else:
		name_Rbins = 'logRbins%skpc'%path_Rbins
	
		
	if 'catalog' in purpose:

		if all(np.array([rankmin,rankmax]) > 0):

			if centering == 'Cen':
				filename_var = 'groupCen'
			else:
				filename_var = 'group'
			
			var_print = 'Group catalogue, center = %s,'%(centering)

		else:
			filename_var = 'all'

			var_print = 'Galaxy catalogue,'
		
	else: # Binnning information of the groups

		filename_var = 'rank%s%g-%g'%(centering, rankmin, rankmax)
		var_print = '%g<=rank%s<=%g,'%(rankmin, centering, rankmax)
		

		if all(np.array([rankmin,rankmax]) > 0):
			filename_var = '%s_Nfof%g-%g'%(filename_var, Nfofmin, Nfofmax)
			var_print = '%s %g<=Nfof<=%g,'%(var_print, Nfofmin, Nfofmax)
		
		if binname!='No': # If there is binning
			filename_var = '%s_%sbin%sof%i'%(filename_var, binname, binnum, Nobsbins)
			var_print = '%s %i %s-bins,'%(var_print, Nobsbins, binname)

		if obslim!='No': # If there is binning
			if type(binnum) == str:
				obslim_min = obslim_min[0]
				obslim_max = obslim_max[-1]
			else:
				obslim_min = obslim_min[binnum-1]
				obslim_max = obslim_max[binnum-1]
				
			filename_var = '%s_%s%g-%g'%(filename_var, obslim, obslim_min, obslim_max)
			var_print = '%s %s-limit: %g - %g,'%(var_print, obslim, obslim_min, obslim_max)
			

	filename_var = '%s_ZB%g-%g_%s_Om%g_h%g'%(filename_var, ZBmin, ZBmax, name_Rbins, O_matter, h*100)

	if 'catalog' in purpose:
		print 'Lens selection: %s.'%(var_print[0:-1])
		print
	
	return filename_var


def define_filename_splits(path_splits, purpose, filename_var, Nsplit, Ncores, filename_addition, blindcat):

	
	# Defining the names of the shear/random catalog
	if 'covariance' in purpose:
		splitname = '%s/%s_%s%s_%s.fits'%(path_splits, purpose, filename_var, filename_addition, Nsplit) # Here Nsplit = kidscatname
	if 'bootstrap' in purpose:
		splitname = '%s/%s_%s%s_%s.fits'%(path_splits, purpose, filename_var, filename_addition, blindcat)
	if 'catalog' in purpose:
		splitname = '%s/%s_%s%s_split%iof%i.fits'%(path_splits, purpose, filename_var, filename_addition, Nsplit, Ncores)
	
	return splitname

	
def define_filename_results(path_results, purpose, filename_var, filename_addition, Nsplit, blindcat): # Paths to the resulting files

	if 'catalog' in path_results:
		resultname = '%s/%s_%s%s.fits'%(path_results, purpose, filename_var, filename_addition)
	else:
		resultname = '%s/%s_%s%s_%s.txt'%(path_results, purpose, filename_var, filename_addition, blindcat)
	
	return resultname


def import_data(path_Rbins, path_kidscats, centering, purpose, Ncat, rankmin, rankmax, O_matter, O_lambda, h, binname, obslim): # Importing all GAMA and KiDS data, and information on radial bins and lens-field matching.

	Rmin, Rmax, Rbins, Rcenters, nRbins = import_Rrange(path_Rbins)
	galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, obslist, obslimlist = import_gamacat(centering, purpose, Ncat, rankmin, rankmax, O_matter, O_lambda, h, binname, obslim)
	kidscoord, kidscat_end = run_kidscoord(path_kidscats)
	catmatch, kidscats, galIDs_infield = run_catmatch(kidscoord, galIDlist, galRAlist, galDEClist, Dallist, Rmax, purpose)

	return catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, Rcenters, nRbins, galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, obslist, obslimlist


def import_Rrange(path_Rbins): # Radial bins around the lenses

	if os.path.isfile(path_Rbins): # from a file

		# Start, End, number of steps and step size of the radius R
		Rrangefile = np.loadtxt(path_Rbins).T
		Rmin = Rrangefile[0,0]
		Rmax = Rrangefile[-1,-1]
		Rbins = np.append(Rrangefile[0],Rmax)
		Rcenters = Rrangefile[1]
		nRbins = len(Rcenters)
		
	else: # from a specified number (of bins)
		try:
			# Start, End, number of steps and step size of the radius R (logarithmic 10^x)
			binlist = path_Rbins.split(':')
			
			nRbins = int(binlist[0])
			Rmin = float(binlist[1])
			Rmax = float(binlist[2])
			Rstep = (np.log10(Rmax)-np.log10(Rmin))/(nRbins)
			Rbins = 10.**np.arange(np.log10(Rmin), np.log10(Rmax), Rstep)
			Rbins = np.append(Rbins,Rmax)
			Rcenters = np.array([(Rbins[r]+Rbins[r+1])/2 for r in xrange(nRbins)])
	
		except:
			print 'Observable bin file does not exist:', path_Rbins
			exit()
		
	"""
	print 'path_Rbins', path_Rbins
	print 'Using: %i radial bins between %.1f and %.1f kpc'%(nRbins, Rmin, Rmax)
	print 'Rmin', Rmin 
	print 'Rmax', Rmax
	print 'Rbins', Rbins
	print 'Rcenters', Rcenters
	print 'nRbins', nRbins
	"""
	
	return Rmin, Rmax, Rbins, Rcenters, nRbins


def import_gamacat(centering, purpose, Ncat, rankmin, rankmax, O_matter, O_lambda, h, binname, obslim): # Load the properties (RA, DEC, Z -> dist) of the gal in the GAMA catalogue
	
	randomcatname = '/disks/shear9/brouwer/shearprofile/shear_2.1/gen_ran_out.randoms.fits'
	starcatname = '/disks/shear9/viola/SG/src/All_DR2/All.DR2.clean.cat'

#	"""
	# Normal GAMA catalogues
		
	mergedcatdir = '/data2/brouwer/MergedCatalogues'
	if os.path.isdir(mergedcatdir):
		pass
	else:
		mergedcatdir = '/disks/shear10/brouwer_veersemeer/MergedCatalogues'
	
	if all(np.array([rankmin,rankmax]) > 0):
		mergedcatname = '%s/CatalogueGroups_v1.0_shuffenv+dens.fits'%mergedcatdir
	else:
		mergedcatname = '%s/ShearMergedCatalogueAll_sv0.8_shuffdeltaR.fits'%mergedcatdir
	
	"""
	
	# New GAMA catalogues
	mergedcatdir = '/data2/brouwer/MergedCatalogues/DMUG3Cv08/groups'
	mergedcatname = '%s/G3CGalv08.fits'%mergedcatdir
	
	"""

	print 'Importing GAMA catalogue:', mergedcatname	
	mergedcat = pyfits.open(mergedcatname)
	
	galIDlist = mergedcat[1].data['ID'] # IDs of all galaxies
	galZlist = mergedcat[1].data['Z'] # Central Z of the galaxy
	galranklist = mergedcat[1].data['Rank%s'%centering] # Rank of the galaxy
	
	if all(np.array([rankmin,rankmax]) > 0):
		Nfoflist = mergedcat[1].data['Nfof'] # Multiplicity of each group
		groupIDlist = mergedcat[1].data['GroupID_1'] # IDs of all groups
	else:
		Nfoflist = np.array([])
		groupIDlist = np.array([])
	
	if centering == 'Cen':
		galRAlist = mergedcat[1].data['CenRA'] # Central RA of the galaxy (in degrees)
		galDEClist = mergedcat[1].data['CenDEC'] # Central DEC of the galaxy (in degrees)
		galZlist = mergedcat[1].data['Zfof'] # Z of the group
	else:
		galRAlist = mergedcat[1].data['RA'] # Central RA of the galaxy (in degrees)
		galDEClist = mergedcat[1].data['DEC'] # Central DEC of the galaxy (in degrees)
		
	if binname != 'No':
		obslist = mergedcat[1].data[binname.replace('corr-', '')] # Chosen binning observable of each group
	else:
		obslist = np.array([])
		
	if obslim != 'No':
		obslimlist = mergedcat[1].data[obslim.replace('corr-', '')] # Chosen binning observable of each group
	else:
		obslimlist = np.array([])

	# print 'Importing merged catalog:', mergedcatname

	if 'random' in purpose or 'star' in purpose:
		# Determine RA and DEC for the random/star catalogs
		Ncatmin = Ncat * len(galIDlist) # The first item that will be chosen from the catalog
		Ncatmax = (Ncat+1) * len(galIDlist) # The last item that will be chosen from the catalog
	
	if 'random' in purpose:
		print 'Importing:', randomcatname
		randomcat = pyfits.open(randomcatname)
		
		galRAlist = randomcat[1].data['ra'][Ncatmin : Ncatmax]
		galDEClist = randomcat[1].data['dec'][Ncatmin : Ncatmax]
	
	if 'star' in purpose:
		print 'Importing:', starcatname
		starcat = np.loadtxt(starcatname).T

		galRAlist = starcat[0]
		galDEClist = starcat[1]
		galIDlist = np.arange(len(galRAlist))
		galZlist = np.hstack([galZlist]*int(len(galIDlist)/len(galZlist)+1))[0:len(galIDlist)]
	
	Dcllist = np.array([distance.comoving(z, O_matter, O_lambda, h) for z in galZlist]) # The comoving distance to the galaxy center (in kpc/h, where h is the dimensionless Hubble constant)
	Dallist = Dcllist/(1+galZlist) # The angular diameter distance to the galaxy center (in kpc/h)

	# Corrections on GAMA catalog observables
	if ('AngSep' in binname and 'corr' in binname) or ('AngSep' in obslim and 'corr' in obslim):
		print 'Applying AngSep correction'
		
		Dclgama = np.array([distance.comoving(z, 0.25, 0.75, 1.) for z in galZlist])
		corr_list = Dcllist/Dclgama
		if 'AngSep' in binname:	
			obslist = obslist * corr_list
		else:
			obslimlist = obslimlist * corr_list

#	if ('logmstar' in binname and 'corr' in binname) or ('logmstar' in obslim and 'corr' in obslim):
	if ('logmstar' in binname) or ('logmstar' in obslim):

		print 'Applying logmstar fluxscale correction'
		
		fluxscalelist = mergedcat[1].data['fluxscale'] # Fluxscale, needed for stellar mass correction
		corr_list = np.log10(fluxscalelist)# - 2*np.log10(h/0.7)
		if 'logmstar' in binname:
			obslist = obslist + corr_list
		else:
			obslimlist = obslimlist + corr_list
	
	return galIDlist, groupIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist, Nfoflist, galranklist, obslist, obslimlist


def run_kidscoord(path_kidscats): # Finding the central coordinates of the KiDS fields

	# Load the names of all KiDS catalogues from the specified folder
	kidscatlist = os.listdir(path_kidscats)
#	kidscatlist = [kidscatlist[0]]

	# Remove all files from the list that are not KiDS catalogues
	for x in kidscatlist:
		if 'KIDS_' not in x:
			kidscatlist.remove(x)

	# Create the dictionary that will hold the names of the KiDS catalogues with their RA and DEC
	kidscoord = dict()

	for i in range(len(kidscatlist)):
		# Of the KiDS file names, keep only "KIDS_RA_DEC"
		
		kidscatstring = kidscatlist[i].split('_',3)
		kidscatname = '_'.join(kidscatstring[0:3])

		# Extract the central coordinates of the field from the file name
		coords = '_'.join(kidscatstring[1:3])
		coords = ((coords.replace('p','.')).replace('m','-')).split('_')
		
		# Fill the dictionary with the catalog's central RA and DEC: {"KIDS_RA_DEC": [RA, DEC]}
		kidscoord[kidscatname] = [float(coords[0]),float(coords[1])]
		
		kidscat_end = kidscatstring[-1]
		
#		print kidscatname, kidscoord[kidscatname]
#	print 
#	print 'Total:', len(kidscatlist), 'KiDS fields'

	return kidscoord, kidscat_end


def run_catmatch(kidscoord, galIDlist, galRAlist, galDEClist, Dallist, Rmax, purpose): # Create a dictionary of KiDS fields that contain the corresponding galaxies.

	Rfield = np.radians(np.sqrt(2)/2) * Dallist
	
	catmatch = dict()
	totgalIDs = np.array([])

	# Adding the lenses to the list that are inside each field
	for kidscat in kidscoord.keys():
	
		# The RA and DEC of the KiDS catalogs
		catRA = kidscoord[kidscat][0]
		catDEC = kidscoord[kidscat][1]
		
		# The difference in RA and DEC between the field and the lens centers
		dRA = catRA-galRAlist
		dDEC = catDEC-galDEClist
		
		# Masking the lenses that are outside the field
		coordmask = (abs(dRA) < 0.5) & (abs(dDEC) < 0.5)
		galIDs = (galIDlist[coordmask])
		
		# Add the proper lenses to the list with all matched lenses
		totgalIDs = np.append(totgalIDs, galIDs)
		
		# If there are matched lenses in this field, add it to the catmatch dictionary
		if len(galIDs)>0:
			
			catmatch[kidscat] = np.array([])
			catmatch[kidscat] = np.append(catmatch[kidscat], galIDs, 0) # Creating a dictionary that contains the corresponding Gama galaxies for each KiDS field.
			

	kidscats = catmatch.keys() # The list of fields with lens centers in them
	galIDs_infield = totgalIDs # The galaxies that have their centers in a field
	
	# Adding the lenses outside the fields to the dictionary
	for kidscat in kidscoord.keys():

		# The RA and DEC of the KiDS catalogs
		catRA = kidscoord[kidscat][0]
		catDEC = kidscoord[kidscat][1]
		
		# Defining the distance R between the lens center and its surrounding background sources
		catR = Dallist*np.arccos(np.cos(np.radians(galDEClist))*np.cos(np.radians(catDEC))*np.cos(np.radians(galRAlist-catRA))+np.sin(np.radians(galDEClist))*np.sin(np.radians(catDEC)))
		
#		coordmask = (catR<(Rmax+Rfield))
		coordmask = (catR<Rmax)
		
		galIDs = np.array(galIDlist[coordmask])
		
#		galRs = catR[coordmask]
#		galRAs = galRAlist[coordmask]
#		galDECs = galDEClist[coordmask]
#		Dals = Dallist[coordmask]

		if 'bootstrap' in purpose:
			lensmask = np.logical_not(np.in1d(galIDs, totgalIDs))
			galIDs = galIDs[lensmask]
		else:
			if kidscat in kidscats:
				lensmask = np.logical_not(np.in1d(galIDs, catmatch[kidscat]))
				galIDs = galIDs[lensmask]
		
		totgalIDs = np.append(totgalIDs, galIDs)
		
		if len(galIDs)>0:
			if kidscat not in kidscats:
				catmatch[kidscat] = []

			catmatch[kidscat] = np.append(catmatch[kidscat], galIDs, 0)
			
			
#			print kidscat, Rmax
#			print 'Average(R):', np.average(catR[coordmask]), ', Max(R):', np.amax(catR[coordmask]), ', Average(RA):', np.average(galRAlist[coordmask]), ', Average(DEC):', np.average(galDEClist[coordmask])

	kidscats = catmatch.keys()
	
	print 'Matched fields:', len(kidscats), ', Matched field-galaxy pairs:', len(totgalIDs), ', Matched galaxies:', len(np.unique(totgalIDs)), ', Percentage(Matched galaxies):',  float(len(np.unique(totgalIDs)))/float(len(galIDlist))*100, '%'
	print
	
	return catmatch, kidscats, galIDs_infield


def split(seq, size): # Split up the list of KiDS fields for parallel processing

	newseq = []
	splitsize = len(seq)/size

	for i in range(size-1):
		newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
	newseq.append(seq[int(round((size-1)*splitsize)):len(seq)])

	return newseq


def import_kidscat(path_kidscats, kidscatname, kidscat_end, ZBmin, ZBmax): # Import and mask all used data from the sources in this KiDS field

	kidscatfile = '%s/%s_%s'%(path_kidscats, kidscatname, kidscat_end) # Full directory & name of the corresponding KiDS catalogue
	kidscat = pyfits.open(kidscatfile)

	srcNr = kidscat[1].data['SeqNr'] # List of the ID's of all sources in the KiDS catalogue
	srcRA = kidscat[1].data['ALPHA_J2000'] # List of the RA's of all sources in the KiDS catalogue
	srcDEC = kidscat[1].data['DELTA_J2000'] # List of the DEC's of all sources in the KiDS catalogue
	w = kidscat[1].data['weight'] # The weight of the sources in the stacking
	srcPZ = kidscat[1].data['PZ_full'] # Full P(z) probability function
	SN = kidscat[1].data['SNratio'] # The Signal to Noise of the sources (needed for bias)

	# Masking: we remove sources with weight=0, too high/low redshift ZB, and those masked by the catalog 
	srcZB = kidscat[1].data['Z_B'] # "Best" Z value from the catalog (to apply Z cuts)
	manmask = kidscat[1].data['MAN_MASK'] # The manual masking of bad sources (0=good, 1=bad)	
	

	if 'DR2' in path_kidscats:
		
		e1_A = kidscat[1].data['e1_A'] # The ellipticity of the sources in the X-direction
		e1_B = kidscat[1].data['e1_B']
		e1_C = kidscat[1].data['e1_C']
		e1_D = kidscat[1].data['e1_D']
		e2_A = kidscat[1].data['e2_A']
		e2_B = kidscat[1].data['e2_B']
		e2_C = kidscat[1].data['e2_C']
		e2_D = kidscat[1].data['e2_D']

		c1_A = kidscat[1].data['c1_A'] # The ellipticity correction of the sources in the X-direction
		c1_B = kidscat[1].data['c1_B']
		c1_C = kidscat[1].data['c1_C']
		c1_D = kidscat[1].data['c1_D']
		c2_A = kidscat[1].data['c2_A']
		c2_B = kidscat[1].data['c2_B']
		c2_C = kidscat[1].data['c2_C']
		c2_D = kidscat[1].data['c2_D']
	
		e1 = np.transpose(np.array([e1_A-c1_A, e1_B-c1_B, e1_C-c1_C, e1_D-c1_D])) # The corrected e1 for all blind catalogs
		e2 = np.transpose(np.array([e2_A-c2_A, e2_B-c2_B, e2_C-c2_C, e2_D-c2_D])) # The corrected e2 for all blind catalogs

		srcm = kidscat[1].data['m_cor'] # The multiplicative bias m
		usedmask = (ZBmin < srcZB) & (srcZB < ZBmax) & (w > 0.) & (srcm < 0) & (0 < SN) & (manmask==0) & (-1 < c1_A)

	else:

		print '*** Warning: You are using an old KiDS catalog: %s ***'%path_kidscats

		"""		
		e1 = np.transpose(np.array([e1_A, e1_B, e1_C, e1_D])) # e1 for all blind catalogs
		e2 = np.transpose(np.array([e2_A, e2_B, e2_C, e2_D])) # e2 for all blind catalogs
		
		size = kidscat[1].data['scalelength']*pix # The size of the sources in arcsec (needed for bias)
		
		srcm = (beta/np.log(SN))*np.exp(-alpha*size*SN) # The multiplicative bias m	
		usedmask = (ZBmin < srcZB) & (srcZB < ZBmax) & (w > 0.) & (srcm < 0) & (0 < SN) & (manmask==0)
		"""
		
		e1_A = kidscat[1].data['e1'] # The ellipticity of the sources in the X-direction
		e1_B = kidscat[1].data['e1']
		e1_C = kidscat[1].data['e1']
		e1_D = kidscat[1].data['e1']
		e2_A = kidscat[1].data['e2']
		e2_B = kidscat[1].data['e2']
		e2_C = kidscat[1].data['e2']
		e2_D = kidscat[1].data['e2']
		
		e1 = np.transpose(np.array([e1_A-c1_A, e1_B-c1_B, e1_C-c1_C, e1_D-c1_D])) # The corrected e1 for all blind catalogs
		e2 = np.transpose(np.array([e2_A-c2_A, e2_B-c2_B, e2_C-c2_C, e2_D-c2_D])) # The corrected e2 for all blind catalogs

		srcm = kidscat[1].data['m_cor'] # The multiplicative bias m
		usedmask = (ZBmin < srcZB) & (srcZB < ZBmax) & (w > 0.) & (srcm < 0) & (0 < SN) & (manmask==0) & (-1 < c1_A)
		
		
	srcNr = srcNr[usedmask]
	srcRA = srcRA[usedmask]
	srcDEC = srcDEC[usedmask]
	w = w[usedmask]
	srcPZ = srcPZ[usedmask]
	srcm = srcm[usedmask] # The multiplicative bias m
	
	e1 = e1[usedmask]
	e2 = e2[usedmask]

	return srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm


def calc_variance(e1_varlist, e2_varlist, w_varlist): # Calculating the variance of the ellipticity for this source selection

	e1_mean = np.sum(w_varlist*e1_varlist, 1)/np.sum(w_varlist)
	e2_mean = np.sum(w_varlist*e2_varlist, 1)/np.sum(w_varlist)
	
	e1_mean = np.reshape(e1_mean, [len(e1_mean), 1])
	e2_mean = np.reshape(e1_mean, [len(e2_mean), 1])
	
	weight = np.sum(w_varlist)/(np.sum(w_varlist)**2 - np.sum(w_varlist**2))
	
	var_e1 = weight * np.sum(w_varlist*(e1_varlist-e1_mean)**2, 1)
	var_e2 = weight * np.sum(w_varlist*(e2_varlist-e2_mean)**2, 1)
	
	variance = np.mean([var_e1, var_e2], 0)

	print 'Variance (A,B,C,D):', variance
	print 'Sigma (A,B,C,D):', variance**0.5

	return variance


def define_obsbins(obslist, binname, path_obsbins, binnum, lenssel): # Binnning information of the groups

	if binname == 'No':
		binrange = [-999, -999]
		binnum = 1
	
	else:
		# If the binning is defined by a file
		if os.path.isfile(path_obsbins):
			binrange = np.loadtxt(path_obsbins).T
			Nobsbins = len(binrange)-1 # Number of observable bins

		# Otherwise, check if the binning is given by a number
		else:
			try:
				Nobsbins = int(path_obsbins) # Number of observable bins
				
				nanmask = (-inf < obslist) & (obslist < inf) & lenssel
				obslist = obslist[nanmask]
				
				# Min/max value of the observable
				obslist_min = np.amin(obslist)
				obslist_max = np.amax(obslist)

				# Create a number of observable bins of containing an equal number of objects
				sorted_obslist = np.sort(obslist) # Sort the observable values
				obsbin_size = len(obslist)/Nobsbins # Determine the number of objects in each bin
				
				binrange = np.array([]) # This array will contain the binning range
				for o in xrange(Nobsbins): # For every observable bin...
					binrange = np.append(binrange, sorted_obslist[o*obsbin_size]) # Append the observable value that contains the determined number of objects
				binrange = np.append(binrange,obslist_max) # Finally, append the max value of the observable
				
			except:
				print 'Observable bin file does not exist:', path_obsbins
				exit()

		print '%i %s-bins:'%(Nobsbins, binname), binrange

	# The current binning values
	bins = np.sort([binrange[binnum-1], binrange[binnum]])
	binmin = float(bins[0])
	binmax = float(bins[1])
	if binmin > binmax:
		print 'Error: Bin minimum is greater than bin maximum!'
		quit()
		
	return binrange, binmin, binmax


def define_lenssel(purpose, galranklist, rankmin, rankmax, Nfoflist, Nfofmin, Nfofmax, binname, binnum, obslist, binmin, binmax, obslim, obslimlist, obslim_min, obslim_max): # Masking the lenses according to the appropriate lens selection and the current KiDS field

	lenssel = [True]*len(galranklist)

	# If the lenses aregalaxies...
	if not 'star' in purpose:
		# The lens selection depends on the galaxy rank
		lenssel = lenssel & (rankmin<=galranklist)&(galranklist<=rankmax)
	
	# If only group members are selected...
	if all(np.array([rankmin, rankmax]) > 0):
		# The group member selection depends on the number of group members Nfof
		lenssel = lenssel & (Nfofmin<=Nfoflist)&(Nfoflist<=Nfofmax)

	# If there is observable binning...
	if binname != 'No':
		# The galaxy selection depends on observable
		lenssel = lenssel & (binmin<=obslist)&(obslist<binmax)
		
	# If there is observable binning...
	if obslim != 'No':
		# The galaxy selection depends on observable
		lenssel = lenssel & (obslim_min[binnum-1]<=obslimlist)&(obslimlist<=obslim_max[binnum-1])

	return lenssel


def mask_gamacat(purpose, matched_galIDs, lenssel, centering, galranklist, galIDlist, galRAlist, galDEClist, galZlist, Dcllist, Dallist): # Mask all GAMA galaxies that are not in this field
	
	# Find the selected lenses that lie in this KiDS field

	if 'star' in purpose:
		galIDmask = np.in1d(galIDlist, matched_galIDs)
	else:
		galIDmask = np.in1d(galIDlist, matched_galIDs) & lenssel
	
	galIDs = galIDlist[galIDmask]
	galRAs = galRAlist[galIDmask]
	galDECs = galDEClist[galIDmask]
	galZs = galZlist[galIDmask]
	Dcls = Dcllist[galIDmask]
	Dals = Dallist[galIDmask]

	return galIDs, galRAs, galDECs, galZs, Dcls, Dals, galIDmask


def mask_shearcat(lenssel, galIDlist, gammatlist, gammaxlist, wk2list, w2k2list, srcmlist): # Mask the galaxies in the shear catalog
	
	galIDs = galIDlist[lenssel]
	gammats = gammatlist[lenssel]
	gammaxs = gammaxlist[lenssel]
	wk2s = wk2list[lenssel]
	w2k2s = w2k2list[lenssel]
	srcms = srcmlist[lenssel]

	return galIDs, gammats, gammaxs, wk2s, w2k2s, srcms
	

def calc_Sigmacrit(Dcls, Dals, Dcsbins, srcPZ): # Calculate Sigma_crit (=1/k) and the weight mask for every lens-source pair

	# Calculate the values of Dls/Ds for all lens/source-redshift-bin pair
	Dcls, Dcsbins = np.meshgrid(Dcls, Dcsbins)
	DlsoDs = (Dcsbins-Dcls)/Dcsbins
	Dcls = [] # Empty unused lists
	Dcsbins = []

	# Mask all values with Dcl=0 (Dls/Ds=1) and Dcl>Dcsbin (Dls/Ds<0)
	DlsoDsmask = np.logical_not((0.< DlsoDs) & (DlsoDs < 1.))
	DlsoDs = np.ma.filled(np.ma.array(DlsoDs, mask = DlsoDsmask, fill_value = 0))
	DlsoDsmask = [] # Empty unused lists
	
	# Matrix multiplication that sums over P(z), to calculate <Dls/Ds> for each lens-source pair
	DlsoDs = np.dot(srcPZ, DlsoDs).T

	# Calculate the values of k (=1/Sigmacrit)
	Dals = np.reshape(Dals,[len(Dals),1])
	k = 1 / ((c.value**2)/(4*np.pi*G.value) * 1/(Dals*DlsoDs/1e3)) # k = 1/Sigmacrit
	DlsoDs = [] # Empty unused lists
	Dals = []
	
	# Create the mask that removes all sources with k not between 0 and infinity
	kmask = np.logical_not((0. < k) & (k < inf))
	
	gc.collect()
	
	return k, kmask


def calc_shear(Dals, galRAs, galDECs, srcRA, srcDEC, e1, e2, Rmin, Rmax): # Calculate the projected distance (srcR) and the shear (gamma_t and gamma_x) of every lens-source pair.
	
	galRA, srcRA = np.meshgrid(galRAs, srcRA)
	galDEC, srcDEC = np.meshgrid(galDECs, srcDEC)
	
	# Defining the distance R and angle phi between the lens' center and its surrounding background sources
	srcR = (Dals * np.arccos(np.cos(np.radians(galDEC))*np.cos(np.radians(srcDEC))*np.cos(np.radians(galRA-srcRA)) + np.sin(np.radians(galDEC))*np.sin(np.radians(srcDEC))))

	# Masking all lens-source pairs that have a relative distance beyond the maximum distance Rmax
	Rmask = np.logical_not((Rmin < srcR) & (srcR < Rmax))
#	print float(np.sum(Rmask))/float(np.size(Rmask))
	
	galRA = np.ma.filled(np.ma.array(galRA, mask = Rmask, fill_value = 0))
	srcRA = np.ma.filled(np.ma.array(srcRA, mask = Rmask, fill_value = 0))
	galDEC = np.ma.filled(np.ma.array(galDEC, mask = Rmask, fill_value = 0))
	srcDEC = np.ma.filled(np.ma.array(srcDEC, mask = Rmask, fill_value = 0))
	srcR = np.ma.filled(np.ma.array(srcR, mask = Rmask, fill_value = 0)).T

	# Calculation the sin/cos of the angle (phi) between the gal and its surrounding galaxies
	theta = np.arccos(np.sin(np.radians(galDEC))*np.sin(np.radians(srcDEC))+np.cos(np.radians(galDEC))*np.cos(np.radians(srcDEC))*np.cos(np.radians(galRA-srcRA))) # in radians
	incosphi = ((-np.cos(np.radians(galDEC))*(np.radians(galRA-srcRA)))**2-(np.radians(galDEC-srcDEC))**2)/(theta)**2
	insinphi = 2*(-np.cos(np.radians(galDEC))*(np.radians(galRA-srcRA)))*np.radians(galDEC-srcDEC)/(theta)**2
	
	incosphi = incosphi.T
	insinphi = insinphi.T
	
	return srcR, incosphi, insinphi

	
def calc_shear_output(incosphilist, insinphilist, e1, e2, Rmask, klist, wlist, Nsrclist, srcm): # For each radial bin of each lens we calculate the output shears and weights
	
	# Calculating the needed errors
	wk2list = wlist*klist**2
	
	w_tot = np.sum(wlist, 1)
	w2_tot = np.sum(wlist**2, 1)

	k_tot = np.sum(klist, 1)
	k2_tot = np.sum(klist**2, 1)

	wk2_tot = np.sum(wk2list, 1)
	w2k4_tot = np.sum(wk2list**2, 1)
	w2k2_tot = np.sum(wlist**2 * klist**2, 1)
	wlist = []
	
	Nsrc_tot = np.sum(Nsrclist, 1)
	
	srcm, foo = np.meshgrid(srcm,np.zeros(len(klist)))
	foo = [] # Empty unused lists
	srcm_tot = np.sum(srcm*wk2list, 1) # the weighted sum of the bias m
	srcm = []

	gc.collect()

	# Calculating the weighted tangential and cross shear of the lens-source pairs
	gammatlists = np.zeros([4, len(incosphilist), len(incosphilist[0])])
	gammaxlists = np.zeros([4, len(incosphilist), len(incosphilist[0])])

	klist = np.ma.filled(np.ma.array(klist, mask = Rmask, fill_value = inf))
	
	for g in xrange(4):
		gammatlists[g] = np.array((-e1[:,g] * incosphilist - e2[:,g] * insinphilist) * wk2list / klist)
		gammaxlists[g] = np.array((e1[:,g] * insinphilist - e2[:,g] * incosphilist) * wk2list / klist)
		
	[gammat_tot_A, gammat_tot_B, gammat_tot_C, gammat_tot_D] = [np.sum(gammatlists[g], 1) for g in xrange(4)]
	[gammax_tot_A, gammax_tot_B, gammax_tot_C, gammax_tot_D] = [np.sum(gammaxlists[g], 1) for g in xrange(4)]
	
	return gammat_tot_A, gammax_tot_A, gammat_tot_B, gammax_tot_B, gammat_tot_C, gammax_tot_C, gammat_tot_D, gammax_tot_D, w_tot, w2_tot, k_tot, k2_tot, wk2_tot, w2k4_tot, w2k2_tot, Nsrc_tot, srcm_tot


def calc_covariance_output(incosphilist, insinphilist, klist): # For each radial bin of each lens we calculate the output shears and weights
	
	# For each radial bin of each lens we calculate the weighted sum of the tangential and cross shear
	Cs_tot = sum(-incosphilist*klist, 0)
	Ss_tot = sum(-insinphilist*klist, 0)
	Zs_tot = sum(klist**2, 0)
	
	return Cs_tot, Ss_tot, Zs_tot


def write_catalog(filename, galIDlist, Rbins, Rcenters, nRbins, output, outputnames, variance, purpose, e1, e2, w, srcm):

	fitscols = []

	Rmin = Rbins[0:nRbins]
	Rmax = Rbins[1:nRbins+1]
	
	# Adding the radial bins
	if 'bootstrap' in purpose:
		fitscols.append(pyfits.Column(name = 'Bootstrap', format='20A', array = galIDlist))
	else:
		fitscols.append(pyfits.Column(name = 'ID', format='J', array = galIDlist))
	
	fitscols.append(pyfits.Column(name = 'Rmin', format = '%iD'%nRbins, array = [Rmin]*len(galIDlist)))
	fitscols.append(pyfits.Column(name = 'Rmax', format='%iD'%nRbins, array = [Rmax]*len(galIDlist)))
	fitscols.append(pyfits.Column(name = 'Rcenter', format='%iD'%nRbins, array = [Rcenters]*len(galIDlist)))
	
	# Adding the output
	[fitscols.append(pyfits.Column(name = outputnames[c], format = '%iD'%nRbins, array = output[c])) for c in xrange(len(outputnames))]

	if purpose == 'covariance':
		fitscols.append(pyfits.Column(name = 'e1', format='4D', array= e1))
		fitscols.append(pyfits.Column(name = 'e2', format='4D', array= e2))
		fitscols.append(pyfits.Column(name = 'lfweight', format='1D', array= w))
		fitscols.append(pyfits.Column(name = 'bias_m', format='1D', array= srcm))
		
	# Adding the variance for the 4 blind catalogs
	fitscols.append(pyfits.Column(name = 'variance(e[A,B,C,D])', format='4D', array= [variance]*len(galIDlist)))
	
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

	return

	
def calc_stack(gammat, gammax, wk2, w2k2, srcm, variance, blindcatnum): # Calculating the final output values
	
	# Choosing the appropriate covariance value	
	variance = variance[blindcatnum]
	
	ESDt_tot = gammat / wk2 / (1e3)**2 # Final Excess Surface Density (tangential component)
	ESDx_tot = gammax / wk2 / (1e3)**2 # Final Excess Surface Density (cross component)
	error_tot = (w2k2 / wk2**2 * variance)**0.5 / (1e3)**2 # Final error
	bias_tot = (1 + (srcm / wk2)) # Final multiplicative bias (by which the signal is to be divided)
	
	return ESDt_tot, ESDx_tot, error_tot, bias_tot


def write_stack(filename, Rcenters, ESDt_tot, ESDx_tot, error_tot, bias_tot, h, variance, blindcatnum, galIDs_matched, galIDs_matched_infield): # Printing stacked shear profile to a file

	# Choosing the appropriate covariance value
	variance = variance[blindcatnum]

	with open(filename, 'w') as file:
		print >>file, '# Radius(kpc)	ESD_t(h%g*M_sun/pc^2)	ESD_x(h%g*M_sun/pc^2)	error(h%g*M_sun/pc^2)^2	bias(1+K)	variance(e_s)'%(h*100, h*100, h*100)

	with open(filename, 'a') as file:
		for R in xrange(len(Rcenters)):
			
			if not (0 < error_tot[R] and error_tot[R]<inf):
				ESDt_tot[R] = int(-999)
				ESDx_tot[R] = int(-999)
				error_tot[R] = int(-999)
				bias_tot[R] = int(-999)
			
			print >>file, '%.12g	%.12g	%.12g	%.12g	%.12g	%.12g'%(Rcenters[R], ESDt_tot[R], ESDx_tot[R], error_tot[R], bias_tot[R], variance)

	print 'Written: ESD profile data:', filename


	if len(galIDs_matched)>0 and blindcatnum == 0:
		# Writing galID's to another file
		galIDsname_split = filename.rsplit('_',1)
		galIDsname = '%s_lensIDs.txt'%galIDsname_split[0]
		kidsgalIDsname = '%s_KiDSlensIDs.txt'%galIDsname_split[0]
		
		with open(galIDsname, 'w') as file:
			print >>file, "# ID's of all stacked lenses:"

		with open(galIDsname, 'a') as file:
			for i in xrange(len(galIDs_matched)):
				print >>file, galIDs_matched[i]

			
		with open(kidsgalIDsname, 'w') as file:
			print >>file, "# ID's of stacked lenses within fields:"

		with open(kidsgalIDsname, 'a') as file:
			for i in xrange(len(galIDs_matched_infield)):
				print >>file, galIDs_matched_infield[i]

		print "Written: List of all stacked lens ID's that contribute to the signal:", galIDsname
		print "Written: List of stacked lens ID's with their center within a KiDS field:", kidsgalIDsname

	return

	
def define_plottitle(purpose, centering, rankmin, rankmax, Nfofmin, Nfofmax, obslim, obslim_min, obslim_max, binname, binrange, ZBmin, ZBmax):

	# Define the labels for the plot
	plottitle = r'%s: %g $\leq$ rank%s $\leq$ %g'%(purpose, rankmin, centering, rankmax)

	if all(np.array([rankmin, rankmax]) > 0):
		# The group member selection depends on the number of group members Nfof
		plottitle = ', %g $\leq$ Nfof $\leq$ %g'%(Nfofmin, Nfofmax)
		
	# If there are observable limits...
	if obslim != 'No':
		# The galaxy selection depends on observable
		plottitle = '%s, %g $\leq$ %s $\leq$ %g'%(plottitle, obslim_min[0], obslim, obslim_max[-1])

	plottitle = '%s, %g $\leq$ Z$_B$ $\leq$ %g'%(plottitle, ZBmin, ZBmax)

	return plottitle

def define_plot(filename, plotlabel, plottitle, plotstyle, Nsubplots, xlabel, ylabel, n): # Plotting the data
	
	from matplotlib import pyplot as plt
	from matplotlib.colors import LogNorm
	from matplotlib import gridspec
	from matplotlib import rc, rcParams
	
	# Make use of TeX
	rc('text',usetex=True)

	# Change all fonts to 'Computer Modern'
	rc('font',**{'family':'serif','serif':['Computer Modern']})

	title_size = 14
	if type(Nsubplots) == int:
		
		Nplot = n
		
		if Nsubplots < 4:
			Nrows = 1
		else:
			Nrows = 2
		
		Ncolumns = int(Nsubplots/Nrows)

		plt.figure(1, figsize=(4*Ncolumns+3,5*Nrows))

		Nsubplot = 100*Nrows+10*Ncolumns+Nplot
		plt.subplot(Nsubplot)

		if Nplot == int(Ncolumns/2+1):
			plt.title(plottitle, fontsize=title_size)

	else:
		
		# Plot and print ESD profile to a file
		plt.title(plottitle,fontsize=title_size)

	# Load the text file containing the stacked profile
	data = np.loadtxt(filename).T

	bias = data[4]
	bias[bias==-999] = 1

	data_x = data[0]
	data_y = data[1]/bias
	data_y[data_y==-999] = np.nan

	errorh = (data[3])/bias # covariance error
	errorl = (data[3])/bias # covariance error
	errorh[errorh==-999] = np.nan
	errorl[errorl==-999] = np.nan
	
	if type(Nsubplots) != int:
		data_x = data_x + n*0.1*data_x
	
	if 'lin' in plotstyle:
	
		plt.autoscale(enable=False, axis='both', tight=None)
		plt.xlim(1e1,1e4)
		
		if plotstyle == 'linR':
			linlabel = r'%s $\cdot$ %s'%(xlabel, ylabel)
			plt.errorbar(data_x, data_x*data_y, yerr=data_x*errorh, marker='o', ls='', label=plotlabel)

			plt.axis([1e1,1e4,-5e3,2e4])
			plt.ylim(-5e3,1e5)

		if plotstyle == 'lin':
			linlabel = r'%s'%(ylabel)
			plt.errorbar(data_x, data_y, yerr=errorh, marker='o', ls='', label=plotlabel)

			plt.axis([1e1,1e4,-20,100])
			plt.ylim(-20,100)
	
		if plotstyle == 'errorlin':
			linlabel = r'%s $\cdot$ Error(%s)'%(xlabel, ylabel)
			plt.plot(data_x, data_x*errorh, marker='o', label=plotlabel) # removed ls=''
			
			plt.axis([1e1,1e4,-5e3,2e4])
			plt.ylim(-5e3,1e5)
			
		plt.ylabel(r'%s'%linlabel,fontsize=15)

		
	if 'log' in plotstyle:
		plt.yscale('log')
		errorl[errorl>=data_y] = ((data_y[errorl>=data_y])*0.9999999999)

		plt.autoscale(enable=False, axis='both', tight=None)
		plt.axis([1e1,1e4,1e-1,1e4])
		plt.ylim(1e-1,1e4)
		
		if plotstyle == 'log':		
			plt.errorbar(data_x, data_y, yerr=[errorl,errorh], ls='', marker='o', label=plotlabel)
			plt.ylabel(r'%s'%ylabel,fontsize=15)
		
		if plotstyle == 'errorlog':
			plt.plot(data_x, errorh, marker='o', label=plotlabel) # removed ls=''
			plt.ylabel(r'Error(%s)'%ylabel,fontsize=15)

	plt.xscale('log')
	plt.xlabel(r'%s'%xlabel,fontsize=15)
	
	plt.legend(loc='upper right',ncol=1, prop={'size':12})

	return
	
	
def write_plot(plotname, plotstyle): # Writing and showing the plot

	from matplotlib import pyplot as plt
	from matplotlib.colors import LogNorm
	from matplotlib import gridspec
	from matplotlib import rc, rcParams

#	# Make use of TeX
	rc('text',usetex=True)

	# Change all fonts to 'Computer Modern'
	rc('font',**{'family':'serif','serif':['Computer Modern']})

	file_ext = plotname.split('.')[-1]
	plotname = plotname.replace('.%s'%file_ext,'_%s.png'%plotstyle)

	plt.savefig(plotname, format='png')
	print 'Written: ESD profile plot:', plotname
#	plt.show()
	plt.close()
	
	return
	

def plot_covariance_matrix(filename, plottitle1, plottitle2, plotstyle, binname, binrange, Rbins, h):
	
	from matplotlib import pyplot as plt
	from matplotlib.colors import LogNorm
	from matplotlib import gridspec
	from matplotlib import rc, rcParams
	
	# Make use of TeX
	rc('text',usetex=True)

	# Change all fonts to 'Computer Modern'
	rc('font',**{'family':'serif','serif':['Computer Modern']})

	# Number of observable bins
	Nobsbins = len(binrange)-1

	# Number and values of radial bins
	nRbins = len(Rbins)-1

	# Plotting the ueber matrix
	fig = plt.figure(figsize=(12,10))

	gs_full = gridspec.GridSpec(1,2, width_ratios=[20,1], wspace=0.1)
	gs = gridspec.GridSpecFromSubplotSpec(Nobsbins, Nobsbins, wspace=0, hspace=0, subplot_spec=gs_full[0,0])
	gs_bar = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[1,3,1], hspace=0, subplot_spec=gs_full[0,1])
	cax = fig.add_subplot(gs_bar[1,0])

	ax = fig.add_subplot(gs_full[0,0])

	data = np.loadtxt(filename).T
	
	covariance = data[4].reshape(Nobsbins,Nobsbins,nRbins,nRbins)
	correlation = data[5].reshape(Nobsbins,Nobsbins,nRbins,nRbins)
	bias = data[6].reshape(Nobsbins,Nobsbins,nRbins,nRbins)
	
#	covariance = covariance/bias
#	correlation = covariance/correlation


	for N1 in xrange(Nobsbins):
		for N2 in xrange(Nobsbins):
			
			# Add subplots
			ax_sub = fig.add_subplot(gs[Nobsbins-N1-1,N2])
			
#			print N1+1, N2+1, N1*Nobsbins+N2+1
			
			ax_sub.set_xscale('log')
			ax_sub.set_yscale('log')
			
			if plotstyle == 'covlin':
				mappable = ax_sub.pcolormesh(Rbins, Rbins, covariance[N1,N2,:,:], vmin=-1e7, vmax=1e7)
			if plotstyle == 'covlog':
				mappable = ax_sub.pcolormesh(Rbins, Rbins, abs(covariance[N1,N2,:,:]), norm=LogNorm(vmin=1e-7, vmax=1e7))
			if plotstyle == 'corlin':
				mappable = ax_sub.pcolormesh(Rbins, Rbins, correlation[N1,N2,:,:], vmin=-1, vmax=1)
			if plotstyle == 'corlog':
				mappable = ax_sub.pcolormesh(Rbins, Rbins, abs(correlation)[N1,N2,:,:], norm=LogNorm(vmin=1e-5, vmax=1e0))

			ax_sub.set_xlim(Rbins[0],Rbins[-1])
			ax_sub.set_ylim(Rbins[0],Rbins[-1])
			

			if N1 != 0:
				ax_sub.tick_params(axis='x', labelbottom='off')
			if N2 != 0:
				ax_sub.tick_params(axis='y', labelleft='off')
			
			if binname != 'No':
				if N1 == Nobsbins - 1:
					ax_sub.xaxis.set_label_position('top')
					if N2 == 0:
						ax_sub.set_xlabel(r'%s = %.3g - %.3g'%(binname, binrange[N2], binrange[N2+1]), fontsize=17)
					else:
						ax_sub.set_xlabel(r'%.3g - %.3g'%(binrange[N2], binrange[N2+1]), fontsize=17)
					
				if N2 == Nobsbins - 1:
					ax_sub.yaxis.set_label_position('right')
					if N1 == 0:
						ax_sub.set_ylabel(r'%s = %.3g - %.3g'%(binname, binrange[N1], binrange[N1+1]), fontsize=17)
					else:
						ax_sub.set_ylabel(r'%.3g - %.3g'%(binrange[N1], binrange[N1+1]), fontsize=17)
				
#			ax_sub.set_xticks([1e2,1e3])
#			ax_sub.set_yticks([1e2,1e3])


	# Turn off axis lines and ticks of the big subplot
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off', left='off', right='off')

	ax.set_xlabel(r'Radial distance (kpc/h$_{%g}$)'%(h*100))
	ax.set_ylabel(r'Radial distance (kpc/h$_{%g}$)'%(h*100))

	ax.xaxis.set_label_coords(0.5, -0.05)
	ax.yaxis.set_label_coords(-0.05, 0.5)

	ax.xaxis.label.set_size(17)
	ax.yaxis.label.set_size(17)

	plt.text(0.5, 1.08, plottitle1, horizontalalignment='center', fontsize=17, transform = ax.transAxes)
	plt.text(0.5, 1.05, plottitle2, horizontalalignment='center', fontsize=17, transform = ax.transAxes)
	plt.colorbar(mappable, cax=cax, orientation='vertical')

	file_ext = filename.split('.')[-1]
	plotname = filename.replace('.%s'%file_ext,'_%s.png'%plotstyle)

	plt.savefig(plotname,format='png')

	print 'Written: Covariance matrix plot:', plotname
#	plt.show()
