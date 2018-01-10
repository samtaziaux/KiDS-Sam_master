#!/usr/bin/python
"""
# This contains all the modules that are needed to
# calculate the shear profile catalog and the covariance.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import gc
import numpy as np
import sys
import os
import fnmatch
import shlex
import subprocess as sub
import time
from glob import glob

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
from astropy.io import ascii, fits as pyfits
from astropy.table import Table

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

from . import distance, esd_utils

# Important constants(very preliminary!)
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
pix = 0.187 # Used to translate pixel to arcsec
alpha = 0.057 # Used to calculate m
beta = -0.37 # Used to calculate m
inf = np.inf


def input_variables(Nsplit, Nsplits, binnum, blindcat, config_file):
    
    ## Input for the codes
    #try:
        #Nsplit = int(sys.argv[1])-1 # The number of this particular core/split
        #Nsplits = int(sys.argv[2]) # The number cores/splits
        #binnum = int(sys.argv[3]) # The number of this particular observable bin
        #blindcat = str(sys.argv[4]) # The number of this blind KiDS catalog
        #config_file = str(sys.argv[5]) # The path to the configuration file
    #except:
        #Nsplit = 1 # The number of this particular core/split
        #Nsplits = 1 # The number cores/splits
        #binnum = 1 # The number of this particular observable bin
        #blindcat = 'D' # The number of this particular blind KiDS catalog
        #config_file = str(sys.argv[1]) # The path to the configuration file

        #print 'Warning: Input not found!'

    # Importing the input parameters from the config file
    path_kidscats, path_gamacat, colnames, specz_file, O_matter, O_lambda, Ok, h, \
        z_epsilon, path_output, filename_addition, purpose, \
        path_Rbins, Runit, Ncores, lensid_file, lens_weights, lens_binning, \
        lens_selection, src_selection, cat_version, wizz, n_boot, cross_cov, \
        blindcats = \
            esd_utils.read_config(config_file)

    print()
    print('Running:', purpose)

    # Defining the number of the blind KiDS catalogue
    if blindcat[0] == 'A':
        blindcatnum = 0
    if blindcat[0] == 'B':
        blindcatnum = 1
    if blindcat[0] == 'C':
        blindcatnum = 2
    if blindcat[0] == 'D':
        blindcatnum = 3


    # Defining the addition to the file name
    if filename_addition == 'None':
        filename_addition = ''
    else:
        filename_addition = '_{0}'.format(filename_addition)

    # Binnning information of the lenses
    obsbins = define_obsbins(1, lens_binning, [], [])
    binname, lens_binning, Nobsbins, binmin, binmax = obsbins

    # Defining the lens-ID lens selection/binning
    if 'None' not in lensid_file:
        selection = define_lensid_selection(
            lensid_file, lens_selection, lens_binning, binname, Nobsbins)
        lens_selection, lens_binning, binname, Nobsbins = selection

    # Defining the center definition
    centers = np.array(['Cen', 'IterCen', 'BCG'])
    centering = 'None'
    for cen in centers:
        if ('rank%s'%cen in binname) or \
            ('rank%s'%cen in lens_selection.keys()):
            centering = cen
            print('Center definition = %s'%centering)
    if centering == 'Cen':
        lens_selection['rank%s'%centering] = ['self', np.array([1])]
        msg = 'WARNING: With the Cen definition,'
        msg += ' you can only use Centrals (Rank = 1)'
        print(msg)
    
    # Name of the Rbins
    if os.path.isfile(path_Rbins): # from a file
        name_Rbins = path_Rbins.split('.')[0]
        name_Rbins = name_Rbins.split('/')[-1]
        name_Rbins = 'Rbins~%s_%s'%(name_Rbins, Runit)
    else:
        name_Rbins = path_Rbins.replace(',', '_')
        name_Rbins = 'Rbins%s_%s'%(name_Rbins, Runit)

    # Creating all necessary folders

    # Path containing the output folders
    output_var = ''
    var_print = ''
    #output_var, var_print, x = define_filename_sel(output_var, var_print,\
    #                                                 '', src_selection)
    if ('ID' in lens_binning) & ('No' in binname):# or ('ID' in lens_selection)
        output_var = 'IDs_from_file'
        path_output = '%s/%s%s' \
            %(path_output, output_var, filename_addition)
    elif ('ID' not in lens_binning) & ('No' in binname):# or ('ID' not in lens_selection)
        output_var = 'No_bins'
        path_output = '%s/%s%s' \
            %(path_output, output_var, filename_addition)
    else:
        output_var = lens_binning[binname][1]
        output_var = '_'.join(map(str, output_var))
        output_var = output_var.replace('.', 'p').replace('-','m')
        path_output = '%s/%s_%s%s' \
            %(path_output, binname, output_var, filename_addition)

    path_catalogs = '%s/catalogs' %(path_output.rsplit('/',1)[0])

    # Path to the output splits and results
    path_splits = '%s/splits_%s' %(path_output, purpose) #'%s/splits_%s' %(path_output, purpose)
    path_results = path_output #'%s/results_%s' %(path_output, purpose)

    if (Nsplit == 0) and (blindcat == blindcats[0]) and (binnum == Nobsbins):

        for path in [path_output, path_catalogs, path_splits, path_results]:
            if not os.path.isdir(path):
                os.makedirs(path)
                print('Creating new folder:', path)
        print()

    if 'catalog' in purpose:

        # Path to the output splits and results
        path_splits = '%s/splits_%s'%(path_catalogs, purpose)
        path_results = '%s/results_%s'%(path_catalogs, purpose)

        if (Nsplit==0) and (blindcat==blindcats[0]) and (binnum == Nobsbins):

            for path in [path_splits, path_results]:
                if not os.path.isdir(path):
                    os.makedirs(path)
		    


    # Determining Ncat, the number of existing random catalogs
    splitslist = [] # This list will contain all created random splits
    
    if ('random' in purpose):

        # Defining the name of the output files
        filename_var = define_filename_var(purpose.replace('bootstrap', \
                                                           'catalog'), \
                                           centering, binname, binnum, \
                                           Nobsbins, lens_selection, lens_binning, \
                                           src_selection, lens_weights, \
                                           name_Rbins, O_matter, \
                                           O_lambda, Ok, h)
        path_randomsplits = '%s/splits_%s'%(path_catalogs, purpose)
        
        for Ncat in xrange(100):
            outname = '%s/%s_%i_%s%s_split%iof*.fits'\
                    %(path_randomsplits.replace('bootstrap', 'catalog'), \
                      purpose.replace('bootstrap', 'catalog'), Ncat+1, \
                      filename_var, filename_addition, Nsplit+1)
            placeholder = outname.replace('*', '0')
            if os.path.isfile(placeholder):
                os.remove(placeholder)

            splitfiles = glob(outname)
            splitslist = np.append(splitslist, splitfiles)
            
            if len(splitfiles) == 0:
                break

        print(outname)
        

    else:
        Ncat = 1
    
    return Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
        lens_selection, lens_weights, binname, Nobsbins, src_selection, \
        cat_version, wizz, path_Rbins, name_Rbins, Runit, path_output, \
        path_splits, path_results, purpose, O_matter, O_lambda, Ok, h, \
        filename_addition, Ncat, splitslist, blindcats, blindcat, \
        blindcatnum, path_kidscats, path_gamacat, colnames, specz_file, \
        z_epsilon, n_boot, cross_cov


# Defining the lensID lens selection/binning
def define_lensid_selection(lensid_file, lens_selection, lens_binning, binname, Nobsbins):
    
    IDname = 'ID'
    
    lensid_files = lensid_file.split(',')

    if len(lensid_files) == 1: # If there is only one lensID bin -> selection
        lensids = np.loadtxt(lensid_files[0], dtype=np.int64)
        lens_selection[IDname] = ['self', lensids]
    else: # If there are multiple lensID bins -> binning
        binname = IDname
        lens_binning = dict()
        Nobsbins = len(lensid_files)
        for i, f in enumerate(lensid_files):
            lensids = np.loadtxt(f, dtype=np.int64)
            lens_binning['%sbin%i' %(binname, i+1)] = ['self', lensids]

    return lens_selection, lens_binning, binname, Nobsbins


# Define the part of the filename and plottitle
# that contains the lens/source selections
def define_filename_sel(filename_var, var_print, plottitle, selection):
    
    selnames = np.sort(selection.keys())
    for selname in selnames:
        sellims = (selection[selname])[1]
        selname = selname.replace('_','')

        if 'ID' in selname:
            filename_var = '%s~%ss_%g'%(filename_var, selname, len(sellims))
            var_print = '%s #%ss = %g,'%(var_print, selname, len(sellims))
            plottitle = '%s $\#$ %ss = %g,'%(plottitle, selname, len(sellims))
        else:
            if len(sellims) == 1:
                filename_var = '%s~%s_%g'%(filename_var, selname, sellims[0])
                var_print = '%s %s = %g,'%(var_print, selname, sellims[0])
                plottitle = '%s %s = %g,'%(plottitle, selname, sellims[0])
            else:
                filename_var = '%s~%s_%g_%g'%(filename_var, selname, \
                                             sellims[0], sellims[1])
                var_print = '%s %s-limit: %g - %g,'%(var_print, selname, \
                                                     sellims[0], sellims[1])
                plottitle = '%s %g $\leq$ %s $<$ %g,'%(plottitle, \
                                                          sellims[0], selname, \
                                                          sellims[1])
                                                          
                #plottitle = '$\mathrm{{{0}\,{1:g}}} \leq'.format(
                #                plottitle, sellims[0])
                #plottitle = '{0} \mathrm{{{1} \leq {2:g},}}'.format(
                #                plottitle, selname, sellims[1])
        
    return filename_var, var_print, plottitle


def define_filename_sel_bin(filename_var, var_print, plottitle, selection, binnum, Nobsbins):
    print(binnum)
    if type(binnum) == int:
        binnum = binnum-1
    elif type(binnum) == str:
        binnum = Nobsbins-1
    print(binnum)
    print()

    selnames = np.sort(selection.keys())
    for selname in selnames:
        sellims = (selection[selname])[1]
        
        if 'ID' in selname:
            pass
        else:
            filename_var = '%s_%.2f~%.2f'%(filename_var, \
                                             sellims[binnum], sellims[binnum+1])
            var_print = '%s %s-limit: %g - %g,'%(var_print, selname, \
                                                sellims[binnum], sellims[binnum+1])
            plottitle = '%s %g $\leq$ %s $<$ %g,'%(plottitle, \
                                                sellims[binnum], selname, \
                                                sellims[binnum+1])

    return filename_var, var_print, plottitle

    
# Defining the part of the filename that contains the chosen variables
def define_filename_var(purpose, centering, binname, binnum, Nobsbins, \
                        lens_selection, lens_binning, src_selection, lens_weights, \
                        name_Rbins, O_matter, O_lambda, Ok, h):
    
    # Define the list of variables for the output filename

    filename_var = ''
    var_print = ''

    if 'catalog' in purpose:
        if centering == 'Cen':
            filename_var = 'Cen'

        filename_var_bins = '_'
        filename_var_lens = '~'

    else: # Binnning information of the groups

        # Lens binning
        if 'No' not in binname: # If there is binning
            if 'ID' in binname:
                binname = 'ID'
            filename_var_bins = '%s_%s_bin_%s'%(filename_var, purpose, \
                                             binnum)
            var_print = '%s %i %s-bins,'%(var_print, Nobsbins, binname)
        else:
            filename_var_bins = '%s_No_bins'%(filename_var)
            var_print = '%s No bins,'%(var_print)
        # Lens selection
        filename_var_lens, var_print, x = define_filename_sel(filename_var, \
                                                         var_print, '', \
                                                         lens_selection)
    
        weightname = lens_weights.keys()[0]
        if weightname != 'None':
            filename_var_lens = '%s_lw~%s'%(filename_var_lens, weightname)
            var_print = '%s Lens weights: %s,'%(var_print, weightname)
    
    # Source selection
    filename_var_source, var_print, x = define_filename_sel(filename_var, var_print,\
                                                     '', src_selection)
    
    filename_var_cosmo = '%s_Om_%g~Ol_%g~Ok_%g~h_%g'%(filename_var, \
                                               O_matter, O_lambda, Ok, h)
    filename_var_radial = '%s_%s'%(filename_var, name_Rbins)
    cosmo_print = ('    %s, Omatter=%g, Olambda=%g, Ok=%g, h=%g'%(name_Rbins, \
                                                    O_matter, \
                                                    O_lambda, Ok, \
                                                    h)).replace('~', '-')

    filename_var_bins = filename_var_bins.split('_', 1)[1]
    filename_var_lens = filename_var_lens.split('~', 1)[1]
    filename_var_cosmo = filename_var_cosmo.split('_', 1)[1]
    filename_var_radial = filename_var_radial.split('_', 1)[1]
    filename_var_source = filename_var_source.split('~', 1)[1]

    if 'catalog' in purpose:
        filename_var = '%s~%s/%s/%s'%(filename_var_source,\
                                    filename_var_cosmo,filename_var_radial,\
                                    purpose)
    else:
        filename_var = '%s/%s~%s/%s/%s/%s'%(filename_var_lens,filename_var_source,\
                                    filename_var_cosmo,filename_var_radial,\
                                    purpose,filename_var_bins)

    filename_var = filename_var.replace('.', 'p')
    filename_var = filename_var.replace('-', 'm')
    filename_var = filename_var.replace('~', '-')
    
    if 'covariance' not in purpose:
        print('Chosen %s-configuration: '%purpose)
        print(var_print)
        print(cosmo_print)
        print()
    
    return filename_var


def define_filename_splits(path_splits, purpose, filename_var, \
                           Nsplit, Nsplits, filename_addition, blindcat):
    
    # Defining the names of the shear/random catalog
    if 'covariance' in purpose:
        splitname = '%s/%s_%s.fits'%(path_splits, filename_var, Nsplit)
                                        # Here Nsplit = kidscatname
    if 'bootstrap' in purpose:
        splitname = '%s/%s_%s_%s.fits'%(path_splits, purpose, filename_var, blindcat)

    if 'catalog' in purpose:
        splitname = '%s/%s_%s_split%iof%i.fits'%(path_splits, purpose, \
                                                   filename_var, \
                                                   Nsplit, Nsplits)
    new_path = '/'.join(splitname.split('/')[:-1])
    #if not os.path.isdir(new_path):
        #os.makedirs(new_path)
    while True:
        #print True
        if not os.path.isdir(new_path):
            try:
                os.makedirs(new_path)
                break
            except OSError as e:
                if e.errno != 17:
                    raise
                time.sleep(2)
                pass
        else:
            break


    return splitname


def define_filename_results(path_results, purpose, filename_var, \
                            filename_addition, Nsplit, blindcat):
    # Paths to the resulting files
    if 'catalogs' in path_results:
        resultname = '%s/%s_%s%s.fits'%(path_results, purpose, \
                                        filename_var, filename_addition)
    else:
        #filename_var = filename_var.partition('purpose')[0]
        #resultname = '%s/%s_%s%s_%s.txt'%(path_results, purpose, filename_var, \
        #                                           filename_addition, blindcat)
        if 'catalog' in purpose:
            filename_var = filename_var.replace('shear','shearcatalog')
    
        resultname = '%s/%s_%s.txt'%(path_results, filename_var, blindcat)
    
    new_path = '/'.join(resultname.split('/')[:-1])
    #if not os.path.isdir(new_path):
        #os.makedirs(new_path)
    while True:
        #print True
        if not os.path.isdir(new_path):
            try:
                os.makedirs(new_path)
                break
            except OSError as e:
                if e.errno != 17:
                    raise
                time.sleep(2)
                pass
        else:
            break

    return resultname


# Importing all GAMA and KiDS data, and
# information on radial bins and lens-field matching.
def import_data(path_Rbins, Runit, path_gamacat, colnames, path_kidscats,
                centering, purpose, Ncat, O_matter, O_lambda, Ok, h,
                lens_weights, filename_addition, cat_version):

    # Import R-range
    Rmin, Rmax, Rbins, Rcenters, \
        nRbins, Rconst = define_Rbins(path_Rbins, Runit)

    # Import GAMA catalogue
    gamacat, galIDlist, galRAlist, galDEClist, galweightlist, galZlist, \
        Dcllist, Dallist = import_gamacat(
            path_gamacat, colnames, centering, purpose, Ncat, O_matter,
            O_lambda, Ok, h, Runit, lens_weights)

    # Determine the coordinates of the KiDS catalogues
    kidscoord, kidscat_end = run_kidscoord(path_kidscats, cat_version)

    # Match the KiDS field and GAMA galaxy coordinates
    catmatch, kidscats, galIDs_infield = run_catmatch(
        kidscoord, galIDlist, galRAlist, galDEClist, Dallist, Rmax, purpose,
        filename_addition, cat_version)

    gc.collect()

    return catmatch, kidscats, galIDs_infield, kidscat_end, Rmin, Rmax, Rbins, \
        Rcenters, nRbins, Rconst, gamacat, galIDlist, galRAlist, \
        galDEClist, galweightlist, galZlist, Dcllist, Dallist


# Define the radial bins around the lenses
def define_Rbins(path_Rbins, Runit):

    if os.path.isfile(path_Rbins): # from a file

        # Start, End, number of steps and step size of the radius R
        Rrangefile = np.loadtxt(path_Rbins).T
        Rmin = Rrangefile[0,0]
        Rmax = Rrangefile[-1,-1]
        Rbins = np.append(Rrangefile[0],Rmax)
        Rcenters = Rrangefile[1]
        nRbins = len(Rcenters)
    
        print('path_Rbins', path_Rbins)
        print('Using: %i radial bins between %.1f and %.1f'%(nRbins, Rmin, Rmax))
        print('Rmin', Rmin)
        print('Rmax', Rmax)
        print('Rbins', Rbins)
        print('Rcenters', Rcenters)
        print('nRbins', nRbins)

    else: # from a specified number (of bins)
        try:
            # Start, End, number of steps and step
            # size of the radius R (logarithmic 10^x)
            binlist = path_Rbins.split(',')

            nRbins = int(binlist[0])
            Rmin = float(binlist[1])
            Rmax = float(binlist[2])
            Rstep = (np.log10(Rmax)-np.log10(Rmin))/(nRbins)
            Rbins = 10.**np.arange(np.log10(Rmin), np.log10(Rmax), Rstep)
            Rbins = np.append(Rbins,Rmax)
            Rcenters = np.array([(Rbins[r]+Rbins[r+1])/2 \
                                 for r in xrange(nRbins)])

        except:
            print('Observable bin file does not exist:', path_Rbins)
            raise SystemExit()
    
    # Translating from k/Mpc to pc, or from arcmin/sec to deg

    Rconst = -999
    if 'pc' in Runit:
        Rconst = 1.
        if 'k' in Runit:
            Rconst = 1e3
        if 'M' in Runit:
            Rconst = 1e6
    
    if 'arc' in Runit:
        if 'sec' in Runit:
            Rconst = 1/(60.**2)
        if 'min' in Runit:
            Rconst = 1/60.
    
    if Rconst == -999:
        print('*** Unit of radial bins not recognized! ***')
        raise SystemExit()
        
    [Rmin, Rmax, Rbins] = [r*Rconst for r in [Rmin, Rmax, Rbins]]

    
    return Rmin, Rmax, Rbins, Rcenters, nRbins, Rconst


# Load the properties (RA, DEC, Z -> dist) of the galaxies in the GAMA catalogue
def import_gamacat(path_gamacat, colnames, centering, purpose, Ncat,
                   O_matter, O_lambda, Ok, h, Runit, lens_weights):

    randomcatname = 'RandomsWindowedV01.fits'
    directory = os.path.dirname(os.path.realpath(path_gamacat))
    randomcatname = os.path.join(directory, randomcatname)

    # Importing the GAMA catalogues
    print('Importing GAMA catalogue:', path_gamacat)
    try:
        gamacat = Table(pyfits.open(
            path_gamacat, ignore_missing_end=True)[1].data)
    except IOError:
        gamacat = ascii.read(path_gamacat)

    # skip the ID column name since if it doesn't exist we create it below
    for colname in colnames[1:]:
        assert colname in gamacat.colnames, \
            'Full list of column names:\n{2}\n\n' \
            'Column {0} not present in catalog {1}. See the full list of' \
            'column names above'.format(
                colname, path_gamacat, gamacat.colnames)

    # IDs of all galaxies
    if colnames[0] not in gamacat.colnames:
        gamacat[colnames[0]] = np.arange(gamacat[colnames[1]].size, dtype=int)
    gamacat[colnames[0]] = np.array(gamacat[colnames[0]], dtype=str)
    galIDlist = gamacat[colnames[0]]
    if galIDlist.size != np.unique(galIDlist).size:
        print('Dear user, you have confused me with non unique IDs for your lenses.')
        print('I will refrain to keep running (it takes me a lot of energy)')
        print('till you make sure that each lens has its own ID.')
        print('I am not a communist code. Sorry for the inconvenience.')
        raise SystemExit()

    # these are very GAMA-specific
    if centering == 'Cen':
        galRAlist = gamacat['CenRA'] # Central RA of the galaxy (in degrees)
        galDEClist = gamacat['CenDEC'] # Central DEC of the galaxy (in degrees)
        galZlist = gamacat['Zfof'] # Z of the group
    else:
        galRAlist = gamacat[colnames[1]] # Central RA of the galaxy (in degrees)
        galDEClist = gamacat[colnames[2]] # Central DEC of the galaxy (in degrees)

    if 'random' in purpose:
        # Determine RA and DEC for the random/star catalogs
        # The first item that will be chosen from the catalog
        Ncatmin = Ncat # * len(galIDlist)
        # The last item that will be chosen from the catalog
        #Ncatmax = (Ncat+1) * len(galIDlist)
        try:
            randomcat = pyfits.open(randomcatname)[1].data
        except:
            print('Could not import random catalogue: ', randomcatname)
            print('Make sure that the random catalogue is next to the GAMA catalogue!')
            raise SystemExit()

        galIDlist_random = randomcat['CATAID']
        slice = np.in1d(galIDlist_random, galIDlist)
        step = 792 #len(galIDlist_random[slice])/len(galIDlist)
        # This is hardcoded for this set of randoms.
        Ncatmax = step*len(galIDlist)

        galRAlist = randomcat[colnames[1]][slice][Ncatmin:Ncatmax:step]
        galDEClist = randomcat[colnames[2]][slice][Ncatmin:Ncatmax:step]
        
    #Defining the lens weights
    weightname = lens_weights.keys()[0]
    if 'No' not in weightname:
        galweightlist = pyfits.open(lens_weights.values()[0])[1].data[weightname]
    else:
        galweightlist = np.ones(len(galIDlist))

    # Defining the comoving and angular distance to the galaxy center
    if 'pc' in Runit: # Rbins in a multiple of pc
        assert len(colnames) == 4, \
            'Please provide the name of the redshift column if you want' \
            ' to use physical projected distances'
        galZlist = gamacat[colnames[3]] # Central Z of the galaxy
        if 'random' in purpose:
            galZlist = randomcat[colnames[3]][slice][Ncatmin:Ncatmax:step]
        #Dcllist = np.array([distance.comoving(z, O_matter, O_lambda, h) \
        #                    for z in galZlist])
        # Distance in pc/h, where h is the dimensionless Hubble constant
        #Dcllist = np.array([distance.comoving(z, O_matter, O_lambda, h) \
        #                            for z in galZlist])

        # New method
        cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)
        Dcllist = np.array((cosmo.comoving_distance(galZlist).to('pc')).value)

    else: # Rbins in a multiple of degrees
        galZlist = np.zeros(len(galIDlist)) # No redshift
        # Distance in degree on the sky
        Dcllist = np.degrees(np.ones(len(galIDlist)))

    # The angular diameter distance to the galaxy center
    Dallist = Dcllist/(1+galZlist)

    return gamacat, galIDlist, galRAlist, galDEClist, \
        galweightlist, galZlist, Dcllist, Dallist


def run_kidscoord(path_kidscats, cat_version):
    # Finding the central coordinates of the KiDS fields
    if cat_version == 0:
        return run_kidscoord_mocks(path_kidscats, cat_version)

    # Load the names of all KiDS catalogues from the specified folder
    kidscatlist = os.listdir(path_kidscats)


    if cat_version == 2:
        # Remove all files from the list that are not KiDS catalogues
        for x in kidscatlist:
            if 'KIDS_' not in x:
                kidscatlist.remove(x)


        # Create the dictionary that will hold the names
        # of the KiDS catalogues with their RA and DEC
        kidscoord = dict()
        kidscat_end = ''
        for i in xrange(len(kidscatlist)):
            # Of the KiDS file names, keep only "KIDS_RA_DEC"

            kidscatstring = kidscatlist[i].split('_',3)
            kidscatname = '_'.join(kidscatstring[0:3])

            # Extract the central coordinates of the field from the file name
            coords = '_'.join(kidscatstring[1:3])
            coords = ((coords.replace('p','.')).replace('m','-')).split('_')

            # Fill the dictionary with the catalog's central RA
            # and DEC: {"KIDS_RA_DEC": [RA, DEC]}
            kidscoord[kidscatname] = [float(coords[0]),float(coords[1]), 0]

            kidscat_end = kidscatstring[-1]


    if cat_version == 3:
        kidscoord = dict()

        for x in kidscatlist:
            # Full directory & name of the corresponding KiDS catalogue
            kidscatfile = '%s/%s'%(path_kidscats, x)
            kidscat = pyfits.open(kidscatfile, memmap=True)[2].data
            #print kidscat['THELI_NAME']
            
            kidscatlist2 = np.unique(np.array(kidscat['THELI_NAME']))
            #kidscatname = np.full(kidscatlist2.shape, x, dtype=np.str)
            #print x
            #print kidscatname

            for i in xrange(len(kidscatlist2)):
                # Of the KiDS file names, keep only "KIDS_RA_DEC"
    
                kidscatstring = kidscatlist2[i].split('_',3)
                kidscatname2 = '_'.join(kidscatstring[0:3])
            
                # Extract the central coordinates of the field from the file name
                coords = '_'.join(kidscatstring[1:3])
                coords = ((coords.replace('p','.')).replace('m','-')).split('_')
            
                # Fill the dictionary with the catalog's central RA
                # and DEC: {"KIDS_RA_DEC": [RA, DEC]}
                kidscoord[x+'-'+str(i)] = [float(coords[0]), float(coords[1]), \
                                           kidscatlist2[i]]
            
                kidscat_end = ''

    gc.collect()
    return kidscoord, kidscat_end


def run_kidscoord_mocks(path_kidscats, cat_version):
    if cat_version == 0:
        kidscoord = dict()
        tile = np.arange(100)
        for i in xrange(10):
            for j in xrange(10):
                kidscoord[path_kidscats.split('/', -1)[-1]+'-'+str(tile[i*10+j])] = [i+0.5+150.0, j+0.5, path_kidscats.split('/', -1)[-1]+'-'+str(tile[i*10+j])]
    
        #kidscoord['mock'] = [5.0, 5.0, 1.0]
                                           
        kidscat_end = ''
    
    gc.collect()
    return kidscoord, kidscat_end


# Create a dictionary of KiDS fields that contain the corresponding galaxies.
def run_catmatch(kidscoord, galIDlist, galRAlist, galDEClist, Dallist, Rmax, \
                 purpose, filename_addition, cat_version):

    Rfield = np.radians(np.sqrt(2.0)/2.0) * Dallist
    if 'oldcatmatch' in filename_addition:
        print("*** Using old lens-field matching procedure! ***")
    else:
        Rmax = Rmax + Rfield
        #print "*** Using new lens-field matching procedure ***"
        #print "(for 'early science' mode, select"\
        #        " 'oldcatmatch' in 'ESD_output_filename')"

    totgalIDs = np.array([])

    catmatch = dict()
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
        name = kidscoord[kidscat][2]

        # Add the proper lenses to the list with all matched lenses
        totgalIDs = np.append(totgalIDs, galIDs)

        # If there are matched lenses in this field,
        # add it to the catmatch dictionary
        
        # Creating a dictionary that contains the corresponding
        # Gama galaxies for each KiDS field.
        if len(galIDs)>0:

            catmatch[kidscat] = np.array([])
            catmatch[kidscat] = np.append(catmatch[kidscat], [galIDs, name], 0)


    kidscats = catmatch.keys() # The list of fields with lens centers in them
    
    galIDs_infield = totgalIDs # The galaxies that have their centers in a field


    # Adding the lenses outside the fields to the dictionary
    for kidscat in kidscoord.keys():

        # The RA and DEC of the KiDS catalogs
        catRA = kidscoord[kidscat][0]
        catDEC = kidscoord[kidscat][1]

        # Defining the distance R between the lens center
        # and its surrounding background sources
        catR = Dallist*np.arccos(np.cos(np.radians(galDEClist))*\
                                 np.cos(np.radians(catDEC))*\
                                 np.cos(np.radians(galRAlist-catRA))+\
                                 np.sin(np.radians(galDEClist))*\
                                 np.sin(np.radians(catDEC)))

        coordmask = (catR < Rmax)

        galIDs = np.array(galIDlist[coordmask])
        name = kidscoord[kidscat][2]


        if 'bootstrap' in purpose:
            lensmask = np.logical_not(np.in1d(galIDs, totgalIDs))
            galIDs = galIDs[lensmask]
        else:
            if kidscat in kidscats:
                lensmask = np.logical_not(np.in1d(galIDs, catmatch[kidscat][0]))
                galIDs = galIDs[lensmask]

        totgalIDs = np.append(totgalIDs, galIDs)

        if len(galIDs)>0:
            if kidscat not in kidscats:
                catmatch[kidscat] = []

            catmatch[kidscat] = np.append(catmatch[kidscat], [galIDs, name], 0)
                    
    kidscats = catmatch.keys()
    

    print('Matched fields:', len(kidscats), ', Matched field-galaxy pairs:', \
        len(totgalIDs), ', Matched galaxies:', len(np.unique(totgalIDs)),\
        ', Percentage(Matched galaxies):',  \
        float(len(np.unique(totgalIDs)))/float(len(galIDlist))*100, '%')
    print()
    
    return catmatch, kidscats, galIDs_infield



def split(seq, size): # Split up the list of KiDS fields for parallel processing

    newseq = []
    splitsize = len(seq)/size

    for i in xrange(size-1):
        newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    newseq.append(seq[int(round((size-1)*splitsize)):len(seq)])

    return newseq


def import_spec_cat(path_kidscats, kidscatname, kidscat_end, specz_file, \
                    src_selection, cat_version):
    filename = '../*specweight.cat'
    if specz_file is None:
        specz_file = os.path.join(path_kidscats, filename)
    files = glob(specz_file)
    if len(files) == 0:
        msg = 'Spec-z file {0} not found.'.format(filename)
        raise IOError(msg)
    elif len(files) > 1:
        msg = 'Spec-z file name {0} ambiguous. Using file {1}.'.format(
                filename, files[0])
    spec_cat = pyfits.open(files[0], memmap=True)[1].data
    

    Z_S = spec_cat['z_spec']
    spec_weight = spec_cat['spec_weight']
    manmask = spec_cat['MASK']
    
    srcmask = (manmask==0)

    # We apply any other cuts specified by the user for Z_B
    srclims = src_selection['Z_B'][1]
    if len(srclims) == 1:
        srcmask *= (spec_cat['Z_B'] == srclims[0])
    else:
        srcmask *= (srclims[0] <= spec_cat['Z_B']) &\
            (spec_cat['Z_B'] < srclims[1])

    return Z_S[srcmask], spec_weight[srcmask]



def import_spec_cat_mocks(path_kidscats, kidscatname, kidscat_end, specz_file, \
                    src_selection, cat_version):
    
    spec_cat = pyfits.open(path_kidscats, memmap=True)[1].data


    Z_S = spec_cat['z_spectroscopic']
    spec_weight = np.ones(Z_S.shape, dtype=np.float64)
    
    srcmask = (spec_weight==1)
    
    # We apply any other cuts specified by the user for Z_B
    srclims = src_selection['z_photometric'][1]
    if len(srclims) == 1:
        srcmask *= (spec_cat['z_photometric'] == srclims[0])
    else:
        srcmask *= (srclims[0] <= spec_cat['z_photometric']) &\
            (spec_cat['z_photometric'] < srclims[1])

    return Z_S[srcmask], spec_weight[srcmask]



"""
#Temp
def import_spec_cat_pz(path_kidscats, kidscatname, kidscat_end, \
                    src_selection, cat_version):
    
    print('Using direct callibration to estimate the redshifts.')
    files = os.listdir(os.path.dirname('/disks/shear10/KiDS/All_tiles/'))
    Z_S_varlist = np.zeros(351)
    for i, filename in enumerate(files):
        print i
        try:
            spec_cat = pyfits.open('/disks/shear10/KiDS/All_tiles/%s'%filename, memmap=True)[1].data
        
            Z_S = spec_cat['PZ_full']
            manmask = spec_cat['MASK']
        
            srcmask = (manmask==0)
        
            # We apply any other cuts specified by the user for Z_B
            srclims = src_selection['Z_B'][1]
            if len(srclims) == 1:
                srcmask *= (spec_cat['Z_B'] == binlims[0])
            else:
                srcmask *= (srclims[0] <= spec_cat['Z_B']) &\
                (spec_cat['Z_B'] < srclims[1])
            Z_S = Z_S[srcmask]
    
            Z_S_varlist += Z_S.sum(axis=0)
            print Z_S_varlist[:20]
        
        except:
            pass
    

    print Z_S_varlist.shape
    return Z_S_varlist, 1.0
"""

"""
#Temp
def import_spec_cat_pz(kidscatname, catmatch, srcNr):
    
    #print('Using direct callibration to estimate the redshifts.')
    
    files = os.listdir(os.path.dirname('/disks/shear10/KiDS/All_tiles/'))
    filename = str(fnmatch.filter(files, kidscatname+'*')[0])
    print filename
    spec_cat = pyfits.open('/disks/shear10/KiDS/All_tiles/%s'%filename, memmap=True)[1].data
    
    Z_S = spec_cat['PZ_full']
    mask = np.in1d(spec_cat['SeqNr'],srcNr)
    Z_S = Z_S[mask]
    Z_S_out = np.zeros((len(srcNr), 70))
    for i in range(len(srcNr)):
        Z_S_out[i,:] = np.interp(np.linspace(0, 351, 70), np.linspace(0, 351, 351), Z_S[i,:])
        Z_S_out[i,:] = Z_S_out[i,:]/Z_S_out[i,:].sum()
    
    # We apply any other cuts specified by the user for Z_B
    return Z_S_out
"""


def import_spec_wizz(path_kidscats, kidscatname, kidscat_end, \
                    src_selection, cat_version, filename_var, ncores):
    
    # Making selection of sources ...
    try:
        pattern = 'KiDS_COSMOS_DEEP2_stomp_masked.cat'
    
        files = os.listdir(os.path.dirname('%s'%(path_kidscats)))
    
        filename = str(fnmatch.filter(files, pattern)[0])
    
        spec_cat_file = os.path.dirname('%s'%(path_kidscats))+'/%s'%(filename)
        path_wizz_data = os.path.dirname('%s'%(path_kidscats))
        spec_cat = pyfits.open(spec_cat_file, memmap=True)[1].data
        print()
        print('Using The-wiZZ to estimate the redshifts.')
    except:
        print()
        print('Cannot run The-wiZZ, please check the required files.')
        raise SystemExit()


    if os.path.isfile('%s/KiDS_COSMOS_DEEP2_stomp_masked_%s.ascii'%(\
                                                        path_wizz_data,\
                                                          filename_var)):
        print('Loading precomputed The-wiZZ redshifts...')
        print()
        n_z = np.genfromtxt('%s/KiDS_COSMOS_DEEP2_stomp_masked_%s.ascii'%(\
                                                                path_wizz_data,\
                                                                filename_var),\
                                                                comments='#')
    else:
        
        # Setting The-wiZZ parameters (for now hardcoded)
        input_pair_hdf5_file = '%s/KiDS_COSMOS_DEEP2_The-wiZZ_pairs.hdf5'%(\
                                                                path_wizz_data)
        #use_inverse_weighting = True
        n_target_load_size = 10000
        z_n_bins = 70
        z_max = 3.5
        n_bootstrap = 1000
        z_binning_type = 'linear'
        pair_scale_name = 'kpc30t300'
        n_processes = ncores
        z_min = 0.025
        #bootstrap_samples = None
        #output_bootstraps_file = None
        unknown_stomp_region_name = 'stomp_region'
        unknown_index_name = 'SeqNr'
        unknown_weight_name = 'recal_weight'
        
        srclims = src_selection['Z_B'][1]
        src_z_max = srclims[1]
        src_z_min = srclims[0]
        step = np.ceil((src_z_max-src_z_min)/0.1)
        n_loops = np.linspace(src_z_min, src_z_max, step, endpoint=True)
        
        
        n_z = np.zeros((z_n_bins, len(n_loops)-1))
        w_i = np.zeros((z_n_bins, len(n_loops)-1))

        for i in xrange(len(n_loops)-1):
            manmask = spec_cat['MASK']
            srcmask = 0
            srcmask = (manmask==0)
            srcmask *= (n_loops[i] <= spec_cat['Z_B']) \
                        & (spec_cat['Z_B'] < n_loops[i+1])
            print('Preselecting sources in photo-z range between %f and %f'%(\
                                                    n_loops[i], n_loops[i+1]))
            # Writing reduced catalog with only selected sources ...
            output_cat = spec_cat[srcmask]
            #orig_cols = spec_cat_file[1].columns
            hdu = pyfits.BinTableHDU(output_cat)
            hdu.writeto('%s/KiDS_COSMOS_DEEP2_stomp_masked_%s.cat'%(\
                                                            path_wizz_data,\
                                                            filename_var),\
                                                            clobber=True)


            # Running The-wiZZ to obtain Z_S
            directory = os.path.dirname(os.path.dirname(__file__))
            indirectory = os.listdir(directory)
            
            unknown_sample_file = '%s/KiDS_COSMOS_DEEP2_stomp_masked_%s.cat'%(\
                                                                path_wizz_data,\
                                                                filename_var)
            output_pdf_file_name = '%s/KiDS_COSMOS_DEEP2_stomp_masked_%s_preselect_%i.ascii'%(path_wizz_data, filename_var, i)

            if 'The-wiZZ' in indirectory:
                path_shearcodes = directory + '/' + 'The-wiZZ' + '/'
            else:
                print('Cannot locate The-wiZZ in the pipeline instalation.')
                raise SystemExit()

            ps = []
            codename = '%spdf_maker.py'%(path_shearcodes)
            runname = 'python %s'%codename
            runname += ' --input_pair_hdf5_file %s --unknown_sample_file %s --output_pdf_file_name %s --use_inverse_weighting --n_target_load_size %i --z_n_bins %i --z_max %f --n_bootstrap %i --z_binning_type %s --pair_scale_name %s --n_processes %i --z_min %f --unknown_stomp_region_name %s --unknown_index_name %s --unknown_weight_name %s'%(input_pair_hdf5_file, unknown_sample_file, output_pdf_file_name, n_target_load_size, z_n_bins, z_max, n_bootstrap, z_binning_type , pair_scale_name, n_processes, z_min, unknown_stomp_region_name, unknown_index_name, unknown_weight_name)

            try:
                p = sub.Popen(shlex.split(runname))
                ps.append(p)
                for p in ps:
                    p.wait()

            except:
                print()
                print('Cannot run The-wiZZ, please check the required files.')
                raise SystemExit()


            # Reading in the calculated Z_S from The-wiZZ output file
            n_z[:,i] = np.nan_to_num(np.genfromtxt(
                '%s/KiDS_COSMOS_DEEP2_stomp_masked_%s_preselect_%i.ascii' \
                    %(path_wizz_data, filename_var, i),
                comments='#')[:,1])
            w_i[:,i] = np.nan_to_num(np.genfromtxt(
                '%s/KiDS_COSMOS_DEEP2_stomp_masked_%s_preselect_%i.ascii' \
                    %(path_wizz_data, filename_var, i),
                comments='#')[:,3])
            #print n_z[:,i], w_i[:,i]

        n_z = np.sum(n_z*w_i, axis=1)/np.sum(w_i, axis=1)
        np.savetxt('%s/KiDS_COSMOS_DEEP2_stomp_masked_%s.ascii' \
                        %(path_wizz_data, filename_var),
                   n_z, delimiter='\t')
        n_z[n_z < 0] = 0

    return np.nan_to_num(n_z)


# Import and mask all used data from the sources in this KiDS field
def import_kidscat(path_kidscats, kidscatname, kidscat_end, \
                   src_selection, cat_version):
    
    # Full directory & name of the corresponding KiDS catalogue
    if cat_version == 2:
        kidscatfile = '%s/%s_%s'%(path_kidscats, kidscatname, kidscat_end)
        kidscat = pyfits.open(kidscatfile, memmap=True)[1].data
    
    if cat_version == 3:
        kidscatfile = '%s/%s'%(path_kidscats, kidscatname)
        kidscat = pyfits.open(kidscatfile, memmap=True)[2].data
    
    if cat_version == 0:
        return import_kids_mocks(path_kidscats, kidscatname, kidscat_end, \
                                 src_selection, cat_version)
    
    # List of the ID's of all sources in the KiDS catalogue
    srcNr = kidscat['SeqNr']
    #srcNr = kidscat['SeqNr_field'] # If ever needed for p(z)
    # List of the RA's of all sources in the KiDS catalogue
    srcRA = kidscat['ALPHA_J2000']
    # List of the DEC's of all sources in the KiDS catalogue
    srcDEC = kidscat['DELTA_J2000']
    

    if cat_version == 3:
        w = np.transpose(np.array([kidscat['weight_A'], kidscat['weight_B'], \
                                   kidscat['weight_C'], kidscat['weight_C']]))
        srcPZ = kidscat['Z_B']
        SN = kidscat['model_SNratio']
        manmask = kidscat['MASK']
        tile = kidscat['THELI_NAME']
        
    elif cat_version == 2:
        srcPZ = kidscat['PZ_full'] # Full P(z) probability function
        w = np.transpose(np.array([kidscat['weight'], kidscat['weight'], \
                                   kidscat['weight'], kidscat['weight']]))
                                   
        # The Signal to Noise of the sources (needed for bias)
        SN = kidscat['SNratio']
        # The manual masking of bad sources (0=good, 1=bad)
        manmask = kidscat['MAN_MASK']
        tile = np.zeros(srcNr.size, dtype=np.float64)
    
    if cat_version == 2:
        srcm = kidscat['m_cor'] # The multiplicative bias m
    if cat_version == 3:
        # Values are hardcoded, if image simulations give
        # different results this must be changed!
        srcm = np.zeros(srcNr.size, dtype=np.float64)
        srcm[(0.1 < srcPZ) & (srcPZ <= 0.2)] = -0.0165984884074
        srcm[(0.2 < srcPZ) & (srcPZ <= 0.3)] = -0.0107643100825
        srcm[(0.3 < srcPZ) & (srcPZ <= 0.4)] = -0.0163154916657
        srcm[(0.4 < srcPZ) & (srcPZ <= 0.5)] = -0.00983059386823
        srcm[(0.5 < srcPZ) & (srcPZ <= 0.6)] = -0.0050563715617
        srcm[(0.6 < srcPZ) & (srcPZ <= 0.7)] = -0.00931232658151
        srcm[(0.7 < srcPZ) & (srcPZ <= 0.8)] = -0.0135538269718
        srcm[(0.8 < srcPZ) & (srcPZ <= 0.9)] = -0.0286749355629
            
            
    e1_A = kidscat['e1_A']
    e1_B = kidscat['e1_B']
    e1_C = kidscat['e1_C']
    try:
        e1_D = kidscat['e1_D']
    except:
        e1_D = kidscat['e1_C']

    e2_A = kidscat['e2_A']
    e2_B = kidscat['e2_B']
    e2_C = kidscat['e2_C']
    try:
        e2_D = kidscat['e2_D']
    except:
        e2_D = kidscat['e2_C']

    try:
        c1_A = kidscat['c1_A']
        c1_B = kidscat['c1_B']
        c1_C = kidscat['c1_C']
        try:
            c1_D = kidscat['c1_D']
        except:
            c1_D = kidscat['c1_C']

        c2_A = kidscat['c2_A']
        c2_B = kidscat['c2_B']
        c2_C = kidscat['c2_C']
        try:
            c2_D = kidscat['c2_D']
        except:
            c2_D = kidscat['c2_C']
    
    
    except:
        c1_A = np.zeros(srcNr.size, dtype=np.float64)
        c1_B = np.zeros(srcNr.size, dtype=np.float64)
        c1_C = np.zeros(srcNr.size, dtype=np.float64)
        c1_D = np.zeros(srcNr.size, dtype=np.float64)
        c2_A = np.zeros(srcNr.size, dtype=np.float64)
        c2_B = np.zeros(srcNr.size, dtype=np.float64)
        c2_C = np.zeros(srcNr.size, dtype=np.float64)
        c2_D = np.zeros(srcNr.size, dtype=np.float64)
    
    # The corrected e1 and e2 for all blind catalogs
    e1 = np.transpose(np.array([e1_A-c1_A, e1_B-c1_B, e1_C-c1_C, e1_D-c1_D]))
    e2 = np.transpose(np.array([e2_A-c2_A, e2_B-c2_B, e2_C-c2_C, e2_D-c2_D]))

    # Masking: We remove sources with weight=0 and those masked by the catalog
    if cat_version == 2:
        srcmask = (w.T[0]>0.0)&(SN>0.0)&(srcm<0.0)&(manmask==0)&(-1<c1_A)
    if cat_version == 3:
        srcmask = (w.T[0]>0.0)&(SN > 0.0)&(manmask==0)&(srcm!=0)
        # srcm != 0 removes all the sources that are not in 0.1 to 0.9 Z_B range

    # We apply any other cuts specified by the user
    for param in src_selection.keys():
        srclims = src_selection[param][1]
        if len(srclims) == 1:
            srcmask *= (kidscat[param] == srclims[0])

        else:
            srcmask *= (srclims[0] <= kidscat[param]) & \
                        (kidscat[param] < srclims[1])


    srcNr = srcNr[srcmask]
    srcRA = srcRA[srcmask]
    srcDEC = srcDEC[srcmask]
    w = w[srcmask]
    srcPZ = srcPZ[srcmask]
    srcm = srcm[srcmask]
    e1 = e1[srcmask]
    e2 = e2[srcmask]
    tile = tile[srcmask]

    return srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm, tile


def import_kids_mocks(path_kidscats, kidscatname, kidscat_end, \
                   src_selection, cat_version):
    
    # Full directory & name of the corresponding KiDS catalogue
    kidscatfile = '%s'%path_kidscats
    kidscat = pyfits.open(kidscatfile, memmap=True)[1].data


    srcRA = (kidscat['x_arcmin']/60.0) + 150.0
    srcDEC = kidscat['y_arcmin']/60.0
    
    srcNr = np.arange(srcRA.size, dtype=np.float64)

    w = np.ones(srcNr.size, dtype=np.float64)
    w = np.transpose(np.array([w, w, w, w]))
    srcPZ = kidscat['z_photometric']
    tile = np.empty(srcNr.size, dtype=object)
    for i in xrange(10):
        for j in xrange(10):
            cond = (srcRA > i+150.0) & (srcRA < i+1+150.0) & (srcDEC > j) & (srcDEC < j+1)
            tile[cond] = path_kidscats.split('/', -1)[-1]+'-'+str(i*10+j)

    srcm = np.zeros(srcNr.size, dtype=np.float64) # The multiplicative bias m

    e1 = np.transpose(np.array([kidscat['eps_obs1'], kidscat['eps_obs1'], kidscat['eps_obs1'], kidscat['eps_obs1']]))
    e2 = np.transpose(np.array([kidscat['eps_obs2'], kidscat['eps_obs2'], kidscat['eps_obs2'], kidscat['eps_obs2']]))

    srcmask = (srcm==0.0)
    for param in src_selection.keys():
        srclims = src_selection[param][1]
        if len(srclims) == 1:
            srcmask *= (kidscat[param] == srclims[0])

    else:
        srcmask *= (srclims[0] <= kidscat[param]) & \
            (kidscat[param] < srclims[1])


    srcNr = srcNr[srcmask]
    srcRA = srcRA[srcmask]
    srcDEC = srcDEC[srcmask]
    w = w[srcmask]
    srcPZ = srcPZ[srcmask]
    srcm = srcm[srcmask]
    e1 = e1[srcmask]
    e2 = e2[srcmask]
    tile = tile[srcmask]
    
    return srcNr, srcRA, srcDEC, w, srcPZ, e1, e2, srcm, tile


# Calculating the variance of the ellipticity for this source selection
def calc_variance(e1_varlist, e2_varlist, w_varlist):

    e1_mean = np.sum(w_varlist*e1_varlist, 1)/np.sum(w_varlist, 1)
    e2_mean = np.sum(w_varlist*e2_varlist, 1)/np.sum(w_varlist, 1)

    e1_mean = np.reshape(e1_mean, [len(e1_mean), 1])
    e2_mean = np.reshape(e1_mean, [len(e2_mean), 1])

    weight = np.sum(w_varlist, 1)/(np.sum(w_varlist, 1)**2 - \
                                   np.sum(w_varlist**2, 1))

    var_e1 = weight * np.sum(w_varlist*(e1_varlist-e1_mean)**2, 1)
    var_e2 = weight * np.sum(w_varlist*(e2_varlist-e2_mean)**2, 1)

    variance = np.mean([var_e1, var_e2], 0)

    print('Variance (A,B,C,D):', variance)
    print('Sigma (A,B,C,D):', variance**0.5)

    return variance


# Create a number of observable bins containing the same number of lenses
def create_obsbins(binname, Nobsbins, lenssel_binning, gamacat):

    obslist = gamacat[binname]

    # We use only selected lenses that have real values for the binning
    nanmask = np.isfinite(obslist) & lenssel_binning
    obslist = obslist[nanmask]

    # Max value of the observable
    obslist_max = np.amax(obslist)

    # Create a number of observable bins of containing an equal number of lenses
    # Sort the observable values
    sorted_obslist = np.sort(obslist)
    
    # Determine the number of objects in each bin
    obsbin_size = len(obslist)/Nobsbins

    obsbins = np.array([])
    
    # For every observable bin
    # append the observable value that contains the determined number of objects
    for o in xrange(Nobsbins):
        obsbins = np.append(obsbins, sorted_obslist[o*obsbin_size])
    
    # Finally, append the max value of the observable
    obsbins = np.append(obsbins, obslist_max)
    
    return obsbins


# Binnning information of the groups
def define_obsbins(binnum, lens_binning, lenssel_binning, gamacat,
                   Dcllist=[], galZlist=[]):

    # Check how the binning is given
    binname = lens_binning.keys()[0]
    if 'No' not in binname:
        
        if 'ID' in binname:
            Nobsbins = len(lens_binning.keys())
            if len(lenssel_binning) > 0:
                print('Lens binning: Lenses divided in %i lens-ID bins' \
                      %(Nobsbins))

        else:
            obsbins = lens_binning[binname][1]
            obsfile = lens_binning[binname][0]
            if len(obsbins) == 1:
                Nobsbins = int(obsbins[0])
                if len(lenssel_binning) > 0:
                    obsbins = create_obsbins(binname, Nobsbins, \
                                             lenssel_binning, gamacat)
                    lens_binning = {binname: [obsfile, obsbins]}
            else:
                Nobsbins = len(obsbins)-1 # If the bin limits are given
            
            # Print the lens binning properties
            if len(lenssel_binning) > 0:
                
                # Importing the binning file
                if obsfile == 'self':
                    obslist = define_obslist(
                        binname, gamacat, 0.7, Dcllist, galZlist)
                else:
                    print('Using %s from %s' %(binname, obsfile))
                    obscat = pyfits.open(obsfile)[1].data
                    obslist = obscat[binname]

                
                print()
                print('Lens binning: Lenses divided in %i %s-bins' \
                      %(Nobsbins, binname))
                print('%s Min:          Max:          Mean:'%binname)
                for b in xrange(Nobsbins):
                    lenssel = lenssel_binning & (obsbins[b] <= obslist) \
                                & (obslist < obsbins[b+1])
                    print('%g    %g    %g' \
                          %(obsbins[b], obsbins[b+1],
                            np.mean(obslist[lenssel])))
            
    else: # If there is no binning
        obsbins = np.array([-999, -999])
        binname = 'No'
        Nobsbins = 1
        lens_binning = {binname: ['self', obsbins]}

    # Try to give the current binning values
    try:
        bins = np.sort([obsbins[binnum-1], obsbins[binnum]])
        binmin = float(bins[0])
        binmax = float(bins[1])
    except:
        binmin = -999
        binmax = -999

    return binname, lens_binning, Nobsbins, binmin, binmax


# Corrections on GAMA catalog observables
def define_obslist(obsname, gamacat, h, Dcllist=[], galZlist=[]):

    obslist = gamacat[obsname]

    if 'AngSep' in obsname and len(Dcllist) > 0:
        print('Applying cosmology correction to "AngSep"')

        #Dclgama = np.array([distance.comoving(z, 0.25, 0.75, 1.)
        #                    for z in gamacat['Z']])
        
        # New method
        cosmogama = LambdaCDM(H0=100., Om0=0.25, Ode0=0.75)
        Dclgama = (cosmogama.comoving_distance(galZlist).to('pc')).value
        
        corr_list = Dcllist/Dclgama
        obslist = obslist * corr_list

    if 'logmstar' in obsname:
        print('Applying fluxscale correction to "logmstar"')

        # Fluxscale, needed for stellar mass correction
        fluxscalelist = gamacat['fluxscale']
        corr_list = np.log10(fluxscalelist) - 2.*np.log10(h/0.7)
        obslist = obslist + corr_list

    return obslist


# Masking the lenses according to the appropriate
# lens selection and the current KiDS field
def define_lenssel(gamacat, colnames, centering, lens_selection, lens_binning,
                   binname, binnum, binmin, binmax, Dcllist, galZlist, h):

    lenssel = np.ones(len(gamacat[colnames[0]]), dtype=bool)
    # introduced by hand (CS) for the case when I provide a lensID_file:
    #binname = 'No'
    
    # Add the mask for each chosen lens parameter
    for param in lens_selection.keys():
        binlims = lens_selection[param][1]
        obsfile = lens_selection[param][0]
        # Importing the binning file
        if obsfile == 'self':
            obslist = define_obslist(param, gamacat, h, Dcllist, galZlist)
        else:
            print('Using %s from %s'%(param, obsfile))
            bincat = pyfits.open(obsfile)[1].data
            obslist = bincat[param]
        
        if colnames[0] in param:
            lenssel *= np.in1d(obslist, binlims)
        else:
            if len(binlims) == 1:
                lenssel *= (obslist == binlims[0])
            else:
                lenssel *= (binlims[0] <= obslist) & (obslist < binlims[1])

    if 'No' not in binname: # If the galaxy selection depends on observable
        # Importing the binning file
        obsfile = lens_binning[binname][0]
        if obsfile == 'self':
            if colnames[0] in binname:
                obslist = define_obslist(colnames[0], gamacat, h, Dcllist, galZlist)
            else:
                obslist = define_obslist(binname, gamacat, h, Dcllist, galZlist)
        else:
            print('Using %s from %s'%(binname, obsfile))
            bincat = pyfits.open(obsfile)[1].data
            obslist = bincat[binname]
        
        if 'ID' in binname:
            lensids = lens_binning['%s%i'%(binname[:-1], binnum+1)][1]
            lenssel *= np.in1d(obslist, lensids)
        else:
            lenssel *= (binmin <= obslist) & (obslist < binmax)

    return lenssel


# Calculate Sigma_crit (=1/k) and the weight mask for every lens-source pair
def calc_Sigmacrit(Dcls, Dals, Dcsbins, srcPZ, cat_version, Dc_epsilon):
    
    # Calculate the values of Dls/Ds for all lens/source-redshift-bin pair
    Dcls, Dcsbins = np.meshgrid(Dcls, Dcsbins)
    DlsoDs = (Dcsbins-Dcls)/Dcsbins

    # Mask all values with Dcl=0 (Dls/Ds=1) and Dcl>Dcsbin (Dls/Ds<0)
    #DlsoDsmask = np.logical_not((0.<DlsoDs) & (DlsoDs<1.))
    #DlsoDs = np.ma.filled(np.ma.array(DlsoDs, mask=DlsoDsmask, fill_value=0))
    #DlsoDs[np.logical_not((0.< DlsoDs) & (DlsoDs < 1.))] = 0.0
    
    if cat_version == 3:
        DlsoDs[np.logical_not(((Dc_epsilon/Dcsbins) < DlsoDs) & (DlsoDs < 1.))] = 0.0
    else:
        DlsoDs[np.logical_not((0.< DlsoDs) & (DlsoDs < 1.))] = 0.0

    DlsoDsmask = [] # Empty unused lists

    # Matrix multiplication that sums over P(z),
    # to calculate <Dls/Ds> for each lens-source pair
    DlsoDs = np.dot(srcPZ, DlsoDs).T

    # Calculate the values of k (=1/Sigmacrit)
    Dals = np.reshape(Dals,[len(Dals),1])
    k = 1 / ((c.value**2)/(4*np.pi*G.value) * 1/(Dals*DlsoDs)) # k = 1/Sigmacrit

    DlsoDs = [] # Empty unused lists
    Dals = []

    Dcls = [] # Empty unused lists
    Dcsbins = []
    # Create the mask that removes all sources with k not between 0 and infinity
    kmask = np.logical_not((0. < k) & (k < inf))

    gc.collect()
    return k, kmask


# Weigth for average m correction in KiDS-450
def calc_mcorr_weight(Dcls, Dals, Dcsbins, srcPZ, cat_version, Dc_epsilon):
    
    # Calculate the values of Dls/Ds for all lens/source-redshift-bin pair
    Dcls, Dcsbins = np.meshgrid(Dcls, Dcsbins)
    DlsoDs = (Dcsbins-Dcls)/Dcsbins
    
    # Mask all values with Dcl=0 (Dls/Ds=1) and Dcl>Dcsbin (Dls/Ds<0)
    #DlsoDsmask = np.logical_not((0.<DlsoDs) & (DlsoDs<1.))
    #DlsoDs = np.ma.filled(np.ma.array(DlsoDs, mask=DlsoDsmask, fill_value=0))
    #DlsoDs[np.logical_not((0.< DlsoDs) & (DlsoDs < 1.))] = 0.0
    
    if cat_version == 3:
        DlsoDs[np.logical_not(((Dc_epsilon/Dcsbins) < DlsoDs) & (DlsoDs < 1.))] = 0.0
    else:
        DlsoDs[np.logical_not((0.< DlsoDs) & (DlsoDs < 1.))] = 0.0

    DlsoDsmask = [] # Empty unused lists
    Dcls = [] # Empty unused lists
    Dcsbins = []
    # Matrix multiplication that sums over P(z),
    # to calculate <Dls/Ds> for each lens-source pair
    DlsoDs = np.dot(srcPZ, DlsoDs).T
    
    gc.collect()
    return DlsoDs


# Calculate the projected distance (srcR) and the
# shear (gamma_t and gamma_x) of every lens-source pair
def calc_shear(Dals, galRAs, galDECs, srcRA, srcDEC, e1, e2, Rmin, Rmax):

    galRA, srcRA = np.meshgrid(galRAs, srcRA)
    galDEC, srcDEC = np.meshgrid(galDECs, srcDEC)

    # Defining the distance R and angle phi between the lens'
    # center and its surrounding background sources
    srcR = Dals * np.arccos(np.cos(np.radians(galDEC))*\
                            np.cos(np.radians(srcDEC))*\
                            np.cos(np.radians(galRA-srcRA))+\
                            np.sin(np.radians(galDEC))*\
                            np.sin(np.radians(srcDEC)))
    
    # Masking all lens-source pairs that have a
    # relative distance beyond the maximum distance Rmax
    Rmask = np.logical_not((Rmin < srcR) & (srcR < Rmax))
    
    galRA = np.ma.filled(np.ma.array(galRA, mask = Rmask, fill_value = 0))
    srcRA = np.ma.filled(np.ma.array(srcRA, mask = Rmask, fill_value = 0))
    galDEC = np.ma.filled(np.ma.array(galDEC, mask = Rmask, fill_value = 0))
    srcDEC = np.ma.filled(np.ma.array(srcDEC, mask = Rmask, fill_value = 0))
    srcR = np.ma.filled(np.ma.array(srcR, mask = Rmask, fill_value = 0)).T
    
    # Calculation the sin/cos of the angle (phi)
    # between the gal and its surrounding galaxies
    theta = np.arccos(np.sin(np.radians(galDEC))*np.sin(np.radians(srcDEC))+\
                      np.cos(np.radians(galDEC))*np.cos(np.radians(srcDEC))*\
                      np.cos(np.radians(galRA-srcRA))) # in radians
    incosphi = ((-np.cos(np.radians(galDEC))*(np.arctan(np.tan(np.radians(galRA-srcRA)))))**2-\
                (np.radians(galDEC-srcDEC))**2)/(theta)**2
    insinphi = 2.0*(-np.cos(np.radians(galDEC))*\
                (np.arctan(np.tan(np.radians(galRA-srcRA)))))*np.radians(galDEC-srcDEC)/(theta)**2

    incosphi = incosphi.T
    insinphi = insinphi.T

    return srcR, incosphi, insinphi


# For each radial bin of each lens we calculate the output shears and weights
def calc_shear_output(incosphilist, insinphilist, e1, e2, \
                      Rmask, klist, wlist, Nsrclist, srcm, Runit):
    
    wlist = wlist.T
    klist_t = np.array([klist, klist, klist, klist]).T
    # Calculating the needed errors
    if 'pc' not in Runit:
        wk2list = wlist
    else:
        wk2list = wlist*klist_t**2

    w_tot = np.sum(wlist, 0)
    w2_tot = np.sum(wlist**2, 0)

    k_tot = np.sum(klist, 1)
    k2_tot = np.sum(klist**2, 1)

    wk2_tot = np.sum(wk2list, 0)
    w2k4_tot = np.sum(wk2list**2, 0)
    if 'pc' not in Runit:
        w2k2_tot = np.sum(wlist**2, 0)
    else:
        w2k2_tot = np.sum(wlist**2 * klist_t**2, 0)

    wlist = []

    Nsrc_tot = np.sum(Nsrclist, 1)
    
    srcm, foo = np.meshgrid(srcm,np.zeros(klist_t.shape[1]))
    srcm = np.array([srcm, srcm, srcm, srcm]).T
    foo = [] # Empty unused lists
    srcm_tot = np.sum(srcm*wk2list, 0) # the weighted sum of the bias m
    srcm = []
    klist_t = []

    gc.collect()

    # Calculating the weighted tangential and
    # cross shear of the lens-source pairs
    gammatlists = np.zeros([4, len(incosphilist), len(incosphilist[0])])
    gammaxlists = np.zeros([4, len(incosphilist), len(incosphilist[0])])

    klist = np.ma.filled(np.ma.array(klist, mask = Rmask, fill_value = inf))
    klist = np.array([klist, klist, klist, klist]).T
    if 'pc' not in Runit:
        for g in xrange(4):
            gammatlists[g] = np.array((-e1[:,g] * incosphilist - e2[:,g] * \
                                   insinphilist) * wk2list[:,:,g].T)
            gammaxlists[g] = np.array((e1[:,g] * insinphilist - e2[:,g] * \
                                   incosphilist) * wk2list[:,:,g].T)

    else:
        for g in xrange(4):
            gammatlists[g] = np.array((-e1[:,g] * incosphilist - e2[:,g] * \
                                    insinphilist) * wk2list[:,:,g].T / \
                                    klist[:,:,g].T)
            gammaxlists[g] = np.array((e1[:,g] * insinphilist - e2[:,g] * \
                                    incosphilist) * wk2list[:,:,g].T / \
                                    klist[:,:,g].T)



    [gammat_tot_A, gammat_tot_B, gammat_tot_C, \
     gammat_tot_D] = [np.sum(gammatlists[g], 1) for g in xrange(4)]
    [gammax_tot_A, gammax_tot_B, gammax_tot_C, \
     gammax_tot_D] = [np.sum(gammaxlists[g], 1) for g in xrange(4)]

    """
    w_tot_A, w_tot_B, w_tot_C, w_tot_D = \
    w_tot.T[0], w_tot.T[1], w_tot.T[2], w_tot.T[3]
    w2_tot_A, w2_tot_B, w2_tot_C, w2_tot_D = \
    w2_tot.T[0], w2_tot.T[1], w2_tot.T[2], w2_tot.T[3]
    """
    wk2_tot_A, wk2_tot_B, wk2_tot_C, wk2_tot_D = \
    wk2_tot.T[0], wk2_tot.T[1], wk2_tot.T[2], wk2_tot.T[3]
    """
    w2k4_tot_A, w2k4_tot_B, w2k4_tot_C, w2k4_tot_D = \
    w2k4_tot.T[0], w2k4_tot.T[1], w2k4_tot.T[2], w2k4_tot.T[3]
    """
    w2k2_tot_A, w2k2_tot_B, w2k2_tot_C, w2k2_tot_D = \
    w2k2_tot.T[0], w2k2_tot.T[1], w2k2_tot.T[2], w2k2_tot.T[3]
    srcm_tot_A, srcm_tot_B, srcm_tot_C, srcm_tot_D = \
    srcm_tot.T[0], srcm_tot.T[1], srcm_tot.T[2], srcm_tot.T[3]

    gc.collect()
    """
    return gammat_tot_A, gammax_tot_A, gammat_tot_B, gammax_tot_B, \
        gammat_tot_C, gammax_tot_C, gammat_tot_D, gammax_tot_D, \
        w_tot_A, w_tot_B, w_tot_C, w_tot_D, \
        w2_tot_A, w2_tot_B, w2_tot_C, w2_tot_D, \
        k_tot, k2_tot, wk2_tot_A, wk2_tot_B, wk2_tot_C, wk2_tot_D, \
        w2k4_tot_A, w2k4_tot_B, w2k4_tot_C, w2k4_tot_D, \
        w2k2_tot_A, w2k2_tot_B, w2k2_tot_C, w2k2_tot_D, Nsrc_tot, \
        srcm_tot_A, srcm_tot_B, srcm_tot_C, srcm_tot_D
    """
    return gammat_tot_A, gammax_tot_A, gammat_tot_B, gammax_tot_B, \
        gammat_tot_C, gammax_tot_C, gammat_tot_D, gammax_tot_D, \
        k_tot, k2_tot, wk2_tot_A, wk2_tot_B, wk2_tot_C, wk2_tot_D, \
        w2k2_tot_A, w2k2_tot_B, w2k2_tot_C, w2k2_tot_D, Nsrc_tot, \
        srcm_tot_A, srcm_tot_B, srcm_tot_C, srcm_tot_D



# For each radial bin of each lens we calculate the output shears and weights
def calc_covariance_output(incosphilist, insinphilist, klist, galweights):
    
    galweights = np.reshape(galweights, [len(galweights), 1])

    # For each radial bin of each lens we calculate
    # the weighted sum of the tangential and cross shear
    Cs_tot = np.sum(-incosphilist*klist*galweights, axis=0)
    Ss_tot = np.sum(-insinphilist*klist*galweights, axis=0)
    Zs_tot = np.sum(klist**2*galweights, axis=0)
    
    return Cs_tot, Ss_tot, Zs_tot


# Write the shear or covariance catalog to a fits file
def write_catalog(filename, galIDlist, Rbins, Rcenters, nRbins, Rconst, \
                  output, outputnames, variance, purpose, e1, e2, w, srcm):
    
    fitscols = []

    Rmin = Rbins[0:nRbins]/Rconst
    Rmax = Rbins[1:nRbins+1]/Rconst

    # Adding the radial bins
    #print('galIDlist =', galIDlist, type(galIDlist[0]))
    if 'bootstrap' in purpose:
        fitscols.append(pyfits.Column(name = 'Bootstrap', format='20A', \
                                      array = galIDlist))
    else:
        # This need to be figured out, it is causing pipeline to stall if there is an empty lens list passed.
        if isinstance(galIDlist[0], basestring):
            fmt = '50A'
        elif isinstance(galIDlist[0], int):
            fmt = 'J'
        else:
            fmt = 'E'
        fitscols.append(pyfits.Column(name = 'ID', format=fmt, \
                                      array = galIDlist))

    fitscols.append(pyfits.Column(name = 'Rmin', format = '%iD'%nRbins, \
                                  array = [Rmin]*len(galIDlist)))
    fitscols.append(pyfits.Column(name = 'Rmax', format='%iD'%nRbins, \
                                  array = [Rmax]*len(galIDlist)))
    fitscols.append(pyfits.Column(name = 'Rcenter', format='%iD'%nRbins, \
                                  array = [Rcenters]*len(galIDlist)))

    # Adding the output
    [fitscols.append(pyfits.Column(name = outputnames[c], \
                                   format = '%iD'%nRbins, \
                                   array = output[c])) \
    for c in xrange(len(outputnames))]

    if 'covariance' in purpose:
        fitscols.append(pyfits.Column(name = 'e1', format='4D', array= e1))
        fitscols.append(pyfits.Column(name = 'e2', format='4D', array= e2))
        fitscols.append(pyfits.Column(name = 'lfweight', format='4D', array= w))
        fitscols.append(pyfits.Column(name = 'bias_m', format='1D', array= srcm))

    # Adding the variance for the 4 blind catalogs
    fitscols.append(pyfits.Column(name = 'variance(e[A,B,C,D])', format='4D', \
                                  array= [variance]*len(galIDlist)))

    cols = pyfits.ColDefs(fitscols)
    tbhdu = pyfits.BinTableHDU.from_columns(cols)

    #	print
    if os.path.isfile(filename):
        os.remove(filename)
        print('Overwritting old catalog:', filename)
    else:
        print('Writting new catalog:', filename)
    print()

    tbhdu.writeto(filename)
    print('Catalog written.')
    print()

    return


# Calculating the final output values for the ESD profile
def calc_stack(gammat, gammax, wk2, w2k2, srcm, variance, blindcatnum):

    # Choosing the appropriate covariance value
    variance = variance[blindcatnum]

    ESDt_tot = gammat / wk2 # Final Excess Surface Density (tangential comp.)
    ESDx_tot = gammax / wk2 # Final Excess Surface Density (cross comp.)
    error_tot = (w2k2 / wk2**2 * variance)**0.5 # Final error
    # Final multiplicative bias (by which the signal is to be divided)
    bias_tot = (1 + (srcm / wk2))

    return ESDt_tot, ESDx_tot, error_tot, bias_tot


# Printing stacked ESD profile to a text file
def write_stack(filename, filename_var, Rcenters, Runit, ESDt_tot, ESDx_tot, error_tot, \
                bias_tot, h, variance, wk2_tot, w2k2_tot, Nsrc, blindcat, blindcats, blindcatnum, \
                galIDs_matched, galIDs_matched_infield):
    
    config_params = filename_var
    # Choosing the appropriate covariance value
    variance = variance[blindcatnum]

    if 'pc' in Runit:
        filehead = '# Radius({0})	ESD_t(h{1:g}*M_sun/pc^2)' \
                   '   ESD_x(h{1:g}*M_sun/pc^2)' \
                   '    error(h{1:g}*M_sun/pc^2)^2	bias(1+K)' \
                   '    variance(e_s)     wk2     w2k2' \
                   '     Nsources'.format(Runit, h*100)
    else:
        filehead = '# Radius({0})    gamma_t    gamma_x    error' \
                   '    bias(1+K)    variance(e_s)    wk2    w2k2' \
                   '    Nsources'.format(Runit)

    index = np.where(np.logical_not((0 < error_tot) & (error_tot < inf)))
    ESDt_tot.setflags(write=True)
    ESDx_tot.setflags(write=True)
    error_tot.setflags(write=True)
    bias_tot.setflags(write=True)
    wk2_tot.setflags(write=True)
    w2k2_tot.setflags(write=True)
    Nsrc.setflags(write=True)
    
    ESDt_tot[index] = int(-999)
    ESDx_tot[index] = int(-999)
    error_tot[index] = int(-999)
    bias_tot[index] = int(-999)
    wk2_tot[index] = int(-999)
    w2k2_tot[index] = int(-999)
    Nsrc[index] = int(-999)

    data_out = np.vstack((Rcenters.T, ESDt_tot.T, ESDx_tot.T, error_tot.T, \
                          bias_tot.T, variance*np.ones(bias_tot.shape).T, \
                          wk2_tot.T, w2k2_tot.T, Nsrc.T)).T
    fmt = ['%.4e' for i in range(data_out.shape[1])]
    fmt[-1] = '%6d'
    np.savetxt(filename, data_out, delimiter=' '*4, fmt=fmt, header=filehead)

    print('Written: ESD profile data:', filename)


    if len(galIDs_matched)>0 & (blindcat==blindcats[0]):
        # Writing galID's to another file
        galIDsname_split = filename.rsplit('_',1)
        galIDsname = '%s_lensIDs.txt'%(galIDsname_split[0])
        #kidsgalIDsname = '%s_KiDSlensIDs.txt'%(galIDsname_split[0])

        galIDs_table = Table(
            [galIDs_matched], names=("# ID's of all stacked lenses:",))
        galIDs_table.write(galIDsname,
            names=["IDs of all stacked lenses:"], quotechar='"',
            format='ascii.commented_header', overwrite=True)
        #np.savetxt(galIDsname, [galIDs_matched], delimiter=' ',
                   #header="ID's of all stacked lenses:", comments='# ')
        
        print("Written: List of all stacked lens ID's"\
                " that contribute to the signal:", galIDsname)
        
    return


# Define the labels for the plot
def define_plottitle(purpose, centering, lens_selection, \
                     binname, Nobsbins, src_selection):

    plottitle = '%s:'%purpose

    # Lens selection
    x, x, plottitle = define_filename_sel('', '', plottitle, lens_selection)
  
    # Source selection
    x, x, plottitle = define_filename_sel('', '', plottitle, src_selection)

    plottitle = plottitle.rsplit(',', 1)[0]

    return plottitle
    

# Setting up the ESD profile plot(s)
def define_plot(filename, plotlabel, plottitle, plotstyle, \
                Nsubplots, n, Runit, h):

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

        plt.suptitle(plottitle, fontsize=title_size)

    else:
        # Plot and print ESD profile to a file
        plt.title(plottitle,fontsize=title_size)

    # Define the labels for the plot
    if 'pc' in Runit:
        xlabel = r'Radial distance R (%s/h$_{%g}$)'%(Runit, h*100)
        ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h$_{%g}$ M$_{\odot}$/pc$^2$]'%(h*100)
    else:
        xlabel = r'Radial distance $\theta$ (%s)'%Runit
        ylabel = r'Shear $\gamma$'
        
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

    if all([not(np.isfinite(x)) for x in data_y]):
        data_y = np.ones(len(data_y))*1e10
        errorh = np.zeros(len(data_y))
        errorl = np.zeros(len(data_y))

    if type(Nsubplots) != int:
        data_x = data_x + n*0.1*data_x

    if 'lin' in plotstyle:

        plt.autoscale(enable=False, axis='both', tight=None)
        plt.xlim(1e1,1e4)

        if plotstyle == 'linR':
            linlabel = r'%s $\cdot$ %s'%(xlabel, ylabel)
            plt.errorbar(data_x, data_x*data_y, yerr=data_x*errorh, \
                         marker='o', ls='', label=plotlabel)

            plt.axis([1e1,1e4,-5e3,2e4])
            plt.ylim(-5e3,1e5)

        if plotstyle == 'lin':
            linlabel = r'%s'%(ylabel)
            plt.errorbar(data_x, data_y, yerr=errorh, \
                         marker='o', ls='', label=plotlabel)

            plt.axis([1e1,1e4,-20,100])
            plt.ylim(-20,100)

        if plotstyle == 'errorlin':
            linlabel = r'%s $\cdot$ Error(%s)'%(xlabel, ylabel)
            plt.plot(data_x, data_x*errorh, \
                     marker='o', label=plotlabel)

            plt.axis([1e1,1e4,-5e3,2e4])
            plt.ylim(-5e3,1e5)

        plt.ylabel(r'%s'%linlabel,fontsize=15)


    if 'log' in plotstyle:
        plt.yscale('log')
        errorl[errorl>=data_y] = ((data_y[errorl>=data_y])*0.9999999999)


        if 'pc' in Runit:
            plt.ylim(0.1, 1e3)
        else:
            plt.ylim(1e-3, 1)

        if plotstyle == 'log':
            plt.errorbar(data_x, data_y, yerr=[errorl,errorh], ls='', \
                         marker='o', label=plotlabel)
            plt.ylabel(r'%s'%ylabel,fontsize=15)

        if plotstyle == 'errorlog':
            plt.plot(data_x, errorh, marker='o', label=plotlabel)
            plt.ylabel(r'Error(%s)'%ylabel,fontsize=15)

    #plt.xlim(np.min(data_x),np.max(data_x))
    plt.xscale('log')
    
    plt.xlabel(r'%s'%xlabel,fontsize=15)

    plt.legend(loc='upper right',ncol=1, prop={'size':12})

    return


# Writing the ESD profile plot
def write_plot(plotname, plotstyle): # Writing and showing the plot

    #	# Make use of TeX
    rc('text',usetex=True)

    # Change all fonts to 'Computer Modern'
    rc('font',**{'family':'serif','serif':['Computer Modern']})


    file_ext = plotname.split('.')[-1]
    plotname = plotname.replace('.%s'%file_ext,'_%s.png'%plotstyle)

    path = os.path.split(plotname)[0]
    if path and not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(plotname, format='png')
    print('Written: ESD profile plot:', plotname)
    #	plt.show()
    plt.close()

    return


# Plotting the analytical or bootstrap covariance matrix
def plot_covariance_matrix(filename, plottitle1, plottitle2, plotstyle, \
                           binname, lens_binning, Rbins, Runit, h,
                           cmap='gray_r'):

    # Make use of TeX
    rc('text',usetex=True)

    # Change all fonts to 'Computer Modern'
    rc('font',**{'family':'serif','serif':['Computer Modern']})

    # Number of observable bins
    obsbins = (lens_binning.values()[0])[1]
    Nobsbins = len(obsbins)-1

    # Number and values of radial bins
    nRbins = len(Rbins)-1

    # Plotting the ueber matrix
    fig = plt.figure(figsize=(12,10))

    gs_full = gridspec.GridSpec(1,2, width_ratios=[20,1])
    gs = gridspec.GridSpecFromSubplotSpec(Nobsbins, Nobsbins, wspace=0, \
                                          hspace=0, subplot_spec=gs_full[0,0])
    gs_bar = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[1,3,1], \
                                              subplot_spec=gs_full[0,1])
                                              
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
                mappable = ax_sub.pcolormesh(Rbins, Rbins, 
                                             covariance[N1,N2,:,:],
                                             vmin=-1e7, vmax=1e7, cmap=cmap)
            if plotstyle == 'covlog':
                mappable = ax_sub.pcolormesh(Rbins, Rbins,
                                             abs(covariance[N1,N2,:,:]),
                                             norm=LogNorm(vmin=1e-7,vmax=1e7),
                                             cmap=cmap)
            if plotstyle == 'corlin':
                mappable = ax_sub.pcolormesh(Rbins, Rbins,
                                             correlation[N1,N2,:,:],
                                             vmin=-1, vmax=1, cmap=cmap)
                                             #cmap=cmap)
            if plotstyle == 'corlog':
                mappable = ax_sub.pcolormesh(Rbins, Rbins,
                                             abs(correlation)[N1,N2,:,:],
                                             #norm=LogNorm(vmin=1e-5,vmax=1e0),
                                             cmap=cmap)

            ax_sub.set_xlim(Rbins[0],Rbins[-1])
            ax_sub.set_ylim(Rbins[0],Rbins[-1])


            if N1 != 0:
                ax_sub.tick_params(axis='x', labelbottom='off')
            if N2 != 0:
                ax_sub.tick_params(axis='y', labelleft='off')

            if 'No' not in binname: # If there is binning
                if N1 == Nobsbins - 1:
                    ax_sub.xaxis.set_label_position('top')
                    if N2 == 0:
                        ax_sub.set_xlabel(r'%s = %.3g - %.3g' \
                                          %(binname, obsbins[N2],
                                            obsbins[N2+1]),
                                          fontsize=12)
                    else:
                        ax_sub.set_xlabel(r'%.3g - %.3g' \
                                          %(obsbins[N2], obsbins[N2+1]),
                                          fontsize=12)

                if N2 == Nobsbins - 1:
                    ax_sub.yaxis.set_label_position('right')
                    if N1 == 0:
                        ax_sub.set_ylabel(r'%s = %.3g - %.3g' \
                                          %(binname, obsbins[N1],
                                            obsbins[N1+1]),
                                          fontsize=12)
                    else:
                        ax_sub.set_ylabel(r'%.3g - %.3g' \
                                          %(obsbins[N1], obsbins[N1+1]),
                                          fontsize=12)


    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off',\
                   left='off', right='off')

    if 'pc' in Runit:
        #labelunit = '%s/h$_{%g}$'%(Runit, h*100)
        labelunit = '{0}/h$_{{{1:g}}}$'.format(Runit, h*100)
    else:
        labelunit = Runit
    
    ax.set_xlabel(r'Radial distance ({0})'.format(labelunit))
    ax.set_ylabel(r'Radial distance ({0})'.format(labelunit))


    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.yaxis.set_label_coords(-0.05, 0.5)

    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)

    plt.text(0.5, 1.08, plottitle1, horizontalalignment='center',
             fontsize=17, transform=ax.transAxes)
    plt.text(0.5, 1.05, plottitle2, horizontalalignment='center',
             fontsize=17, transform=ax.transAxes)
    plt.colorbar(mappable, cax=cax, orientation='vertical')

    file_ext = filename.split('.')[-1]
    plotname = filename.replace('.%s'%file_ext,'_%s.png'%plotstyle)

    path = os.path.split(plotname)[0]
    if path and not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(plotname,format='png')

    print('Written: Covariance matrix plot:', plotname)









