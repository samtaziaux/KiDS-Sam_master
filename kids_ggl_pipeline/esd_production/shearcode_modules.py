#!/usr/bin/python
"""
# This contains all the modules that are needed to
# calculate the shear profile catalog and the covariance.
"""
import astropy.io.fits as pyfits
import numpy as np
import sys
import os
import fnmatch
import time
from astropy import constants as const, units as u
import glob
import gc

import distance
import esd_utils

# Important constants(very preliminary!)
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
pix = 0.187 # Used to translate pixel to arcsec
alpha = 0.057 # Used to calculate m
beta = -0.37 # Used to calculate m
inf = np.inf


def input_variables():

    # Input for the codes
    try:
        Nsplit = int(sys.argv[1])-1 # The number of this particular core/split
        Nsplits = int(sys.argv[2]) # The number cores/splits
        binnum = int(sys.argv[3]) # The number of this particular observable bin
        blindcat = str(sys.argv[4]) # The number of this blind KiDS catalog
        config_file = str(sys.argv[5]) # The path to the configuration file
    except:
        Nsplit = 1 # The number of this particular core/split
        Nsplits = 1 # The number cores/splits
        binnum = 1 # The number of this particular observable bin
        blindcat = 'D' # The number of this particular blind KiDS catalog
        config_file = str(sys.argv[1]) # The path to the configuration file

        print 'Warning: Input not found!'

    # Importing the input parameters from the config file
    path_kidscats, path_gamacat, O_matter, O_lambda, Ok, h, \
    path_output, filename_addition, purpose, path_Rbins, Runit, Ncores, \
    lensid_file, lens_weights, lens_binning, lens_selection, \
    src_selection, cat_version, blindcats = esd_utils.read_config(config_file)

    print
    print 'Running:', purpose

    # Defining the number of the blind KiDS catalogue
    if blindcat == 'A':
        blindcatnum = 0
    if blindcat == 'B':
        blindcatnum = 1
    if blindcat == 'C':
        blindcatnum = 2
    if blindcat == 'D':
        blindcatnum = 3


    # Defining the addition to the file name
    if filename_addition == 'None':
        filename_addition = ''
    else:
        filename_addition = '_%s'%filename_addition
    
    
    # Binnning information of the lenses
    obsbins = define_obsbins(1, lens_binning, [], [])
    binname, lens_binning, Nobsbins, binmin, binmax = obsbins
    
    # Defining the lens-ID lens selection/binning
    if 'None' not in lensid_file:
        selection = define_lensid_selection(lensid_file, lens_selection, \
                                            lens_binning, binname)
        lens_selection, lens_binning, binname = selection
    
    # Defining the center definition
    centers = np.array(['Cen', 'IterCen', 'BCG'])
    centering = 'None'
    for cen in centers:
        if ('rank%s'%cen in binname) or \
            ('rank%s'%cen in lens_selection.keys()):
            centering = cen
            print 'Center definition = %s'%centering
    if centering == 'Cen':
        lens_selection['rank%s'%centering] = ['self', np.array([1])]
        msg = 'WARNING: With the Cen definition,'
        msg += ' you can only use Centrals (Rank = 1)'
        print msg

    # Creating all necessary folders

    # Path containing the output folders
    path_output = '%s/output_%sbins%s' \
                  %(path_output, binname, filename_addition)
    path_catalogs = '%s/catalogs' %(path_output.rsplit('/',1)[0])

    # Path to the output splits and results
    path_splits = '%s/splits_%s' %(path_output, purpose)
    path_results = '%s/results_%s' %(path_output, purpose)
    print path_splits

    if (Nsplit == 0) and (blindcat == blindcats[0]) and (binnum == Nobsbins):

        #print 'Nsplit:', Nsplit
        #print 'blindcat:', blindcat
        #print 'binnum:', binnum
        #print 'Nobsbins:', Nobsbins

        for path in [path_output, path_catalogs, path_splits, path_results]:
            if not os.path.isdir(path):
                os.makedirs(path)
                print 'Creating new folder:', path
        print

    if 'catalog' in purpose:

    #    print 'Nsplit:', Nsplit
    #    print 'blindcat:', blindcat
    #    print 'binnum:', binnum
    #    print 'Nobsbins:', Nobsbins
        
        # Path to the output splits and results
        path_splits = '%s/splits_%s'%(path_catalogs, purpose)
        path_results = '%s/results_%s'%(path_catalogs, purpose)

        if (Nsplit==0) and (blindcat==blindcats[0]) and (binnum == 1):

            for path in [path_splits, path_results]:
                if not os.path.isdir(path):
                    os.makedirs(path)


    # Name of the Rbins
    if os.path.isfile(path_Rbins): # from a file
        name_Rbins = path_Rbins.split('.')[0]
        name_Rbins = name_Rbins.split('/')[-1]
        name_Rbins = 'Rbins~%s%s'%(name_Rbins, Runit)
    else:
        name_Rbins = path_Rbins.replace(',', '~')
        name_Rbins = 'Rbins%s%s'%(name_Rbins, Runit)


    # Determining Ncat, the number of existing random catalogs
    splitslist = [] # This list will contain all created random splits

    if ('random' in purpose):

        # Defining the name of the output files
        filename_var = define_filename_var(purpose.replace('bootstrap', \
                                                           'catalog'), \
                                           centering, binname, binnum, \
                                           Nobsbins, lens_selection, \
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

            splitfiles = glob.glob(outname)
            splitslist = np.append(splitslist, splitfiles)
            
            if len(splitfiles) == 0:
                break

        print outname
        

    else:
        Ncat = 1

    return Nsplit, Nsplits, centering, lensid_file, lens_binning, binnum, \
    lens_selection, lens_weights, binname, Nobsbins, src_selection, \
    cat_version, path_Rbins, name_Rbins, Runit, path_output, path_splits, \
    path_results, purpose, O_matter, O_lambda, Ok, h, \
    filename_addition, Ncat, splitslist, blindcats, blindcat, \
    blindcatnum, path_kidscats, path_gamacat


# Defining the lensID lens selection/binning
def define_lensid_selection(lensid_file, lens_selection, lens_binning, binname):
    
    IDname = 'ID'
    
    lensid_files = lensid_file.split(',')

    if len(lensid_files) == 1: # If there is only one lensID bin -> selection
        lensids = np.loadtxt(lensid_files[0])
        lens_selection[IDname] = ['self', lensids]
    else: # If there are multiple lensID bins -> binning
        binname = IDname
        lens_binning = dict()
        for i, f in enumerate(lensid_files):
            lensids = np.loadtxt(f)
            lens_binning['%sbin%i' %(binname, i)] = ['self', lensids]

    return lens_selection, lens_binning, binname


# Define the part of the filename and plottitle
# that contains the lens/source selections
def define_filename_sel(filename_var, var_print, plottitle, selection):

    selnames = np.sort(selection.keys())
    for selname in selnames:
        sellims = (selection[selname])[1]

        if 'ID' in selname:
            filename_var = '%s_%ss%g'%(filename_var, selname, len(sellims))
            var_print = '%s #%ss = %g,'%(var_print, selname, len(sellims))
            plottitle = '%s $\#$ %ss = %g,'%(plottitle, selname, len(sellims))
        else:
            if len(sellims) == 1:
                filename_var = '%s_%s%g'%(filename_var, selname, sellims[0])
                var_print = '%s %s = %g,'%(var_print, selname, sellims[0])
                plottitle = '%s %s = %g,'%(plottitle, selname, sellims[0])
            else:
                filename_var = '%s_%s%g~%g'%(filename_var, selname, \
                                             sellims[0], sellims[1])
                var_print = '%s %s-limit: %g - %g,'%(var_print, selname, \
                                                     sellims[0], sellims[1])
                plottitle = '%s %g $\leq$ %s $\leq$ %g,'%(plottitle, \
                                                          sellims[0], selname, \
                                                          sellims[1])
        
    return filename_var, var_print, plottitle

    
# Defining the part of the filename that contains the chosen variables
def define_filename_var(purpose, centering, binname, binnum, Nobsbins, \
                        lens_selection, src_selection, lens_weights, \
                        name_Rbins, O_matter, O_lambda, Ok, h):
    
    # Define the list of variables for the output filename

    filename_var = ''
    var_print = ''

    if 'catalog' in purpose:
        if centering == 'Cen':
            filename_var = 'Cen'
        
        var_print = 'Galaxy catalogue,'

    else: # Binnning information of the groups

        # Lens binning
        if 'No' not in binname: # If there is binning
            filename_var = '%s_%sbin%sof%i'%(filename_var, binname, \
                                             binnum, Nobsbins)
            var_print = '%s %i %s-bins,'%(var_print, Nobsbins, binname)
        
        # Lens selection
        filename_var, var_print, x = define_filename_sel(filename_var, \
                                                         var_print, '', \
                                                         lens_selection)
    
        weightname = lens_weights.keys()[0]
        if weightname != 'None':
            filename_var = '%s_lw~%s'%(filename_var, weightname)
            var_print = '%s Lens weights: %s,'%(var_print, weightname)
    
    # Source selection
    filename_var, var_print, x = define_filename_sel(filename_var, var_print,\
                                                     '', src_selection)
    
    filename_var = '%s_%s_Om%g_Ol%g_Ok%g_h%g'%(filename_var, name_Rbins, \
                                               O_matter, O_lambda, Ok, h)
    cosmo_print = ('    %s, Omatter=%g, Olambda=%g, Ok=%g, h=%g'%(name_Rbins, \
                                                    O_matter, \
                                                    O_lambda, Ok, \
                                                    h)).replace('~', '-')
    
    # Replace points with p and minus with m
    filename_var = filename_var.replace('.', 'p')
    filename_var = filename_var.replace('-', 'm')
    filename_var = filename_var.replace('~', '-')
    filename_var = filename_var.split('_', 1)[1]

    if 'covariance' not in purpose:
        print 'Chosen %s-configuration: '%purpose
        print var_print
        print cosmo_print
        print

    return filename_var


def define_filename_splits(path_splits, purpose, filename_var, \
                           Nsplit, Nsplits, filename_addition, blindcat):

    # Defining the names of the shear/random catalog
    if 'covariance' in purpose:
        splitname = '%s/%s_%s%s_%s.fits'%(path_splits, purpose, filename_var, \
                                          filename_addition, Nsplit)
                                        # Here Nsplit = kidscatname
    if 'bootstrap' in purpose:
        splitname = '%s/%s_%s%s_%s.fits'%(path_splits, purpose, filename_var, \
                                          filename_addition, blindcat)
    if 'catalog' in purpose:
        splitname = '%s/%s_%s%s_split%iof%i.fits'%(path_splits, purpose, \
                                                   filename_var, \
                                                   filename_addition, \
                                                   Nsplit, Nsplits)

    return splitname


def define_filename_results(path_results, purpose, filename_var, \
                            filename_addition, Nsplit, blindcat):
    # Paths to the resulting files

    if 'catalogs' in path_results:
        resultname = '%s/%s_%s%s.fits'%(path_results, purpose, \
                                        filename_var, filename_addition)
    else:
        resultname = '%s/%s_%s%s_%s.txt'%(path_results, purpose, filename_var, \
                                          filename_addition, blindcat)

    return resultname


# Importing all GAMA and KiDS data, and
# information on radial bins and lens-field matching.
def import_data(path_Rbins, Runit, path_gamacat, path_kidscats, centering, \
                purpose, Ncat, O_matter, O_lambda, Ok, h, \
                lens_weights, filename_addition, cat_version):

    # Import R-range
    Rmin, Rmax, Rbins, Rcenters, \
        nRbins, Rconst = define_Rbins(path_Rbins, Runit)
    
    # Import GAMA catalogue
    gamacat, galIDlist, galRAlist, galDEClist, galweightlist, galZlist, \
    Dcllist, Dallist = import_gamacat(path_gamacat, centering, purpose, Ncat, \
    O_matter, O_lambda, Ok, h, Runit, lens_weights)
    
    # Determine the coordinates of the KiDS catalogues
    kidscoord, kidscat_end = run_kidscoord(path_kidscats, cat_version)
    
    # Match the KiDS field and GAMA galaxy coordinates
    catmatch, kidscats, galIDs_infield = run_catmatch(kidscoord, galIDlist, \
                                                      galRAlist, galDEClist, \
                                                      Dallist, Rmax, purpose, \
                                                      filename_addition, \
                                                      cat_version)
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
            print 'Observable bin file does not exist:', path_Rbins
            exit()
    
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
        print '*** Unit of radial bins not recognized! ***'
        quit()
        
    [Rmin, Rmax, Rbins] = [r*Rconst for r in [Rmin, Rmax, Rbins]]

    """
    print 'path_Rbins', path_Rbins
    print 'Using: %i radial bins between %.1f and %.1f'%(nRbins, Rmin, Rmax)
    print 'Rmin', Rmin
    print 'Rmax', Rmax
    print 'Rbins', Rbins
    print 'Rcenters', Rcenters
    print 'nRbins', nRbins
    """

    return Rmin, Rmax, Rbins, Rcenters, nRbins, Rconst


# Load the properties (RA, DEC, Z -> dist) of the galaxies in the GAMA catalogue
def import_gamacat(path_gamacat, centering, purpose, Ncat, \
                    O_matter, O_lambda, Ok, h, Runit, lens_weights):

    randomcatname = 'gen_ran_out.randoms.fits'
    directory = os.path.dirname(os.path.realpath(path_gamacat))
    randomcatname = directory + '/' + randomcatname
    
    # Importing the GAMA catalogues
    print 'Importing GAMA catalogue:', path_gamacat

    gamacat = pyfits.open(path_gamacat, ignore_missing_end=True)[1].data


    galIDlist = gamacat['ID'] # IDs of all galaxies
    
    if centering == 'Cen':
        galRAlist = gamacat['CenRA'] # Central RA of the galaxy (in degrees)
        galDEClist = gamacat['CenDEC'] # Central DEC of the galaxy (in degrees)
        galZlist = gamacat['Zfof'] # Z of the group
    else:
        galRAlist = gamacat['RA'] # Central RA of the galaxy (in degrees)
        galDEClist = gamacat['DEC'] # Central DEC of the galaxy (in degrees)

    if 'random' in purpose:
        # Determine RA and DEC for the random/star catalogs
        # The first item that will be chosen from the catalog
        Ncatmin = Ncat * len(galIDlist)
        # The last item that will be chosen from the catalog
        Ncatmax = (Ncat+1) * len(galIDlist)
        try:
            randomcat = pyfits.open(randomcatname)[1].data
        except:
            print 'Could not import random catalogue: ', randomcatname
            quit()

        galRAlist = randomcat['ra'][Ncatmin : Ncatmax]
        galDEClist = randomcat['dec'][Ncatmin : Ncatmax]

    #Defining the lens weights
    weightname = lens_weights.keys()[0]
    if 'No' not in weightname:
        galweightlist = pyfits.open(lens_weights.values()[0])[1].data[weightname]
    else:
        galweightlist = np.ones(len(galIDlist))

    # Defining the comoving and angular distance to the galaxy center
    if 'pc' in Runit: # Rbins in a multiple of pc
        galZlist = gamacat['Z'] # Central Z of the galaxy
        Dcllist = np.array([distance.comoving(z, O_matter, O_lambda, h) \
                            for z in galZlist])
        # Distance in pc/h, where h is the dimensionless Hubble constant

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


# Create a dictionary of KiDS fields that contain the corresponding galaxies.
def run_catmatch(kidscoord, galIDlist, galRAlist, galDEClist, Dallist, Rmax, \
                 purpose, filename_addition, cat_version):

    Rfield = np.radians(np.sqrt(2)/2) * Dallist
    if 'oldcatmatch' in filename_addition:
        print "*** Using old lens-field matching procedure! ***"
    else:
        Rmax = Rmax + Rfield
        print "*** Using new lens-field matching procedure ***"
        print "(for 'early science' mode, put"\
                " 'oldcatmatch' in 'ESD_output_filename')"

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
                lensmask = np.logical_not(np.in1d(galIDs, catmatch[kidscat]))
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
    print
    
    return catmatch, kidscats, galIDs_infield


def split(seq, size): # Split up the list of KiDS fields for parallel processing

    newseq = []
    splitsize = len(seq)/size

    for i in xrange(size-1):
        newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    newseq.append(seq[int(round((size-1)*splitsize)):len(seq)])

    return newseq


def import_spec_cat(path_kidscats, kidscatname, kidscat_end, \
                    src_selection, cat_version):
    
    pattern = '*specweight.cat'

    files = os.listdir(os.path.dirname('%s'%(path_kidscats)))
    
    filename = str(fnmatch.filter(files, pattern)[0])
    
    spec_cat_file = os.path.dirname('%s'%(path_kidscats))+'/%s'%(filename)
    spec_cat = pyfits.open(spec_cat_file, memmap=True)[1].data

    Z_B = spec_cat['z_spec']
    spec_weight = spec_cat['spec_weight']
    manmask = spec_cat['MASK']
    
    srcmask = (manmask==0)
    
    # We apply any other cuts specified by the user for Z_B
    srclims = src_selection['Z_B'][1]
    if len(srclims) == 1:
        srcmask *= (spec_cat['Z_B'] == binlims[0])
    else:
        srcmask *= (srclims[0] <= spec_cat['Z_B']) &\
            (spec_cat['Z_B'] < srclims[1])

    return Z_B[srcmask], spec_weight[srcmask]


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
    
    # List of the ID's of all sources in the KiDS catalogue
    srcNr = kidscat['SeqNr']
    # List of the RA's of all sources in the KiDS catalogue
    srcRA = kidscat['ALPHA_J2000']
    # List of the DEC's of all sources in the KiDS catalogue
    srcDEC = kidscat['DELTA_J2000']
    

    if cat_version == 3:
        w = np.transpose(np.array([kidscat['weight_A'], kidscat['weight_B'], \
                                   kidscat['weight_C'], kidscat['weight_C']]))
        Z_B = kidscat['Z_B']
        SN = kidscat['model_SNratio']
        manmask = kidscat['MASK']
        tile = kidscat['THELI_NAME']
        
    elif cat_version == 2:
        Z_B = kidscat['PZ_full'] # Full P(z) probability function
        w = np.transpose(np.array([kidscat['weight'], kidscat['weight'], \
                                   kidscat['weight'], kidscat['weight']]))
                                   
        # The Signal to Noise of the sources (needed for bias)
        SN = kidscat['SNratio']
        # The manual masking of bad sources (0=good, 1=bad)
        manmask = kidscat['MAN_MASK']
        tile = np.zeros(srcNr.size, dtype=np.float64)
    
    try:
        srcm = kidscat['m_cor'] # The multiplicative bias m
    except:
        srcm = np.zeros(srcNr.size, dtype=np.float64)

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
        srcmask = (w.T[0]>0.0)&(SN > 0.0)&(manmask==0)

    # We apply any other cuts specified by the user
    for param in src_selection.keys():
        srclims = src_selection[param][1]
        if len(srclims) == 1:
            srcmask *= (kidscat[param] == binlims[0])

        else:
            srcmask *= (srclims[0] <= kidscat[param]) & \
                        (kidscat[param] < srclims[1])


    srcNr = srcNr[srcmask]
    srcRA = srcRA[srcmask]
    srcDEC = srcDEC[srcmask]
    w = w[srcmask]
    Z_B = Z_B[srcmask]
    srcm = srcm[srcmask]
    e1 = e1[srcmask]
    e2 = e2[srcmask]
    tile = tile[srcmask]

    return srcNr, srcRA, srcDEC, w, Z_B, e1, e2, srcm, tile


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

    print 'Variance (A,B,C,D):', variance
    print 'Sigma (A,B,C,D):', variance**0.5

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
def define_obsbins(binnum, lens_binning, lenssel_binning, gamacat):

    # Check how the binning is given
    binname = lens_binning.keys()[0]
    if 'No' not in binname:
        
        if 'ID' in binname:
            Nobsbins = len(lens_binning.keys())
            if len(lenssel_binning) > 0:
                print 'Lens binning: Lenses divided in %i lens-ID bins'%(Nobsbins)

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
                    obslist = define_obslist(binname, gamacat)
                else:
                    print 'Using %s from %s'%(binname, obsfile)
                    obscat = pyfits.open(obsfile)[1].data
                    obslist = obscat[binname]

                
                print
                print 'Lens binning: Lenses divided in %i %s-bins'%(Nobsbins, \
                                                                    binname)
                print '%s Min:          Max:          Mean:'%binname
                for b in xrange(Nobsbins):
                    lenssel = lenssel_binning & (obsbins[b] <= obslist) \
                                & (obslist < obsbins[b+1])
                    print '%g    %g    %g'%(obsbins[b], obsbins[b+1], \
                                            np.mean(obslist[lenssel]))
            
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
def define_obslist(obsname, gamacat):

    obslist = gamacat[obsname]

    if 'AngSep' in obsname:
        print 'Applying cosmology correction "AngSep"'

        Dclgama = np.array([distance.comoving(z, 0.25, 0.75, 1.) \
                            for z in galZlist])
        corr_list = Dcllist/Dclgama
        obslist = obslist * corr_list

    if 'logmstar' in obsname:
        print 'Applying fluxscale correction to "logmstar"'

        # Fluxscale, needed for stellar mass correction
        fluxscalelist = gamacat['fluxscale']
        corr_list = np.log10(fluxscalelist)# - 2*np.log10(h/0.7)
        obslist = obslist + corr_list

    return obslist


# Masking the lenses according to the appropriate
# lens selection and the current KiDS field
def define_lenssel(gamacat, centering, lens_selection, lens_binning,
                   binname, binnum, binmin, binmax):

    lenssel = np.ones(len(gamacat['ID']), dtype=bool)
    # introduced by hand (CS) for the case when I provide a lensID_file:
    #binname = 'No'
    
    # Add the mask for each chosen lens parameter
    for param in lens_selection.keys():
        binlims = lens_selection[param][1]
        obsfile = lens_selection[param][0]
        # Importing the binning file
        if obsfile == 'self':
            obslist = define_obslist(param, gamacat)
        else:
            print 'Using %s from %s'%(param, obsfile)
            bincat = pyfits.open(obsfile)[1].data
            obslist = bincat[param]
        
        if 'ID' in param:
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
            obslist = define_obslist(binname, gamacat)
        else:
            print 'Using %s from %s'%(binname, obsfile)
            bincat = pyfits.open(obsfile)[1].data
            obslist = bincat[binname]
        
        if 'ID' in binname:
            lensids = lens_binning['%s%i'%(binname, binnum)]
            lenssel *= np.in1d(obslist, lensids)
        else:
            lenssel *=  (binmin <= obslist) & (obslist < binmax)

    return lenssel


# Calculate Sigma_crit (=1/k) and the weight mask for every lens-source pair
def calc_Sigmacrit(Dcls, Dals, Dcsbins, srcPZ, cat_version):

    # Calculate the values of Dls/Ds for all lens/source-redshift-bin pair
    Dcls, Dcsbins = np.meshgrid(Dcls, Dcsbins)
    DlsoDs = (Dcsbins-Dcls)/Dcsbins
    Dcls = [] # Empty unused lists
    Dcsbins = []

    # Mask all values with Dcl=0 (Dls/Ds=1) and Dcl>Dcsbin (Dls/Ds<0)
    #DlsoDsmask = np.logical_not((0.<DlsoDs) & (DlsoDs<1.))
    #DlsoDs = np.ma.filled(np.ma.array(DlsoDs, mask=DlsoDsmask, fill_value=0))
    DlsoDs[np.logical_not((0.< DlsoDs) & (DlsoDs < 1.))] = 0.0
    
    if cat_version == 3:
        cond = np.array(np.where(DlsoDs == 0.0))
        
        cond = cond + np.array((np.ones(cond[0].size, dtype=np.int32), \
                                np.zeros(cond[1].size, dtype=np.int32)))
        cond2 = cond + np.array((np.ones(cond[0].size, dtype=np.int32), \
                                 np.zeros(cond[1].size, dtype=np.int32)))
        
        cond[cond >= DlsoDs[0].size-1] = DlsoDs[0].size-1
        cond2[cond2 >= DlsoDs[0].size-1] = DlsoDs[0].size-1
        
        cond[cond >= DlsoDs[1].size-1] = DlsoDs[1].size-1
        cond2[cond2 >= DlsoDs[1].size-1] = DlsoDs[1].size-1
        
        DlsoDs[(cond[0], cond[1])] = 0.0
        DlsoDs[(cond2[0], cond2[1])] = 0.0
    
    DlsoDsmask = [] # Empty unused lists

    # Matrix multiplication that sums over P(z),
    # to calculate <Dls/Ds> for each lens-source pair
    DlsoDs = np.dot(srcPZ, DlsoDs).T

    # Calculate the values of k (=1/Sigmacrit)
    Dals = np.reshape(Dals,[len(Dals),1])
    k = 1 / ((c.value**2)/(4*np.pi*G.value) * 1/(Dals*DlsoDs)) # k = 1/Sigmacrit

    DlsoDs = [] # Empty unused lists
    Dals = []

    # Create the mask that removes all sources with k not between 0 and infinity
    kmask = np.logical_not((0. < k) & (k < inf))

    gc.collect()

    return k, kmask


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
    incosphi = ((-np.cos(np.radians(galDEC))*(np.radians(galRA-srcRA)))**2-\
                (np.radians(galDEC-srcDEC))**2)/(theta)**2
    insinphi = 2*(-np.cos(np.radians(galDEC))*\
                  (np.radians(galRA-srcRA)))*np.radians(galDEC-srcDEC)/(theta)**2

    incosphi = incosphi.T
    insinphi = insinphi.T

    return srcR, incosphi, insinphi


# For each radial bin of each lens we calculate the output shears and weights
def calc_shear_output(incosphilist, insinphilist, e1, e2, \
                      Rmask, klist, wlist, Nsrclist, srcm):
    
    wlist = wlist.T
    klist_t = np.array([klist, klist, klist, klist]).T

    # Calculating the needed errors
    wk2list = wlist*klist_t**2

    w_tot = np.sum(wlist, 0)
    w2_tot = np.sum(wlist**2, 0)

    k_tot = np.sum(klist, 1)
    k2_tot = np.sum(klist**2, 1)

    wk2_tot = np.sum(wk2list, 0)
    w2k4_tot = np.sum(wk2list**2, 0)
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
    Cs_tot = sum(-incosphilist*klist*galweights, 0)
    Ss_tot = sum(-insinphilist*klist*galweights, 0)
    Zs_tot = sum(klist**2*galweights, 0)

    return Cs_tot, Ss_tot, Zs_tot


# Write the shear or covariance catalog to a fits file
def write_catalog(filename, galIDlist, Rbins, Rcenters, nRbins, Rconst, \
                  output, outputnames, variance, purpose, e1, e2, w, srcm):

    fitscols = []

    Rmin = Rbins[0:nRbins]/Rconst
    Rmax = Rbins[1:nRbins+1]/Rconst

    # Adding the radial bins
    if 'bootstrap' in purpose:
        fitscols.append(pyfits.Column(name = 'Bootstrap', format='20A', \
                                      array = galIDlist))
    else:
        fitscols.append(pyfits.Column(name = 'ID', format='J', \
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
def write_stack(filename, Rcenters, Runit, ESDt_tot, ESDx_tot, error_tot, \
                bias_tot, h, variance, blindcatnum, \
                galIDs_matched, galIDs_matched_infield):

    # Choosing the appropriate covariance value
    variance = variance[blindcatnum]

    if 'pc' in Runit:
        filehead = '# Radius(%s)	ESD_t(h%g*M_sun/pc^2)   '\
        'ESD_x(h%g*M_sun/pc^2)	error(h%g*M_sun/pc^2)^2	bias(1+K)'\
        'variance(e_s)'%(Runit, h*100, h*100, h*100)
    else:
        filehead = '# Radius(%s)	gamma_t gamma_x	error   '\
        'bias(1+K)	variance(e_s)'%(Runit)

    with open(filename, 'w') as file:
        print >>file, filehead

    with open(filename, 'a') as file:
        for R in xrange(len(Rcenters)):

            if not (0 < error_tot[R] and error_tot[R]<inf):
                ESDt_tot[R] = int(-999)
                ESDx_tot[R] = int(-999)
                error_tot[R] = int(-999)
                bias_tot[R] = int(-999)

            print >>file, '%.12g	%.12g	%.12g	%.12g'\
                '%.12g	%.12g'%(Rcenters[R], \
                ESDt_tot[R], ESDx_tot[R], error_tot[R], bias_tot[R], variance)

    print 'Written: ESD profile data:', filename


    if len(galIDs_matched)>0 and blindcatnum == 3:
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

        print "Written: List of all stacked lens ID's"\
                " that contribute to the signal:", galIDsname
        print "Written: List of stacked lens ID's"\
                " with their center within a KiDS field:", kidsgalIDsname

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


        plt.ylim(1e-1,1e3)

        if plotstyle == 'log':
            plt.errorbar(data_x, data_y, yerr=[errorl,errorh], ls='', \
                         marker='o', label=plotlabel)
            plt.ylabel(r'%s'%ylabel,fontsize=15)

        if plotstyle == 'errorlog':
            plt.plot(data_x, errorh, marker='o', label=plotlabel)
            plt.ylabel(r'Error(%s)'%ylabel,fontsize=15)

    plt.xlim(1e1,1e4)
    plt.xscale('log')
    
    plt.xlabel(r'%s'%xlabel,fontsize=15)

    plt.legend(loc='upper right',ncol=1, prop={'size':12})

    return


# Writing the ESD profile plot
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


# Plotting the analytical or bootstrap covariance matrix
def plot_covariance_matrix(filename, plottitle1, plottitle2, plotstyle, \
                           binname, lens_binning, Rbins, Runit, h):

    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib import gridspec
    from matplotlib import rc, rcParams

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
                mappable = ax_sub.pcolormesh(Rbins, Rbins, \
                                             covariance[N1,N2,:,:], \
                                             vmin=-1e7, vmax=1e7)
            if plotstyle == 'covlog':
                mappable = ax_sub.pcolormesh(Rbins, Rbins, \
                                             abs(covariance[N1,N2,:,:]), \
                                             norm=LogNorm(vmin=1e-7, vmax=1e7))
            if plotstyle == 'corlin':
                mappable = ax_sub.pcolormesh(Rbins, Rbins, \
                                             correlation[N1,N2,:,:], \
                                             vmin=-1, vmax=1)
            if plotstyle == 'corlog':
                mappable = ax_sub.pcolormesh(Rbins, Rbins, \
                                             abs(correlation)[N1,N2,:,:], \
                                             norm=LogNorm(vmin=1e-5, vmax=1e0))

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
                        ax_sub.set_xlabel(r'%s = %.3g - %.3g'%(binname, \
                                                               obsbins[N2], \
                                                               obsbins[N2+1]), \
                                          fontsize=12)
                    else:
                        ax_sub.set_xlabel(r'%.3g - %.3g'%(obsbins[N2], \
                                                          obsbins[N2+1]), \
                                          fontsize=12)

                if N2 == Nobsbins - 1:
                    ax_sub.yaxis.set_label_position('right')
                    if N1 == 0:
                        ax_sub.set_ylabel(r'%s = %.3g - %.3g'%(binname, \
                                                               obsbins[N1], \
                                                               obsbins[N1+1]), \
                                          fontsize=12)
                    else:
                        ax_sub.set_ylabel(r'%.3g - %.3g'%(obsbins[N1], \
                                                          obsbins[N1+1]), \
                                          fontsize=12)


    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off',\
                   left='off', right='off')

    if 'pc' in Runit:
        labelunit = '%s/h$_{%g}$)'%(Runit, h*100)
    else:
        labelunit = Runit
    
    ax.set_xlabel(r'Radial distance (%s)'%labelunit)
    ax.set_ylabel(r'Radial distance (%s)'%labelunit)


    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.yaxis.set_label_coords(-0.05, 0.5)

    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)

    plt.text(0.5, 1.08, plottitle1, horizontalalignment='center', fontsize=17, \
             transform = ax.transAxes)
    plt.text(0.5, 1.05, plottitle2, horizontalalignment='center', fontsize=17, \
             transform = ax.transAxes)
    plt.colorbar(mappable, cax=cax, orientation='vertical')

    file_ext = filename.split('.')[-1]
    plotname = filename.replace('.%s'%file_ext,'_%s.png'%plotstyle)

    plt.savefig(plotname,format='png')

    print 'Written: Covariance matrix plot:', plotname









