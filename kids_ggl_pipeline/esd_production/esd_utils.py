from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from glob import glob
import numpy as np
#from ConfigParser import SafeConfigParser

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number


def read_config(config_file):

    blindcats = np.array(['A', 'B', 'C', 'D'])

    #parser = SafeConfigParser()
    #parser.read(config_file)
    ##print parser.
    #return parser
    lens_selection = {}
    src_selection = {}
    model_params = []
    sampler_params = []
    cat_version = 3
    kids_path = 'None'
    specz_file = None
    z_epsilon = 0.2
    n_boot = 1
    cross_cov = bool(1)
    com = bool(1)
    lens_photoz = bool(0)
    galSigma = 0.0
    lens_pz_redshift = bool(0)
    gama_path = 'None'
    filename = 'None'
    colnames = ['ID','RA','DEC','Z']
    kidscolnames = ['SeqNr', 'ALPHA_J2000', 'DELTA_J2000', 'Z_B', 'model_SNratio', 'MASK', 'THELI_NAME', 'weight', 'm_cor', 'e1', 'e2']
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        if line == []:
            continue

        ## General settings
        if line[0] == 'KiDS_version':
            #kidsversion = line[1]
            cat_version = np.int(line[1])
        elif line[0] == 'GAMA_version':
            gamaversion = line[1]
        elif line[0] == 'KiDS_path':
            kids_path = line[1]
        elif line[0] == 'specz_file':
            specz_file = line[1]
        elif line[0] == 'm_corr_file':
            m_corr_file = line[1]
        elif line[0] == 'GAMA_path':
            gama_path = line[1]
        elif line[0] == 'lens_catalog':
            lens_catalog = line[1]
        elif line[0] == 'lens_columns':
            colnames = line[1].split(',')
            assert len(colnames) in (3,4), \
                'Parameter `lens_columns` must specify the three or four' \
                ' required columns (depending on whether you want physical' \
                ' distances)'
        elif line[0] == 'kids_columns':
            kidscolnames = line[1].split(',')
            assert len(kidscolnames) == 11, \
                'Parameter `kids_columns` must specify the 11 required columns.'

        # Cosmology
        elif line[0] == 'Om':
            Om = float(line[1])
        elif line[0] == 'Ol':
            Ol = float(line[1])
        elif line[0] == 'Ok':
            Ok = float(line[1])
        elif line[0] == 'h':
            h = float(line[1])
        elif line[0] == 'z_epsilon':
            z_epsilon = float(line[1])

        ## ESD production

        # Algorithm
        elif line[0] == 'ESD_output_folder':
            folder = line[1]
        elif line[0] == 'ESD_output_filename':
            filename = line[1]
        elif line[0] == 'ESD_purpose':
            purpose = line[1]
        elif line[0] == 'Rbins':
            Rbins = line[1]
        elif line[0] == 'Runit':
            Runit = line[1]
        elif line[0] == 'ncores':
            ncores = int(line[1])
        elif line[0] == 'nbootstraps':
            n_boot = int(line[1])
        elif line[0] == 'cross_covariance':
            cross_cov = bool(int(line[1]))
        elif line[0] == 'comoving':
            com = bool(int(line[1]))
        elif line[0] == 'lens_pz_sigma':
            lens_photoz = bool(1)
            galSigma = float(line[1])
            lens_pz_redshift = bool(int(line[2]))

        # Lens selection
        elif line[0] == 'lensID_file':
            lensid_file = line[1]

        # Lens weights
        elif line[0] == 'lens_weights':
            weightname = line[1]
            if weightname == 'None':
                weightfile = ''
            else:
                weightfile = weightfile = line[2]
            lens_weights = {weightname: weightfile}

        elif line[0] == 'lens_binning':
            binname = line[1]
            if 'No' in binname:
                bins = np.array([])
                paramfile = 'self'
            else:
                paramfile = line[2]
                bins = np.array([float(i) for i in line[3].split(',')])
            lens_binning = {binname: [paramfile, bins]}

        elif line[0][:11] == 'lens_limits':
            param = line[1]
            paramfile = line[2]
            lims = np.array([float(i) for i in line[3].split(',')])
            lens_selection[param] = [paramfile, lims]

        if line[0] == 'kids_blinds':
            blindcats = line[1].split(',')

        # Source selection
        elif line[0][:10] == 'src_limits':
            param = line[1]
            bins = np.array([float(i) for i in line[2].split(',')])
            src_selection[param] = ['self', bins]

        """
        try: # If a KiDS version number was given
            print 'KiDS version:', kids_version
            # kids_path = ...
        except:
            pass

        try: # If a GAMA version number was given
            print 'GAMA version:', gama_version
            # gama_path = ...
        except:
            pass
        """

    try: # If a custom lens catalog was given
        if 'None' not in lens_catalog:
            gama_path = lens_catalog
            print('Lens catalog:', lens_catalog)
    except:
        pass

    out = (kids_path, gama_path, colnames, kidscolnames, specz_file, m_corr_file,
            Om, Ol, Ok, h, z_epsilon,
            folder, filename, purpose, Rbins, Runit, ncores,
            lensid_file, lens_weights, lens_binning, lens_selection,
            src_selection, cat_version, n_boot, cross_cov, com, lens_photoz, galSigma, lens_pz_redshift, blindcats)

    return out
