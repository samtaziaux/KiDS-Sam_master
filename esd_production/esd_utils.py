import os
from glob import glob
import numpy as np
#from ConfigParser import SafeConfigParser

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number


def read_config(config_file, version='0.5.7', Om=0.315, Ol=0.685, Ok=0, h=0.7,
                lensid_file=None, folder='./'):

    #parser = SafeConfigParser()
    #parser.read(config_file)
    ##print parser.
    #return parser
    lens_selection = {}
    src_selection = {}
    model_params = []
    sampler_params = []
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        if line == []:
            continue

        ## General settings
        if line[0] == 'KiDS_version':
            kidsversion = line[1]
        elif line[0] == 'GAMA_version':
            gamaversion = line[1]
        elif line[0] == 'KiDS_path':
            kids_path = line[1]
        elif line[0] == 'GAMA_path':
            gama_path = line[1]
        elif line[0] == 'Lens_catalog':
			lens_catalog = line[1]

        # Cosmology
        elif line[0] == 'Om':
            Om = float(line[1])
        elif line[0] == 'Ol':
            Ol = float(line[1])
        elif line[0] == 'Ok':
            Ok = float(line[1])
        elif line[0] == 'h':
            h = float(line[1])

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

        # Lens selection
        elif line[0] == 'lensID_file':
            lensid_file = line[1]
 
        # Lens weights
        elif line[0] == 'lens_weights':
            lens_weights = line[1]
            
        elif line[0] == 'group_centre':
            group_centre = line[1]

        elif line[0] == 'lens_ranks':
            try:
                ranks = np.array([int(i) for i in line[1].split(',')])
            except:
                ranks = line[1]
            try:
                if ranks[0] == ranks[1]:
                    ranks = np.array([ranks[0]])
            except:
                pass

        elif line[0] == 'lens_binning':
            binname = line[1]
            if 'No' in binname:
                bins = np.array([])
            else:
                bins = np.array([float(i) for i in line[2].split(',')])
            lens_binning = {binname: bins}
            
        elif line[0][:11] == 'lens_limits':
            param = line[1]
            bins = np.array([float(i) for i in line[2].split(',')])
            lens_selection[param] = bins

    #    elif line[0] == 'kids_blinds':
    #        blindcats = line[1].split(',')
        
        # Source selection
        elif line[0][:10] == 'src_limits':
            param = line[1]
            bins = np.array([float(i) for i in line[2].split(',')])
            src_selection[param] = bins
        
        blindcats = np.array(['A', 'B', 'C', 'D'])
    
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
        
        try: # If a custom lens catalog was given
            print 'Lens catalog:', lens_catalog
            gama_path = lens_catalog
        except:
            pass
        """
        
    out = (kids_path, gama_path,
            Om, Ol, Ok, h,
            folder, filename, purpose, Rbins, Runit, ncores,
            lensid_file, lens_weights, group_centre, ranks, lens_binning, lens_selection,
            src_selection, blindcats)

    return out
