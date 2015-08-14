import os
from glob import glob
#from ConfigParser import SafeConfigParser

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
        ## General settings
        if line[0] == 'KiDS_version':
<<<<<<< HEAD
            kidsversion = line[1]
        elif line[0] == 'GAMA_version':
            gamaversion = line[1]
=======
            kids_version = line[1]
        elif line[0] == 'KiDS_path':
            kids_path = line[1]
        elif line[0] == 'GAMA_version':
            gama_version = line[1]
        elif line[0] == 'GAMA_path':
            gama_path = line[1]
>>>>>>> 23fa859e75f0d2b89876589d0a878919b09abbd9
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
        elif line[0] == 'output_folder':
            folder = line[1]
        elif line[0] == 'output_filename':
            filename = line[1]
        elif line[0] == 'purpose':
            purpose = line[1]
        elif line[0] == 'Rbins':
            Rbins = line[1]
        elif line[0] == 'ncores':
            ncores = int(line[1])
        # Lens selection
        elif line[0] == 'lensid_file':
            lensid_file = line[1]
        elif line[0] == 'group_centre':
            group_centre = line[1]
        elif line[0] == 'lens_binning':
            binparam = {line[1]: [int(i) for i in line[2].split(',')]}
        elif line[0][:10] == 'lens_param':
            param = line[1]
            bins = line[2].split(',')
            if len(bins) == 1:
                bins = bins[0]
            lens_selection[param] = bins
        elif line[0][:9] == 'src_param':
            param = line[1]
            bins = line[2].split(',')
            if len(bins) == 1:
                bins = bins[0]
            src_selection[param] = bins
<<<<<<< HEAD
    out = (purpose, folder, filename, kidsversion, gamaversion,
            ncores, Rbins, src_selection,
            lens_selection, group_centre, lensid_file, binparam,
            Om, Ol, Ok, h)

=======
    #if kids_version and not kids_path:
        #kids_path =
    #if gama_version and not gama_path:
        #gama_path =
    out = (kids_path, gama_path, Om, Ol, Ok, h,
           folder, filename, purpose, Rbins, ncores,
           lensid_file, group_centre,
           binparam, lens_selection, src_selection)
>>>>>>> 23fa859e75f0d2b89876589d0a878919b09abbd9
    return out


