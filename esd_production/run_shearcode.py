#!/usr/bin/python

import shearcode
import numpy as np


def create_config(replacefile, find, replacelist):

    config_files = np.array([])

    f = open(replacefile,'r')
    filedata = f.read()
    f.close()

    for r in xrange(len(replacelist)):
        replace = '%s'%replacelist[r]
        newdata = filedata.replace(find, replace)

        fileout = '%s_%s'%(replacefile, replace)
        f = open(fileout, 'w')
        f.write(newdata)
        f.close()
    
        config_files = np.append(config_files, [fileout])
        
    return config_files



# Defining the config file(s)

replacefile = '/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/ggl_Ned.config'
replacelist = np.arange(6)+1

config_files = create_config(replacefile, '@', replacelist)

#config_files = ['/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/ggl_Ned.config']

#config_files = ['/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/ggl_margot.config']

# Edo's age bins
#config_files = ['/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/Edo_logagebins_%i.config'%(i+1) for i in xrange(3)]


print 'Running:', config_files

for config_file in config_files:
    shearcode.run_esd(config_file)
