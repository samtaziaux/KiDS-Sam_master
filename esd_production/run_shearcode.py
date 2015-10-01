#!/usr/bin/python

import shearcode
import numpy as np


def create_config(replacefile, searchlist, replacelist):

    config_files = np.array([])

    f = open(replacefile,'r')
    filedata = f.read()
    f.close()

    newdata = filedata

    for r in xrange(len(replacelist[0])):
        
        fileout = '%s_%s'%(replacefile, replacelist[0, r])
        
        print
        print 'For %s:'%fileout
        
        for s in xrange(len(searchlist)):
            search = '%s'%searchlist[s]
            replace = '%s'%replacelist[s, r]
            
            
            print '     - replace %s with %s'%(search, replace)
        
            newdata = newdata.replace(search, replace)

        
        f = open(fileout, 'w')
        f.write(newdata)
        f.close()
    
        config_files = np.append(config_files, [fileout])
        
    return config_files


# Defining the config file(s)

"""
replacefile = '/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/ggl_Ned.config'
findlist = ['@', '710']
replacelist = np.array([np.arange(6)+1, np.array([1724, 2634, 2201, 1577, 3379, 710])])
config_files = create_config(replacefile, findlist, replacelist)
"""

#config_files = ['/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/ggl_Ned.config']
config_files = ['/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/ggl_margot.config']

# Edo's age bins
#config_files = ['/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/Edo_logagebins_%i.config'%(i+1) for i in xrange(3)]


print 'Running:', config_files

for config_file in config_files:
    shearcode.run_esd(config_file)

