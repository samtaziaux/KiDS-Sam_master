#!/usr/bin/python

import numpy as np
import sys

sys.path.append('esd_production')
import shearcode

def create_config(replacefile, searchlist, replacelist):
    
    config_files = np.array([])

    f = open(replacefile,'r')
    filedata = f.read()
    f.close()

    shape = np.shape(replacelist)
    if len(shape) == 1:
        Nr = 1
    else:
        Nr = shape[1]
    
    for r in xrange(Nr):

        newdata = filedata
        fileout = '%s_%s'%(replacefile, r+1)
        
        print
        print 'For %s:'%fileout
        
        for s in xrange(len(searchlist)):
            search = '%s'%searchlist[s]
            try:
                replace = '%s'%replacelist[s, r]
            except:
                replace = '%s'%replacelist[s]
            
            print '     - replace %s with %s'%(search, replace)
        
            newdata = newdata.replace(search, replace)
        
        f = open(fileout, 'w')
        f.write(newdata)
        f.close()
    
        config_files = np.append(config_files, [fileout])
        
    return config_files


# Defining the config file(s)

"""
replacefile = '/data2/brouwer/shearprofile/KiDS-GGL/brouwer/configs_margot/ggl_Ned.config'
findlist = ['@', '710']
replacelist = np.array([np.arange(6)+1, np.array([1724, 2634, 2201, 1577, 3379, 710])])
config_files = create_config(replacefile, findlist, replacelist)
"""
"""
replacefile = '/data2/brouwer/shearprofile/KiDS-GGL/brouwer/configs_margot/Edo_logagebins.config'
findlist = np.array(['agelimit'])
replacelist = np.array([np.array(['0,9.3', '9.3,9.5', '9.5,inf'])])
config_files = create_config(replacefile, findlist, replacelist)
"""
"""
replacefile = '/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/ggl_redshifts.config'
findlist = np.array(['Zlim'])
replacelist = np.array([np.array(['inf', '0.13', '0.2', '0.3'])])
#replacelist = np.array([np.array(['0.13'])])
config_files = create_config(replacefile, findlist, replacelist)
"""
"""
# Environments with logmstar-weights
replacefile = '/data2/brouwer/shearprofile/KiDS-GGL/configs_margot/ggl_margot_mstarweight.config'
findlist = np.array(['envbin'])
replacelist = np.array([np.arange(4)])
config_files = create_config(replacefile, findlist, replacelist)
"""

#config_files = ['brouwer/configs_margot/ggl_margot.config']

# Edo's age bins
#config_files = ['brouwer/configs_margot/Edo_logagebins_%i.config'%(i+1) for i in xrange(3)]
#config_files = ['brouwer/configs_margot/Edo_logagebins.config']

#config_files = ['brouwer/configs_margot/ggl_environments_all.config']
#config_files = ['brouwer/configs_margot/ggl_environments_cen+iso.config']
config_files = [ ['brouwer/configs_margot/ggl_environments_auto.config_%s_rank%s'%(e,r) for e in ['envS4', 'shuffenvR4']] for r in ['-999-inf', '-999-2'] ]

#config_files = ['brouwer/configs_margot/troughs.config']

#config_files = ['brouwer/configs_margot/ggl_environments_%s.config'%x for x in ['all', 'cen+iso', 'all_shuffled', 'cen+iso_shuffled']]

config_files = np.reshape(config_files, np.size(config_files))

print
print 'Running:', config_files


for config_file in config_files:
    shearcode.run_esd(config_file)
