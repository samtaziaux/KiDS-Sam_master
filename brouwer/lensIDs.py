
#!/usr/bin/python

import numpy as np



def create_config(outfiles, filenamein, filenameout):



    f = open(filenamein,'r')
    filedata = f.read()
    f.close()

    
    f = open(filenameout, 'w')
    f.write(filedata)
    f.close()
    
    outfiles = np.append(outfiles, [filenameout])
    
    return outfiles

outfiles = np.array([])

zlims = ['inf', '0p13', '0p2', '0p3']

for i in xrange(4):
    
    x = zlims[i]
    
    filenamein = '/disks/shear10/brouwer_veersemeer/pipeline_testresults/output_Nobins/results_shearcovariance/shearcovariance_Nfof5-inf_RankBCG1_Z0-%s_Z_B0p005-1p2_Rbins10-20-2000kpc_Om0p315_Ol0p685_Ok0_h1_KiDSlensIDs.txt'%(x)
    filenameout = filenamein.replace('catalog', 'covariance')

    outfiles = create_config(outfiles, filenamein, filenameout)
    
print outfiles
