
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

for i in xrange(2):
    filenamein = '/disks/shear10/brouwer_veersemeer/pipeline_testresults/output_logmstarbins_oldcatmatch/results_shearcatalog/shearcatalog_logmstarbin%iof2_loglwage9p5-inf_Z_B0p005-1p2_Rbins10-20-2000kpc_Om0p315_Ol0p685_Ok0_h1_oldcatmatch_lensIDs.txt'%(i+1)
    filenameout = filenamein.replace('catalog', 'covariance')

    outfiles = create_config(outfiles, filenamein, filenameout)
    
print outfiles
