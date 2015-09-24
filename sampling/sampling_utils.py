import numpy
import os
from glob import glob
#from ConfigParser import SafeConfigParser

def load_datapoints(datafile, datacols, exclude_bins=None):
    if type(datafile) == str:
        R, esd = numpy.loadtxt(datafile, usecols=datacols[:2]).T
        # better in Mpc
        if R[-1] > 500:
            R /= 1000
        if len(datacols) == 3:
            oneplusk = numpy.loadtxt(datafile, usecols=[datacols[2]]).T
            esd /= oneplusk
        if exclude_bins is not None:
            R = numpy.array([R[i] for i in xrange(len(R))
                             if i not in exclude_bins])
            esd = numpy.array([esd[i] for i in xrange(len(R))
                               if i not in exclude_bins])
    else:
        R, esd = numpy.transpose([numpy.loadtxt(df, usecols=datacols[:2])
                                  for df in datafile], axes=(2,0,1))
        if len(datacols) == 3:
            oneplusk = numpy.array([numpy.loadtxt(df, usecols=[datacols[2]])
                              for df in datafile])
            esd /= oneplusk
        for i in xrange(len(R)):
            if R[i][-1] > 500:
                R[i] /= 1000
        if exclude_bins is not None:
            R = numpy.array([[Ri[j] for j in xrange(len(Ri))
                              if j not in exclude_bins] for Ri in R])
            esd = numpy.array([[esdi[j] for j in xrange(len(esdi))
                                if j not in exclude_bins] for esdi in esd])
    return R, esd

def load_covariance(covfile, covcols, Nobsbins, Nrbins, exclude_bins=None):
    cov = numpy.loadtxt(covfile, usecols=[covcols[0]])
    if len(covcols) == 2:
        cov /= numpy.loadtxt(covfile, usecols=[covcols[1]])
    # 4-d matrix
    if exclude_bins is None:
        nexcl = 0
    else:
        nexcl = len(exclude_bins)
    cov = cov.reshape((Nobsbins,Nobsbins,Nrbins+nexcl,Nrbins+nexcl))
    cov2d = cov.transpose(0,2,1,3)
    cov2d = cov2d.reshape((Nobsbins*(Nrbins+nexcl),
                           Nobsbins*(Nrbins+nexcl)))
    icov = numpy.linalg.inv(cov2d)
    #import pylab
    #from matplotlib import cm
    #fig, axes = pylab.subplots(figsize=(12,12), ncols=2, nrows=2)
    #axes[0][0].imshow(cov2d[::-1], interpolation='nearest', cmap=cm.jet,
                      #vmin=0, vmax=500)
    #axes[1][0].imshow(icov[::-1], interpolation='nearest', cmap=cm.jet,
                      #vmin=-1, vmax=3.5)
    # are there any bins excluded?
    if exclude_bins is not None:
        for b in exclude_bins[::-1]:
            cov = numpy.delete(cov, b, axis=3)
            cov = numpy.delete(cov, b, axis=2)
    # product of the determinants
    detC = numpy.array([numpy.linalg.det(cov[m][n])
                        for m in xrange(Nobsbins)
                        for n in xrange(Nobsbins)])
    prod_detC = detC[detC > 0].prod()
    # likelihood normalization
    likenorm = -(Nobsbins**2*numpy.log(2*numpy.pi) + numpy.log(prod_detC)) / 2
    # switch axes to have the diagonals aligned consistently to make it
    # a 2d array
    cov2d = cov.transpose(0,2,1,3)
    cov2d = cov2d.reshape((Nobsbins*Nrbins,Nobsbins*Nrbins))
    # errors are sqrt of the diagonal of the covariance matrix
    esd_err = numpy.sqrt(numpy.diag(cov2d)).reshape((Nobsbins,Nrbins))
    # invert
    icov = numpy.linalg.inv(cov2d)
    #axes[0][1].imshow(cov2d[::-1], interpolation='nearest', cmap=cm.jet,
                      #vmin=0, vmax=500)
    #axes[1][1].imshow(icov[::-1], interpolation='nearest', cmap=cm.jet,
                      #vmin=-1, vmax=3.5)
    #fig.tight_layout()
    #pylab.show()
    # reshape back into the desired shape (with the right axes order)
    icov = icov.reshape((Nobsbins,Nrbins,Nobsbins,Nrbins))
    icov = icov.transpose(2,0,3,1)
    return cov, icov, likenorm, esd_err, cov2d

def read_config(config_file, version='0.5.7',
                path_data='', path_covariance=''):
    valid_types = ('normal', 'lognormal', 'uniform', 'exp',
                   'fixed', 'read', 'function')
    params = []
    param_types = []
    prior_types = []
    val1 = []
    val2 = []
    val3 = []
    val4 = []
    starting = []
    meta_names = []
    fits_format = []
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == 'path_data':
            path_data = line[1]
        if line[0] == 'data':
            datafiles = line[1]
            datacols = [int(i) for i in line[2].split(',')]
            if len(datacols) not in (2,3):
                msg = 'datacols must have either two or three elements'
                raise ValueError(msg)
        if line[0] == 'path_covariance':
            path_covariance = line[1]
        elif line[0] == 'covariance':
            covfile = line[1]
            covcols = [int(i) for i in line[2].split(',')]
            if len(covcols) not in (1,2):
                msg = 'covcols must have either one or two elements'
                raise ValueError(msg)
        elif line[0] == 'sampler_output':
            output = line[1]
            if output[-5:].lower() != '.fits' and \
                output[-4:].lower() != '.fit':
                output += '.fits'
        # all of this will have to be made more flexible to allow for
        # non-emcee options
        elif line[0] == 'sampler':
            sampler = line[1]
        elif line[0] == 'nwalkers':
            nwalkers = int(line[1])
        elif line[0] == 'nsteps':
            nsteps = int(line[1])
        elif line[0] == 'nburn':
            nburn = int(line[1])
        elif line[0] == 'thin':
            thin = int(line[1])
        # this k is only needed for mis-centred groups in my implementation
        # so maybe it should go in the hm_utils?
        elif line[0] == 'k':
            k = int(line[1])
        elif line[0] == 'threads':
            threads = int(line[1])
        elif line[0] == 'sampler_type':
            sampler_type = line[1]
    if path_data:
        datafiles = os.path.join(path_data, datafiles)
    datafiles = sorted(glob(datafiles))
    if path_covariance:
        covfile = os.path.join(path_covariance, covfile)
    covfile = glob(covfile)
    if len(covfile) > 1:
        msg = 'ambiguous covariance filename'
        raise ValueError(msg)
    covfile = covfile[0]

    out = (datafiles, datacols, covfile, covcols, output,
           sampler, nwalkers, nsteps, nburn, thin, k, threads,
           sampler_type)
    return out

def read_function(function):
    print 'Reading function', function
    function_path = function.split('.')
    if len(function_path) < 2:
        msg = 'ERROR: the parent module(s) must be given with'
        msg += 'a function'
        print msg
        exit()
    else:
        module = __import__(function)
        for attr in function_path:
            func = getattr(func, attr)
    return func

def setup_integrand(R, k=7):
    """
    These are needed for integration and interpolation and should always
    be used. k=7 gives a precision better than 1% at all radii

    """
    if R.shape[0] == 1:
        Rrange = numpy.logspace(numpy.log10(0.99*R.min()),
                                numpy.log10(1.01*R.max()), 2**k)
        # this assumes that a value at R=0 will never be provided, which is
        # obviously true in real observations
        R = numpy.append(0, R)
        Rrange = numpy.append(0, Rrange)
    else:
        Rrange = [numpy.logspace(numpy.log10(0.99*Ri.min()),
                                 numpy.log10(1.01*Ri.max()), 2**k)
                  for Ri in R]
        R = [numpy.append(0, Ri) for Ri in R]
        Rrange = [numpy.append(0, Ri) for Ri in Rrange]
        R = numpy.array(R)
        Rrange = numpy.array(Rrange)
    return R, Rrange
