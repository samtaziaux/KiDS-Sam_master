#!/usr/bin/env python
"""
Galaxy-galaxy lensing EMCEE wrapper

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import emcee
import numpy as np
import os
import sys
from astropy.io.fits import BinTableHDU, Column, Header, PrimaryHDU
from itertools import count
from matplotlib import cm, pylab
from numpy import all as npall, array, append, concatenate, dot, inf, isnan, \
                  isfinite, log, log10, outer, pi, sqrt, squeeze, transpose, \
                  where, zeros
from os import remove
from os.path import isfile
from scipy import stats
from six import string_types
from time import ctime, time
from multiprocessing import Pool

# Python 2/3 compatibility
if sys.version_info[0] == 2:
    from itertools import izip as zip
    range = xrange
    input = raw_input

# local
from . import priors, sampling_utils
from ..helpers import io, plotting
from .. import __version__
from ..halomodel.halo import initialize_beta_nl
# debugging
from ..helpers._debugging import import_icecream
ic = import_icecream()


def run(hm_options, options, args):

    function, function_cov, preamble, parameters, names, prior_types, \
        nparams, repeat, join, starting, output = \
            hm_options

    val1, val2, val3, val4 = parameters[1][parameters[0].index('parameters')]
    setup = parameters[1][parameters[0].index('setup')]

    obs_idx = parameters[0].index('observables')
    observables = parameters[1][obs_idx]
    init_mlf = observables.mlf # No idea why, but mlf instance doesn't get initilized to adopt R
    
    ingr_idx = parameters[0].index('ingredients')
    ingredients = parameters[1][ingr_idx]
    
    assert_output(setup, observables)

    print_opening_msg(args, options)

    # identify fixed and free parameters
    jfree = np.array([(p in priors.free_priors) for p in prior_types])
    ndim = len(val1[where(jfree)])
    assert len(starting) == ndim, \
        'ERROR: Not all starting points defined for free parameters.'

    if args.mock:
        assert function != None, \
            'ERROR: model function not defined.'
        assert function_cov != None, \
            'ERROR: covariance function not defined.'
        #mock_options = parameters[1][parameters[0].index('mock')]
        mock(
            args, function, function_cov, options, setup, #mock_options
            parameters, join, starting, jfree, repeat,
            nparams, observables, obs_idx)
        return

    if args.cov:
        assert function_cov != None, \
            'ERROR: covariance function not defined.'
        Ndatafiles = len(options['data'][0])
        assert Ndatafiles > 0, 'No data files found'
        R = io.load_data_esd_only(options, setup)
        # We need to make sure one can also read in the mock data to calculate the covariance for!
        cov_calc(
            args, function_cov, R, options, setup,
            parameters, join, starting, jfree, repeat,
            nparams, observables, obs_idx)
        return

    assert function != None, \
        'ERROR: model function not defined.'
    # load data files
    Ndatafiles = len(options['data'][0])
    assert Ndatafiles > 0, 'No data files found'
    # Rrange, angles are used in nfw_stack only
    R, esd, cov, Rrange, angles, Nobsbins, Nrbins = io.load_data(options, setup)
    cov, icov, likenorm, esd_err, cov2d, cor = cov
    # utility variables
    rng_obsbins = range(Nobsbins)

    observables._add_R(R)
    parameters[1][obs_idx] = observables

    meta_names = sampling_utils.initialize_metanames(options, output, len(esd))
    # initialize
    lnprior = np.zeros(ndim)

    names_extend = ['lnprior','chi2','lnlike']
    meta_names.extend(names_extend)
    # this needs to be slightly more flexible!
    formats = [np.dtype((np.float64, esd[i].shape)) for i in rng_obsbins] + \
              [np.float64 for i in rng_obsbins] + [np.float64, np.float64, np.float64]
    dtype = np.dtype({'names':meta_names, 'formats':formats})
    fail_value = np.zeros(1, dtype=dtype)
    for n in names_extend:
        fail_value[n] = 9999
    fail_value = list(fail_value[0])

    if args.demo or not options['resume']:
        print('Starting values =', starting)

    # if there is a preamble function we can run it now
    if preamble:
        parameters = preamble(parameters)

    # are we just running a demo?
    if args.demo:
        demo(args, function, R, esd, esd_err, cov, icov, cor, options, setup,
             prior_types, parameters, join, starting, jfree, repeat, nparams,
             names, lnprior, rng_obsbins, fail_value, Ndatafiles, meta_names,
             array, dot, inf, outer, pi, zip)
        return

    # write header file
    hdrfile = io.write_hdr(options, function, parameters, names, prior_types)

    # initialize sampler
    if options['resume']:
        backend = emcee.backends.HDFBackend(options['output'])
        if os.path.isfile(options['output']):
            print('Initial size: {0}'.format(backend.iteration))
        po = backend.get_chain(discard=backend.iteration - 1)[0]
    else:
        backend = emcee.backends.HDFBackend(options['output'])
        backend.reset(options['nwalkers'], ndim)
        # set up starting point for all walkers
        po = sample_ball(
            names, prior_types, jfree, starting, parameters,
            options['nwalkers'], ndim)

    if not os.path.isfile(options['output']) or not options['resume']:
        print('Running a new model. Good luck!\n')

    # burn-in
    if (options['nburn'] > 0) and not (options['resume']):
        with Pool(processes=options['threads']) as pool:
            sampler = emcee.EnsembleSampler(
                 options['nwalkers'], ndim, lnprob,
                 args=(R,esd,icov,function,names,prior_types[jfree],
                       parameters,nparams,join,jfree,repeat,lnprior,likenorm,
                       rng_obsbins,fail_value,array,dot,inf,zip,outer,pi,args),
                 pool=pool, blobs_dtype=dtype)
            pos = sampler.run_mcmc(po, options['nburn'], progress=True)
        sampler.reset()
        print('{1}: {0} Burn-in steps finished'.format(
            options['nburn'], ctime()))
    else:
        pos = po

    index = 0
    autocorr = np.empty(options['nsteps'])
    # This will be useful to testing convergence
    old_tau = np.inf
    with Pool(processes=options['threads']) as pool:
        sampler = emcee.EnsembleSampler(
             options['nwalkers'], ndim, lnprob,
             args=(R,esd,icov,function,names,prior_types[jfree],
                   parameters,nparams,join,jfree,repeat,lnprior,likenorm,
                   rng_obsbins,fail_value,array,dot,inf,zip,outer,pi,args),
             pool=pool, backend=backend, blobs_dtype=dtype)#, moves=[(emcee.moves.StretchMove(), 0.5), (emcee.moves.KDEMove(), 0.5)])
        #result = sampler.run_mcmc(pos, options['nsteps'], thin_by=options['thin'], progress=True, store=True)
        for sample in sampler.sample(pos, iterations=options['nsteps'],
                                     thin_by=options['thin'], progress=True, store=True):
            # Using the pattern below we can use this to update things in sampler iteratively every n steps
            # (provided by "update" for instance)
            if options['stop_when_converged']:
                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue
                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
                # Check convergence
                # should we offer more flexibility here?
                # n-times autocorrelation time as a measure of convergence
                converged = np.all(
                    tau * options['autocorr_factor'] < sampler.iteration)
                # and chains change less than 1%
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
            if ingredients['bnl']: # This is to iterativelly calculate the non-linear halo bias every n steps
                if sampler.iteration % options['update']:
                    continue
                # Recalculate the bnl
                walk = sampler.get_chain(discard=sampler.iteration - 1, flat=True)
                state = np.nanmedian(walk, axis=0)
                p = update_parameters(state, parameters, nparams, join, jfree, repeat)
                cosmo = p[1][p[0].index('parameters')][0]
                # Cosmology parameters need be set to a dictionary or something, this hardcoding is not optimal at all
                #s8 = cosmo[3], h = cosmo[2], om = cosmo[0],
                #ob = cosmo[1], ns = cosmo[4], w0 = cosmo[5]
                beta = initialize_beta_nl(cosmo[1], cosmo[0]-cosmo[1], 1-cosmo[0],
                        cosmo[3], cosmo[4], cosmo[5], cosmo[2], reset=True)

    io.finalize_hdr(sampler, hdrfile)
    print('Everything saved to {0}!'.format(options['output']))

    return


def assert_output(setup, observables):
    if setup['return'] in ('wp', 'esd_wp') and not observables.gg:
        raise ValueError(
            'If return=wp or return=esd_wp then you must toggle the' \
            ' clustering as an ingredient. Similarly, if return=esd' \
            ' or return=esd_wp then you must toggle the lensing' \
            ' as an ingredient as well.')
    if setup['return'] in ('esd', 'esd_wp') and not observables.gm:
        raise ValueError(
            'If return=wp or return=esd_wp then you must toggle the' \
            ' clustering as an ingredient. Similarly, if return=esd' \
            ' or return=esd_wp then you must toggle the lensing' \
            ' as an ingredient as well.')
    return



def demo(args, function, R, esd, esd_err, cov, icov, cor, options, setup,
         prior_types, parameters, join, starting, jfree, repeat, nparams,
         names, lnprior, rng_obsbins, fail_value, Ndatafiles, meta_names,
         array, dot, inf, outer, pi, zip, plot_ext='png'):
    to = time()
    lnlike, model = lnprob(
        starting, R, esd, icov, function, names, prior_types[jfree],
        parameters, nparams, join, jfree, repeat, lnprior, 0,
        rng_obsbins, fail_value, array, dot, inf, zip, outer, pi, args)
    print('\nDemo run took {0:.2e} seconds'.format(time()-to))
    chi2 = model[-2]
    if chi2 == fail_value[-2]:
        msg = 'Could not calculate model prediction. It is likely that one' \
              ' of the parameters is outside its allowed prior range.'
        raise ValueError(msg)
    #dof = esd.size - starting.size - 1
    esd_size = array([e.size for e in esd]).sum()
    dof = esd_size - starting.size - 1
    # print model values
    print()
    print('Model values:')
    print('-------------')
    im = 0
    ip = 0
    for i, name in enumerate(meta_names[:-3]):
        print('{0:<20s}  {1}'.format(name, model[im][ip]))
        ip += 1
        if ip == len(model[im]):
            ip = 0
            im += 1
    print()
    print(' ** chi2 = {0:.2f}/{1:d} **'.format(chi2, dof))
    print()

    # this for now. The idea is that setup['return'] should eventually
    # be a list with all the quantities that are being modelled.
    if isinstance(setup['return'], str):
        setup['return'] = [setup['return']]
    output = '{0}_demo_{1}.{2}'.format(
        '.'.join(options['output'].split('.')[:-1]),
                 '-'.join(setup['return']).replace('.', ''),
        plot_ext)
    # if necessary, create a demo folder within the output path
    path, output = os.path.split(output)
    if 'demo' not in os.path.split(path)[1]:
        path = os.path.join(path, 'demo')
    os.makedirs(path, exist_ok=True)
    output = os.path.join(path, output)
    # make plots
    plotting.signal(
        R, esd, esd_err, rng_obsbins, model=model[0], observable=setup['return'],
        output=output)
    if not args.no_demo_cov:
        output = output.replace(
            '.{0}'.format(plot_ext), '_cov.{0}'.format(plot_ext))
        plotting.covariance(R, cov, cor, output)

    return


def mock(args, function, function_cov, options, setup, parameters, join, starting, jfree,
         repeat, nparams, observables, obs_idx):
    """Generate mock observations given a set of cosmological and
    astrophysical parameters, as well as the observational setup

    steps:
    * call halo.model just like in lnprob
    * call covariance module, and pass mock_options to it (it
      should be possible to disable this though)
    """
    
    if observables.mlf:
        nbins = observables.nbins - observables.mlf.nbins
    else:
        nbins = observables.nbins
    
    R = np.array([np.logspace(setup['logR_min'], setup['logR_max'],
                     setup['logR_bins']) for n in range(nbins)], dtype=object)
    if observables.mlf:
        R = np.concatenate((R, np.array([np.logspace(sam[0], sam[-1], setup['logR_bins'])
                    for sam in observables.mlf.sampling], dtype=object)), axis=0)

    R, _ = sampling_utils.setup_integrand(R, options['precision'])
    
    observables._add_R(R)
    parameters[1][obs_idx] = observables
    
    p = update_parameters(starting, parameters, nparams, join, jfree, repeat)
    # run model!
    model = function(p, R)
    # apply random noise
    #
    # save
    if args.cov:
        print('Mock signal generated, calculating the covariance, sit tight.')
        cov = cov_calc(args, function_cov, R, options, setup, parameters, join, starting, jfree,
            repeat, nparams, observables, obs_idx)
    #else:
    if not args.cov:
        cov = None
    outputs = io.write_signal(R, model[0], options, setup, cov)
    print('Saved mock signal to:')
    print(*outputs, sep = "\n")
    return
    
    
def cov_calc(args, function, R, options, setup, parameters, join, starting, jfree,
            repeat, nparams, observables, obs_idx):
    
    """Generate  the covariance
    including astrophysical terms through the --cov cmd line option

    steps:
    * call covariance.covariance just like in halo.model in lnprob
    * saves covariance as 2D matrix to a text file as specified in the config file
    """
    
    observables._add_R(R)
    parameters[1][obs_idx] = observables
    
    to = time()
    p = update_parameters(starting, parameters, nparams, join, jfree, repeat)
    # run model!
    covariance = function(p, R)
    print('\nCovariance calculation took {0:.2e} seconds'.format(time()-to))
    
    if args.mock:
        return covariance
    # save
    output = parameters[1][parameters[0].index('covariance')]['output']
    
    np.savetxt(output, covariance, header='2D analytical covariance matrix', comments='# ')
    print('Saved covariance to {0}'.format(output))
    return


def lnprob(theta, R, esd, icov, function, names, prior_types,
           parameters, nparams, join, jfree, repeat, lnprior, likenorm,
           rng_obsbins, fail_value, array, dot, inf, zip, outer, pi, args):
    """
    Probability of a model given the data, i.e., log-likelihood of the data
    given a model, times the prior on the model parameters.

    Parameters
    ----------
        theta
            whatever *free* parameters are received by the model selected.
        R
            lens-source separations
        esd
            excess surface density at distances R
        icov
            inverse covariance
        function
            the model used to calculate the likelihood
        prior_types
            one value per parameter in *theta*, {'normal', 'uniform', 'fixed'}
        parameters
            list of length 4, containing all parameters of the models. In
            terms of the configuration file, these are [ingredients,
            observable, parameters, setup]; where `parameters` is another
            length-4 list
        jfree
            indices of the free values
        lnprior
            just a placeholder, should be an array with a length
            equal to theta, so that it doesn't have to be defined every
            time

    """
    lnprior_total = priors.calculate_lnprior(
        lnprior, theta, prior_types, parameters, jfree)
    ic(lnprior_total)
    if not isfinite(lnprior_total):
        if args.demo:
            return -inf, fail_value
        return (-inf, *fail_value)

    p = update_parameters(theta, parameters, nparams, join, jfree, repeat)
    # run model!
    model = function(p, R)
    # no covariance
    #chi2 = (((esd-model[0]) / esd_err) ** 2).sum()
    # full covariance included
    residuals = esd - model[0]
    chi2 = array([dot(residuals[m], dot(icov[m][n], residuals[n]))
                  for m in rng_obsbins for n in rng_obsbins]).sum()
    if not isfinite(chi2):
        if args.demo:
            return -inf, fail_value
        return (-inf, *fail_value)
    lnlike = -chi2/2. + likenorm
    if args.demo:
        model.extend([lnprior_total, chi2, lnlike])
        return lnlike + lnprior_total, model
    else:
        for i,m in enumerate(model[0]):
            model[0][i] = m.astype(np.float64)
        model.extend([[lnprior_total], [chi2], [lnlike]])
        flat = [m for inner_list in model for m in inner_list]
        post = lnlike + lnprior_total
        return (post, *flat)


def print_opening_msg(args, options):
    print('\nRunning KiDS-GGL pipeline version {0} - sampler\n'.format(
        __version__))
    if args.mock and args.cov:
        print(' ** Generating mock observations and covariance matrix **')
    elif args.cov:
        print(' ** Calculating covariance matrix only **')
    elif args.demo:
        print(' ** Running demo only **')
    elif args.mock:
        print(' ** Generating mock observations **')
    elif options['resume']:
        print('** Resuming sampler from last sample **')
    elif isfile(options['output']) and not args.force_overwrite \
            and not options['resume']:
        msg = 'Warning: output file {0} exists. Overwrite? [y/N] '.format(
            options['output'])
        answer = input(msg)
        if len(answer) == 0:
            sys.exit()
        if answer.lower() not in ('y', 'yes'):
            sys.exit()
    if not args.demo:
        print('\n{0}: Started\n'.format(ctime()))
    return


def sample_ball(names, prior_types, jfree, starting, parameters, nw, ndim):
    """Create a ball around the starting values by drawing random
    samples from the prior
    """
    val1, val2, val3, val4 = parameters[1][parameters[0].index('parameters')]
    v1free = val1[where(jfree)]
    v2free = val2[where(jfree)]
    v3free = val3[where(jfree)]
    v4free = val4[where(jfree)]
    names_free = names[where(jfree)]
    prior_free = prior_types[where(jfree)]

    ball = np.zeros((nw, ndim))
    for n, p in enumerate(prior_free):
        ball[:,n] = priors.draw(
            p, (v1free[n],v2free[n]), (v3free[n],v4free[n]), size=nw)
    #ball = starting + 5e-1 * np.random.randn(nw, ndim)
    return ball


def update_parameters(theta, parameters, nparams, join, jfree, repeat):
    # update parameters
    v1 = parameters[1][parameters[0].index('parameters')][0].copy()
    v1[where(jfree)] = theta
    # joined parameters
    v1_list = list(v1)
    for j in join[::-1]:
        try:
            v1_list[j[0]] = np.array(v1[j], dtype=float)
        except ValueError:
            v1_list[j[0]] = v1[j]
    # repeat parameters
    for i, j in enumerate(repeat):
        if j != -1:
            v1_list[i] = v1_list[j]
    # do this all the way at the end to preserve indices
    for j in join:
        # remove parameters that have just been joined
        for i in j[1:][::-1]:
            v1_list.pop(i)
    # convert histograms (especially joined histograms) into np arrays
    for i, v in enumerate(v1_list):
        if not isinstance(v, string_types) and hasattr(v, '__iter__'):
            if hasattr(v[0], '__iter__'):
                v1_list[i] = [np.array(vi) for vi in v]
            v1_list[i] = np.array(v1_list[i], dtype=float)
    # join into sections and subsections as per the config file
    # by doing this I'm losing Rrange and angles, but I suppose
    # those should be defined in the config file as well.
    v1 = np.array([v1_list[sum(nparams[:i]):sum(nparams[:i+1])]
                   for i in range(len(nparams))], dtype=object) # object added because of DeprecateWarning!
    # note that by now we discard the other v's!
    # we don't want to overwrite the old list now that we've
    # changed one of its components
    p = [parameters[0], [p for p in parameters[1]]]
    p[1][p[0].index('parameters')] = v1
    return p



