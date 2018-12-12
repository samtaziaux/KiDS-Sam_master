from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np


def axlabel(return_value):
    # probably want to have units in `setup` as well
    labels = {'esd': r'\Delta\Sigma',
              'kappa': r'\kappa',
              'power': r'\P(k)',
              'sigma': r'\Sigma',
              'xi': r'\xi'}
    return labels[return_value]


def covariance(R, cov, output=None):
    # still need to add axes labels (complicated by the fact that
    # bins aree usually in log space), and also add a colorbar
    # with a label i.e., this function should probably take `setup`
    # as an (optional) argument
    fig, axes = plt.subplots(
        figsize=(10,8), nrows=cov.shape[0], ncols=cov.shape[0])
    vmin, vmax = np.percentile(cov, [1,99])
    if len(R) == 1:
        axes = [[axes]]
    for m, axm in enumerate(axes):
        for n, axmn in enumerate(axm):
            axmn.imshow(cov[m][-n-1][::-1], interpolation='nearest',
                        vmin=vmin, vmax=vmax)
    fig.tight_layout(pad=0.4)
    if output:
        save(output, fig, 'covariance')
    else:
        plt.show()
    return


def save(output, fig=None, name='', tight=True, close=True,
         verbose=True, pad=0.1, **kwargs):
    if fig is None:
        fig = plt
    if tight:
        fig.tight_layout(pad=pad, **kwargs)
    plt.savefig(output)
    if close:
        plt.close()
    if verbose:
        if name:
            name = ' {0}'.format(name)
        print('Saved {0} to {1}'.format(name, output))
    return


def signal(R, y, yerr, model, observable, output=None):
    """
    observable should be the name of the thing on the y axis

    how exactly should we decide when the y axis should be
    log scale?
    """
    n = len(R)
    fig, axes = plt.subplots(figsize=(4*n,4), ncols=n)
    ylabel = axlabel(observable)
    if n == 1:
        axes = [axes]
    for ax, Ri, yi, ei, fi in zip(axes, R, y, yerr, model):
        Ri = Ri[1:]
        ax.errorbar(Ri, yi, yerr=ei, fmt='ko', ms=10)
        ax.plot(Ri, fi, 'r-', lw=3)
        ax.set_xscale('log')
        # do something fancier later
        ax.set_xlabel('$k$' if observable == 'power' else '$R$')
    axes[0].set_ylabel(r'${0}$'.format(ylabel))
    if np.all(y-yerr > 0):
        for ax in axes:
            ax.set_yscale('log')
    ylims = np.array([ax.get_ylim() for ax in axes])
    ylim = (ylims[:,0].min(), ylims[:,1].max())
    for ax in axes:
        ax.set_ylim(*ylim)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout(pad=0.4)
    if output:
        save(output, fig, 'signal')
    else:
        plt.show()
    return


