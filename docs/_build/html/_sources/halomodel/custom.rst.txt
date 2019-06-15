=============================
 Customizing the Halo Model
=============================

Since ``v2.0``, ``kids_ggl`` can interpret any model passed to it, no matter the number of parameters or the ordering of the 
ingredients, provided that the configuration file is consistent with the model structure.

At the lowest level, the user can modify ``halomodel/halo.py`` to add or remove ingredients, although this is not generally 
recommended. We have built the default model in ``kids_ggl`` to be flexible enough to acommodate most weak lensing uses (including 
CMB weak lensing). The most notable shortcoming of the model right now is the requirement that distances be in physical or comoving 
units; angular units are not implemented.

More common might be the desire of the user to modify the halo occupation distribution (HOD), which describes how galaxies populate 
dark matter haloes. For this purpose we have implemented a few generic functions and distributions, which can serve as backbones for 
particular-case functions, or which can be replaced altogether by user-supplied definitions.

Custom functions
****************

Several functional forms_ have been implemented. Let's take as an example a function that is a power law of both mass and redshift:

.. math::
    \log m_\star = A + B\log\left(\frac{M_\mathrm{h}}{M_0}\right) + C\left(\frac{1+z}{1+z_0}\right)\,.

This function is implemented in ``helpers/functions.py`` as (see below for function-decorators_) ::

    @logfunc
    @zarray
    def powerlaw_mz(M, z, logM0, z0, a, b, c, return_log=True):
        return (a + b*(log10(M)-logM0)) + c*log10((1+z)/(1+z0))

This function, as well as any other custom function, must include the halo mass as the first argument; all other arguments must be 
defined in the configuration file -- including the redshift, in this case. This would therefore require the central MOR section in 
the configuration file to look like this: ::

    [hod/centrals/mor]
    name           powerlaw_mz
    z              array          0.188,0.195
    logM0          fixed          14
    z0             fixed          0.2
    A              uniform        10      16    12
    B              uniform        0       5     1
    C              uniform        -1      1     0

The only condition when writing a custom model is that the order of sections **must follow the order defined in the coded 
model**.

For the time being, additional scaling relations must be included in ``hod/relations.py``, and custom distributions for the scatter 
about this relation must be included in the file ``hod/scatter.py``. (This has the undesirable effect that this file, common to all 
users, might get clogged with trial-and-error attempts by various users, so please try to only push files with working models [and 
if you have a new working model, please push it!]. We will implement the ability to input user-provided files in the future.)

The user may also use the generic functions and distributions implemented in ``helpers/functions.py`` and 
``helpers/distributions.py``. For instance, the Duffy et al. (2008) mass-concentration relation is implemented in 
``halomodel/concentration.py`` as: ::

    from ..helpers.functions import powerlaw_mz

    def duffy08_crit(M, z, f, h):
        return f * powerlaw_mz(
            M, z, 12.301-log10(h), 0, 0.8267, -0.091, -0.44, return_log=False)

In this case, ``return_log`` has been set to ``False`` because ``kids_ggl`` always works with the concentration, rather than the 
log of the concentration. Furthermore, the ``return_log`` parameter, which activates the ``logfunc`` decorator, only checks if **the 
observable** has been defined in log space, and therefore does not work for any other quantity. For convenience, we have set it to 
``True`` by default, so the user must always set it to ``False`` if using these functions for purposes other than mass-observable 
relations.

See implemented-functions_ for all available functions.

.. _hod-decorators:

Decorators
----------

All decorators must be applied to all custom functions that do not rely on functions implemented in ``helpers/functions.py``:

* ``logfunc``: Return the function correctly in linear or log space depending on the ``observables`` section in the configuration file
* ``zarray``: Add an extra dimension to redshift in order to allow vectorized operations including mass and redshift. NOT YET IMPLEMENTED


Custom distributions
********************

Similar to implemented functions, there are a few *distributions* implemented in ``kids_ggl`` by default. Taking the log normal 
distribution as an example:

.. math::
    \Phi(m_\star|M_h) = \frac{\exp\left[-\frac{\log_{10}(m_\star/m_0)^2}{2\sigma^2}\right]}{\sqrt{2\pi}\log(10)\,\sigma\,m_0}

where :math:`m_0` is the characteristic stellar mass and :math:`sigma` is the scatter in stellar mass at fixed halo mass. This is 
implemented in ``kids_ggl`` as ::

    @logdist
    def lognormal(obs, Mo, Mh, sigma, obs_is_log=False):
        return exp(-((log10(obs/Mo)**2) / (2*sigma**2))) \
            / ((2*pi)**0.5 * sigma * obs * log(10))

where distributions now have the ``obs_is_log`` keyword which activates the ``logdist`` decorator analogously to the ``logfunc`` 
decorator introduced above. The distributions implemented by default can be wrapped analogously to the demonstration with the 
mass-concentration relation above.

