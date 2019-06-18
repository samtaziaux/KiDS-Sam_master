=============================
 Customizing the Halo Model
=============================

Since ``v2.0``, ``kids_ggl`` can interpret any model passed to it, no matter the number of parameters or the ordering of the 
ingredients, provided that the configuration file is consistent with the model structure.

At the lowest level, the user can modify ``halomodel/halo.py`` to add or remove ingredients, although this is not generally 
recommended. We have built the default model in ``kids_ggl`` to be flexible enough to acommodate most weak lensing uses (including 
CMB weak lensing), and if there are specific aspects that a user would like implemented or allowed, it is best to raise an issue in 
Github. Perhaps the most notable shortcoming of the model right now is the requirement that distances be in physical or comoving 
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

Similar to implemented functions, there are a few *distributions* implemented in ``kids_ggl`` by default. For instance, let's say we 
would like to implement a lognormal distribution in mass, modulated by a power law in redshift, :math:`z`, for the scatter in 
stellar mass for fixed halo mass for central galaxies:

.. math::
    \Phi_c(m_\star|M_h,z) = \frac1{\sqrt{2\pi}\log(10)\,\sigma\,m_\star}\exp\left[-\frac{\log_{10}[m_\star/m_\star^c(M_h)]^2}{2\sigma^2}\right] \left(1+z\right)^{a_z}

where :math:`m_\star^c(M_h)` is the stellar mass predicted by the mass-observable relation, :math:`\sigma` the scatter about 
:math:`M_0`, and :math:`a_z` is the exponent of the power-law dependence in redshift. This distribution should be implemented in 
:``hod/scatter.py`` as: ::

    @logdist
    def lognormal_mz(obs, Mo, Mh, z, sigma, az, obs_is_log=False):
        return (1 / ((2*pi)**0.5*log(10)*sigma*Mo) \
            * exp(-log10(Mh/Mo)**2/(2*sigma**2)) * (1+z)**az

Analogously to ``logfunc``, distributions must be decorated with ``logdist``, and the argument controlling the decoration is now 
called ``obs_is_log``, which should be set to ``False`` by default. All distribution definitions take three mandatory parameters: 
``obs``, the observable (e.g., stellar mass), ``Mo``, the observable predicted by the mass-observable relation 
(:math:`m_\star^c(M_h)` in the equation above), and ``Mh``, the halo mass. These arguments are passed internally within 
``kids_ggl``. All otherarguments are arbitrary so long as they are reflected in the corresponding entry in the configuration file: 
::

    [hod/centrals/scatter]
    cosmo.z
    sigma       jeffreys    0.01    1.0
    az          student     1

Here, we've assigned redshift as a repeat parameter that was already defined in the ``cosmo`` section. We also sample ``sigma`` with 
a Jeffreys prior, and ``az``, being a slope (in log space), with a Student's :math:`t` prior with one degree of freedom. The two 
values passed to ``sigma`` are the lower and upper bounds (it cannot be less than zero, but we assign it to a small non-zero number 
to avoid infinities). 

