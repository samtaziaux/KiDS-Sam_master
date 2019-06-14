=========================
 Sampling the Halo Model
=========================

Parameters in ``kids_ggl`` are sampled using :code:`emcee`'s `EnsemleSampler`. This is well suited for unimodal likelihoods when 
sampling a relatively small number of parameters, but might either break down or become too slow if we wanted to sample, say, 
cosmological parameters; in all cases, the user should make sure that the implementation in ``kids_ggl`` is suitable for their 
particular problem.


Priors
******

Available priors
^^^^^^^^^^^^^^^^

Just as important as the sampler used are the priors used. Since ``v2.0.0``, several priors are available in ``kids_ggl``. These are 
all defined in ``sampling/priors.py``:

* ``exp``: Exponential prior, :math:`f=exp(-x)`

  * Note: location and scale as per ``scipy.stats`` not yet implemented.)
* ``jeffreys``: Jeffreys prior (`https://en.wikipedia.org/wiki/Jeffreys_prior`_), :math:`f=1/x`

  * Typically used for scatter (more generally, for *scale estimators*).
* ``lognormal``: Lognormal distribution, :math:`f = (1/x)(2\pi\sigma^2)^{0.5}\exp\left[-(\log(x)-x_0)^2/(2\sigma^2)\right]`
* ``normal``: Normal (Gaussian) distribution, :math:`f = (2\pi\sigma^2)^{0.5}\exp\left[-(x-x_0)^2/(2\sigma^2)\right]`
* ``student``: Student's t distribution (`https://en.wikipedia.org/wiki/Student%27s_t-distribution`_)

  *  Appropriate for slopes (when used with 1 degree of freedom). Exercise: see why this works and a uniform prior does not!
* ``uniform``: Uniform distribution.

Three additional "priors" for fixed parameters (i.e., that are not allowed to vary in the MCMC) are available:

* ``fixed``: Fixed scalar
* ``array``: Array of fixed scalars
* ``read``: Array of fixed scalars, read from a file

And finally, there is a "prior" that signals that a parameter has already been defined in the configuration file: ``repeat``.

=============   ====================================    ==================================================================================================================================  ==============================================================
 Prior          Description                             Expression                                                                                                                          Notes
-------------   ------------------------------------    ----------------------------------------------------------------------------------------------------------------------------------  --------------------------------------------------------------
``exp``         Exponential distribution                :math:`\exp(-x)`                                                                                                                     Location and scale as per ``scipy.stats`` not yet implemented
``jeffreys``    Jeffreys prior                          :math:`\frac1{x}`                                                                                                                         ADD LINK
``lognormal``   Lognormal distribution                  :math:`\frac{\sqrt{2\pi\sigma^2}}{x}\exp\left[-\frac{(\log(x)-x_0)^2}{2\sigma^2}\right]`
``normal``      Gaussian distribution                   :math:`\sqrt{2\pi\sigma^2}\exp\left[-\frac{(x-x_0)^2}{2\sigma^2}\right]`
``student``     Student's *t* distribution              :math:`\frac{\Gamma(\frac{\nu+1}{2})} {\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})} \left(1+\frac{t^2}{\nu} \right)^{\!-\frac{\nu+1}{2}}`  Appropriate for slopes (when used with 1 degree of freedom).
                                                                                                                                                                                            **Exercise:** see why this works and a uniform prior does not!
``uniform``     Uniform distribution                    :math:`x\in[a,b]`
``fixed``       Fixed scalar
``array``       Array of fixed scalars
``read``        Like ``array`` but read from a file
``repeat``      Repeat parameter
=============   ====================================    ==================================================================================================================================  ==============================================================



Usage
^^^^^

All priors are defined in the configuration file as: ::

    param_name    prior    [arg1    [arg2]]    [lower    upper]    [starting]


but they all take different kinds of arguments. The notation here follows the ``unix`` convention that values in brackets are 
optional; if a set of brackets includes more than one value then if they are specified they must all be specified. The values taken 
by each of the available priors are: ::

    param_name            exp          [lower    upper]    [starting]
    param_name            jeffreys     [lower    upper]    [starting]
    param_name            lognormal    centre     scale    [lower    upper]     [starting]
    param_name            normal       centre     scale    [lower    upper]     [starting]
    param_name            student      dof    [lower    upper]     [starting]
    param_name            uniform      lower    upper    [starting]
    param_name            fixed        value
    param_name            array        value1,value2,value3,...
    param_name            read         file    comma_separated_columns
    section.param_name

The last line is a ``repeat`` "prior". It is recognized simply by having a period in its name and takes no additional information 
(i.e., don't give any other parameters names with periods!). Above, ``lower`` and ``uper`` are lower and upper bounds of the allowed 
range for the prior. For instance, a mass might have a normal prior :math:`2\pm1`, but it cannot physically go below zero. In this 
case, you'd want ``lower=0``. If not provided, the default limits are as follows:

================  ==================    ===================    ====================
 Prior             Lower bound           Upper bound            Cum. prob. outside
----------------  ------------------    -------------------    --------------------
``exp``           :math:`-10`           :math:`10`             :math:`2\cdot10^{-9}`
``jeffreys``      :math:`10^{-10}`      :math:`100`            
``lognormal``     :math:`-10\sigma`     :math:`10\sigma`
``normal``        :math:`-10\sigma`     :math:`10\sigma`
``student``       :math:`-10^6`         :math:`10^6`           :math:`3\cdot10^{-7}`
================  ==================    ===================    ====================

The last column shows what is the probability of a data point falling outside the default range under each distribution. The default 
ranges for ``normal`` and ``lognormal`` depend on the chosen width of the distribution, but for all others the default range is 
defined by absolute values. In general it is a good idea for all free parameters to be of order 1; the user should use appropriate 
normalizations to this end.

Notes
-----

* The ``exp`` and ``jeffreys`` priors take no free parameters. Make sure the parameters are of order unity or the results might not be sensible.
* As mentioned before, we recommend using the ``student`` with ``dof=1`` as the prior for any slope parameter.
