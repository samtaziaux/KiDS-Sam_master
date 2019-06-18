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


=============   ====================================    ==================================================================================================================================  ==============================================================
 Prior          Description                             Expression                                                                                                                          Notes
-------------   ------------------------------------    ----------------------------------------------------------------------------------------------------------------------------------  --------------------------------------------------------------
``exp``         Exponential distribution                :math:`\exp(-x)`                                                                                                                     Location and scale as per ``scipy.stats`` not yet implemented
``jeffreys``    Jeffreys prior                          :math:`\frac1{x}`                                                                                                                   See `Wikipedia <https://en.wikipedia.org/wiki/Jeffreys_prior>`_
``lognormal``   Lognormal distribution                  :math:`\frac{\sqrt{2\pi\sigma^2}}{x}\exp\left[-\frac{(\log(x)-x_0)^2}{2\sigma^2}\right]`
``normal``      Gaussian distribution                   :math:`\sqrt{2\pi\sigma^2}\exp\left[-\frac{(x-x_0)^2}{2\sigma^2}\right]`
``student``     Student's *t* distribution              :math:`\frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})}\left(1+\frac{x^2}{\nu}\right)^{\!-\frac{\nu+1}{2}}`     Appropriate for slopes (when used with 1 degree of freedom). See `Wikipedia <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_,
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

    name    prior    [arg1    [arg2]]    [lower    upper]    [starting]


but they all take different kinds of arguments. The notation here follows the ``unix`` convention that values in brackets are 
optional; if a set of brackets includes more than one value then if they are specified they must all be specified. The values taken 
by each of the available priors are: ::

    name            exp          [lower    upper]    [starting]
    name            jeffreys     [lower    upper]    [starting]
    name            lognormal    centre     scale    [lower    upper]     [starting]
    name            normal       centre     scale    [lower    upper]     [starting]
    name            student      dof    [lower    upper]     [starting]
    name            uniform      lower    upper    [starting]
    name            fixed        value
    name            array        value1,value2,value3,...
    name            read         file    comma_separated_columns
    section.name

The last line is a ``repeat`` "prior"; it is recognized simply by having a period in its name and takes no additional information 
(i.e., don't give any other parameters names with periods!). Above, ``name`` is a user-defined name for each parameter, and 
corresponds to the name that the parameter will have in the output MCMC chain (see `<outputs>`_), while ``lower`` and ``uper`` are 
lower and upper bounds of the allowed range for the prior. For instance, a mass might have a normal prior :math:`2\pm1`, but it 
cannot physically go below zero. In this case, you'd want ``lower=0``. If not provided, the default limits are as follows:

================  ==================    ===================
 Prior             Lower bound           Upper bound
----------------  ------------------    -------------------
``exp``           :math:`-10`           :math:`10`
``jeffreys``      :math:`10^{-10}`      :math:`100`
``lognormal``     :math:`-10\sigma`     :math:`10\sigma`
``normal``        :math:`-10\sigma`     :math:`10\sigma`
``student``       :math:`-10^6`         :math:`10^6`
================  ==================    ===================

The total probability for points outside these ranges is :math:`<10^{-7}` in all cases. The default 
ranges for ``normal`` and ``lognormal`` depend on the chosen width of the distribution, but for all others the default range is 
defined by absolute values. In general it is a good idea for all free parameters to be of order 1; the user should use appropriate 
normalizations to this end.

Notes
-----

* The ``exp`` and ``jeffreys`` priors take no free parameters. Make sure the parameters are of order unity or the results might not be sensible.
* As mentioned before, we recommend using the ``student`` with ``dof=1`` as the prior for any slope parameter.



Outputs
*******

A successful run of the ``kids_ggl`` sampler will output a ``FITS`` file containing all sampled parameters as well as the outputs of 
the halo model
