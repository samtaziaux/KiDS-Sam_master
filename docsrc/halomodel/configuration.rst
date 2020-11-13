===================================
 The Halo Model Configuration File
===================================

This document explains how to write a ``kids_ggl`` halo model configuration file with sections and subsections, allowing the user to 
provide their own functions for e.g., the mass-observable relation and its intrinsic scatter.

..
    replace the important note below by a link to examples or so


**Important Note:**
A working configuration file as required by ``halo.model`` can be found in the ``demo`` folder within the pipeline directory tree. 
Also note that none of this applies to the ``esd_production`` functionality; the configuration file remains exactly the same as in 
``v1.x`` for ``esd_production``.


File structure
**************

The configuration file in ``KiDS-GGL`` versions ``2.0.0`` and newer is defined by a set of *sections*, each of which specifies a 
particular aspect of the pipeline. Sections are defined with square brackets, and the particular sections are fixed and their names 
cannot be changed. Each section takes a different kind of input depending on its purpose. Throughout the configuration file, lines 
that start with a ``#`` will be ignored.


Below we explain every section separately; a full working file can be found in the ``demo`` folder.

Model definition
****************

The configuration file starts with a single line specifying the model to be used: ::

    model            halo_2.model

which is fixed for now.

Model ingredients
*****************


HOD observables
^^^^^^^^^^^^^^^

The first section defines the observables: ::

    [observables]
    logmstar    gm  10,11   11,12.5     log

The first column here is an arbitrary (and dummy) name for the observable in question; the second colum specifies the type of
observable being passed: it is ``gm`` for galaxy-galaxy lensing, ``gg`` for galaxy clustering, ``mm`` for matter-matter correlation
and ``mlf`` for the stellar mass or luminosity function. The third and fourth columns are the lower
and upper bin limits used in the stacking (comma-separated). In this case, we use two bins in the ranges :math:`10 \leq \log_{10} 
m_\star \leq 11` and :math:`11 \leq \log_{10} m_\star \leq 12.5`; note that the bins need not be adjacent nor exclusive.

The final column is optional, and its only possible value is ``log``. This tells the pipeline that the observable is given in 
log-space. This is necessary information for properly applying the HOD (i.e., mass-observable relation and mass-observable scatter 
function, a.k.a. occupation probability). If the column is not present, it is understood that the observable is given in linear 
space. Any value different than ``log`` will raise an ``AssertionError``.

Note that the HOD itself does not care what observable it receives so long as everything is passed consistently. It is possible, for 
instance, to define an observable :math:`m_\star^{12} \equiv m_\star/10^{12}`: ::

    mstar12        mlf  0.01,0.1,1       0.1,1,10

Note here that :math:`m_\star^{12}` is not in log space and we therefore omit the fourth column.

One can pass multiple observables. In that case it is important that the order follows the order in which the data one is passing
is structured in the same way. Remember that the number of bins will be determined by the number of bins specified in the
third and fourth column: ::
    
    logmstar    gm  10,11   11,12.5     log
    logmstar    gg  10,11   11,12.5     log
    logmstar    mlf  6   18     log


Selection function
^^^^^^^^^^^^^^^^^^

The second section concerns the selection function: ::

    [selection]
    selection_file.txt       z,logmstar,completeness      fixed_width

Columns here correspond to the file name, the names of the columns to be used, and the ``astropy.io.ascii`` format to be used when 
reading the file. Using inconsistent formats, or not providing one (which is not allowed by ``kids_ggl``) would probably result in 
the column names not being interpreted correctly. We recommend always using ``fixed_width`` with its default settings to avoid any 
confusion. A file generated with ``ascii.write(*args, format='fixed_width')`` is also easily human-readable: ::

    |    z | logmstar | completeness |
    | 0.10 | 2.00e+09 |     3.36e-01 |
    | 0.15 | 2.00e+09 |     3.36e-01 |
    | 0.20 | 2.00e+09 |     3.36e-01 |
    | 0.30 | 2.00e+09 |     3.36e-01 |
    | 0.40 | 2.00e+09 |     3.36e-01 |
    | 0.50 | 2.00e+09 |     3.36e-01 |
    | 0.60 | 2.00e+09 |     3.36e-01 |
    | 0.80 | 2.00e+09 |     3.36e-01 |
    | 1.00 | 2.00e+09 |     3.36e-01 |
    ...

Column names specified in the second column above must be in that order: ``(redshift,observable,completeness)``. They will be 
recorded in that order irrespective of the order in which they are present in the file.

This section is the only section that can be disabled altogether. We've chosen to allow this so the user is not required to produce 
a dummy selection file if they do not have/want one. To ignore selection effects -- which is equivalent to assuming that the sample 
is 100% complete -- simply write ::

    [selection]
    None


Halo model ingredients
^^^^^^^^^^^^^^^^^^^^^^

Here we define which ingredients will be included in the halo model: ::

    [ingredients]
    centrals        True
    pointmass       True
    satellites      False
    miscentring     False
    twohalo         True
    haloexclusion    False
    nzlens          False

The names of the ingredients cannot be changed, and any missing ingredients will be set to ``False``. If any provided ingredient 
does not match those listed above, a ``ValueError`` will be raised. All of these accept values of ``True`` or ``False``. In the 
example above, ```kids_ggl`` will calculate a model of central galaxies only, including a point mass component to account for the 
stellar mass, and a two halo term, and it is assumed that central galaxies are always co-located with the center of mass. In this case
will not account for the fact that haloes cannot overlap.

The parameter ``nzlens`` enables a model in which the final signal is the weighted average of signals calculated following a 
user-provided lens redshift distribution (see below). *WARNING*: setting ``nzlens`` to ``True`` significantly slows down the halo 
model computation.


Model parameters
****************

General description
^^^^^^^^^^^^^^^^^^^

Like above, we now define model *sections*, which refer to each of the components of the halo model. The order of sections must be 
followed for the model to be set up properly, but each component may have a custom number of parameters. This is structure allows 
full flexibility in the model used without having to modify the backbone provided by ``halo.model``: ::

    [section1/subsection1]
    name1         prior1     [values1 ...]     [join:label1]
    name2         prior2     [values2 ...]     [join:label2]
    ...
    [section1/subsection1/subsubsection1]
    name1         prior1     [values1 ...]     [join:label1]
    name2         prior2     [values2 ...]     [join:label2]
    ...
    [section2/subsection1]
    name1         prior1     [values1 ...]     [join:label1]
    name2         prior2     [values2 ...]     [join:label2]
    ...

etc, etc. This will be enough to explain what's going on. The first column is the name that will be used in the MCMC output if the 
parameter is varied during the chain; the second specifies the prior function, and following columns specify parameters passed to 
the prior function. See `Priors <sampler.html#Priors>`_ for details.

.. _join:

Joint parameters
----------------

The last column, which is also optional, allows the use of "array variables" by *joining* individual entries within the model, and a 
*join label* identifies which entries should be joined. Thus for instance in the default ``halo.model`` the satellite occupation 
function (see ``demo/ggl_model_demo.txt``) takes an array variable :math:`b=(b_0,b_1,...b_n)` in Eq. 18 of |ref:vanuitert16|_. The 
advantage of this is that you may specify any number of these variables and the model will use them consistently, in the case of van 
Uitert et al., through the equation

.. math::

    \log[\phi_s(M_\mathrm{h})] = \sum_{i=0}^{n} b_i\,\left(\log M_\mathrm{13}\right)^i


where :math:`n=2` in the paper.

**Notes**
 * Joint parameters need not be fixed. You may use any prior for each of the joint parameter separately, even e.g., fixing some but not others. 
 * Even if you decide to use only one of the joint parameters (e.g., only :math:`b_0` above), **you must still give it the join label**. This will allow ``kids_ggl`` to interpret it as an array rather than a scalar (the ``array`` prior can only be used if the variable is fixed).

Repeat parameters
-----------------

In addition, the use of repeat parameters is supported: ::

    [section1/subsection1]
    name1         prior1     [values1 ...]     [join:label1]
    name2         prior2     [values2 ...]     [join:label2]
    ...
    [section2/subsection1]
    section1.subsection1.name1
    name2         prior2     [values2 ...]     [join:label2]
    ...

The above syntax -- defining a parameter as a "child" of another section, where the section hierarchy is split by dots -- means that 
the first parameter of ``section2/subsection1`` is always the same as ``name1`` in ``section1/subsection1``. This is useful if there 
are free parameters required in more than one place (for instance, ``h`` may be used in the cosmology as well as the 
mass-concentration or mass-observable relations, or some of the parameters used for satellite galaxies might be based on those 
obtained for centrals). Although redshift would hardly be a free parameter in any model, it might be useful to define it as a repeat 
parameter as well if it is required in more than one place so that only one instance has to be modified by hand in the configuration 
file; the most common example is redshift.

Cosmological parameters
^^^^^^^^^^^^^^^^^^^^^^^

The first section including model parameters that may be sampled in the MCMC is ``cosmo``, the section listing cosmological 
parameters: ::

    [cosmo]
    sigma_8         fixed     0.8159
    h               fixed     0.6774
    omegam          fixed     0.3089
    omegab_h2       fixed     0.02230
    n               fixed     0.9667
    w0              fixed     -1.0
    wa              fixed     0.0
    Neff            fixed     3.046
    z               array     0.188,0.195,0.188,0.195,0.19


Both the list of parameters and their order in ``cosmo`` are mandatory (and therefore the names are just for the user's reference). 
The first 8 parameters define the ``Flatw0waCDM`` cosmology within which the model is evaluated. The default values for ``w0``, 
``wa``, and ``Neff`` make the default model a flat ``LambdaCDM`` cosmology. They need not all be fixed, but in this example they are 
(note, however, that at the moment the ``kids_ggl`` halo model can only handle physical distances, which require a cosmology to be 
calculated). The last parameter above is a list of point estimates for the redshifts of each bin. List of redshifts should reflect
the total number of bins given by the observable(s). In this case, it should have 5 entries.


Optional cosmological parameters
--------------------------------

In addition to the above list, there are a few optional parameters, depending on the chosen setup:

Lens redshift distribution
++++++++++++++++++++++++++

If the ``nzlens`` ingredient is activated, then the parameter ``z`` must instead be an array, defined with either an ``array`` or 
``read`` prior (see `Priors <sampler.html#Priors>`_), containing the redshift values at which the distribution is calculated (e.g., 
the central values used to construct a histogram); and there should be an additional set of parameters ``nz``, one per observable 
bin: ::

    z           read    zlens.txt       0
    nz          read    zlens.txt       1   join:nz
    nz          read    zlens.txt       2   join:nz

where the join_  flag must be set for all ``nz`` parameters, even if there is only one such parameter. 

Source redshift
+++++++++++++++

Finally, if the observable returned by the halo model requires a source redshift (e.g., convergence; see `General setup`_), then the 
last entry in ``cosmo`` must be a source redshift, ``zs``. Currently ``zs`` should be a single number rather than a source redshift 
distribution, as it has been implemented to work with CMB lensing measurements, which are usually reported as a convergence, and 
where the source redshift is known to be :math:`z_s=1100`: ::

    zs          fixed   1100


HOD parameters
^^^^^^^^^^^^^^

Now we move on to the HOD proper, and this is where the fun starts. The following are sections that can be modified seamlessly to 
produce a variety of halo model prescriptions, taking advantage of the backbone established by ``halo.model`` (i.e., the user need 
not write their own full-fledged model to do this): ::

    [hod/centrals/pointmass]
    logmstar        array     10.3,11.5
    point_norm      uniform     0.5     5
    [hod/centrals/concentration]
    name            duffy08_crit
    cosmo.z         repeat
    fc              uniform     0.2     5
    cosmo.h
    [hod/centrals/mor]
    name            powerlaw
    logM0           fixed          12.0
    a               uniform        -5      5
    b               student        1
    [hod/centrals/scatter]
    name            lognormal
    sigma_c         jeffrey
    [hod/centrals/miscentring]
    name            fiducial
    p_off           uniform        0       1
    R_off           uniform        0       1.5
    
    [hod/twohalo]
    bias            uniform     0.2     5
    
    [hod/satellites/beta]
    beta            fixed    1

The idea behind this structure is that the HOD may be fully specified by the user, including for instance the complexity of the 
mass-observable scaling relation (see `Customizing the Halo Model <custom.html>`_). Note that the HOD may also contain a model for 
satellites and potentially other ingredients (as suggested by the ``ingredients`` section above), but a simple centrals-only model 
will serve our purpose here (but note that ``halo.model`` does require satellite sections to be defined; please refer to 
``demo/ggl_model_demo.txt`` for a full working configuration file). While it is the order of the sections that is enforced by the 
halo model, the ``hod/`` prefix to all HOD sections is required for the configuration file to be read properly. The names and depths 
of sections/subsections is arbitrary, though we recommend that the section names not be modified for consistency and ease of 
interpretation by other users.

In the example above we've only included mandatory parameters for each prior type, to keep it simple. For more 
information see the `Priors <sampler.html#Priors>`_ section.

**Note:** The miscentring implementation has not yet been modified from ``v1``, and therefore the ``name`` parameter is silent 
for now (but still must be defined and given a value). No matter the value given, miscentring will be modelled as in |ref:viola15|_
Should anyone require more flexibility please raise an issue and we will make this a more urgent update.

**Note:** beta is a Poisson parameter, beta = <N_s (N_s - 1)|M>/<N_s|M>^2 (Eq. 42 in Dvornik et al. 2018)

General setup
*************

There is an additional section, ``setup``, which includes details on, well, the setup of the model. There are four mandatory
parameters in this section: ::

    [setup]
    return          esd        # one of {'esd', 'kappa', 'sigma', 'esd_wp', 'wp', 'power'}
    delta           200        # adopted overdensity
    delta_ref       SOMean       # reference background density. Can be one of {'FOF', 'SOCritical', 'SOMean', 'SOVirial'}, see `hmf` documentation for details.
    distances       comoving     # whether to work with 'comoving' or 'proper' distances

and other parameters that, if omitted, are assigned their default values. For the time being, these are the :math:`\ln k` and 
:math:`\log M` binning schemes (the former is set to 1,000 bins in the range (-15,10), and the latter to 200 bins over (10,16)), as 
well as the transfer function solver, which can be set to either ``EH`` (default) or ``CAMB`` (refer to the `hmf documentation 
<https://hmf.readthedocs.io/en/latest/>`_) for details). When using ``CAMB``, be sure to use a smaller :math:`k` range, as the 
default numbers used here make it *very* slow. Specifically, the available optional setup parameters are: ::

    transfer        EH
    lnk_bins        1000
    lnk_min         -15
    lnk_max         10
    logM_bins       200
    logM_min        10
    logM_max        16

In your own model, the ``setup`` section should include any parameter in the halo model that would *never* be a free parameter (not 
even a nuisance parameter); for instance, binning schemes or any precision-setting values. Note that ``setup`` is a dictionary in 
``kids_ggl`` and therefore the order of its entries is irrelevant.

One should also provide the units in which the data is given, so that it is correctly converted to the default units used in the model: ::

    R_unit          kpc
    esd_unit        Msun/pc^2
    cov_unit        Msun^2/pc^4

Finally, there are three additional parameters only used when running ``kids_ggl`` in ``mock`` mode: ::

    logR_bins       12
    logR_min        -1
    logR_max        0.7

which define the radial binning (in Mpc) used to generate the mock data.


Model output
************

*NOTE*: As the pipeline now uses the emcee version 3.0, the specification of ``FITS`` data format and dimensions is not
needed anymore. The pipeline will use the built-in backend HDF5 data handling. See more information at: http://emcee.readthedocs.io

In addition, the configuration file should include a section ``output``, containing any outputs produced by the model in addition to 
the free parameters. These are given as a name and the data format to be used in the ``FITS`` file, in addition to the number of 
dimensions, if applicable. The typical ``FITS`` format would be ``E``, corresponding to single-precision floating point. See the 
`astropy help <http://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation>`_) for more details.

You will usually want to have each ESD component here at the very least. ``halo.model`` outputs the total ESD and the effective halo 
mass per bin (yes, we will fix this soon). In our 2-bin example, we would write ::

    [output]
    esd            2,8E
    Mavg           2,E

where the second column in the first line means to register two separate columns, each with elements corresponding to arrays of 
length 8 (the 8 R-measurements that make up the ESD profile in the demo data); and the second line means to create two other columns 
each containing scalars corresponding to the effective halo masses (this is given by the output of ``halo.model``, *not* by the name 
given in the first column above, i.e., changing the name in ``output`` does not change which parameters are returned!). The first 
column corresponds to the names given to the columns in the output ``FITS`` file. When there is more than one "dimension" (the 
number before the comma), columns are labelled e.g., ``esd1,esd2,...``.

There is one alternative to the example above: ::

    [output]
    esd       2,8E
    Mavg        2E

which, consistent with the description above, will produce a single column ``Mavg``, as opposed to two columns ``Mavg1,Mavg2`` 
above, where each entry contains both masses, rather than making separate columns.


Sampler configuration
*********************

Finally, the ``sampler`` section: ::

    [sampler]
    path_data            path/to/data
    data                 shearcovariance_bin_*_A.txt     0,1,4
    path_covariance      path/to/covariance
    covariance           shearcovariance_matrix_A.txt    4,6

where both ``path_data`` and ``path_covariance`` are optional. Note the (optional) use of a wildcard (``*``) in ``data``: the 
pipeline will then select all matching files. Note that the file names must be such that, when sorted alpha-numerically, they are 
sorted consistent with the observable binning defined in the ``observables`` section. (This is properly taken care of by the 
``kids_ggl`` `ESD production pipeline <../esd-production/index.html>`_).

The third column in ``data`` specifies which columns from the file should be used: R-binning column, ESD column, and optionally the 
multiplicative bias correction column (:math:`1+\mu` in the `convention of the ESD production pipeline 
<../esd-production/procedure.html>`_). Similarly, the third column in ``covariance`` specifies the covariance column and the 
multiplicative bias correction column, given as :math:`(1+\mu_i)(1+\mu_j)`. The covariance file should follow the format produced by 
the ``kids_ggl`` ESD production pipeline. In both cases, the multiplicative bias correction column is optional (omit if the 
correction has already been applied). The column numbers used above correspond to those required if the data come from the 
``KiDS-GGL`` ESD production pipeline.

The ``sampler`` section then continues with a few more settings: ::

    exclude              11,12              # bins excluded from the analysis (count from 0)
    cov_format           2d                 # shape of covariance matrix (2d if a file with data in matrix shape,
                                            #       skip entry if the covariance is in the shape of this pipeline output
    sampler_output       output/model.h5  # output filename (must be .h5)
    sampler              emcee              # MCMC sampler (fixed)
    nwalkers             100                # number of walkers used by emcee
    nsteps               2000               # number of steps per walker
    nburn                0                  # size of burn-in sample
    thin                 1                  # thinning (every n-th sample will be saved, but values !=1 not fully tested)
    threads              3                  # number of threads (i.e., cores)
    sampler_type         ensemble           # emcee sampler type (fixed)
    # update               20000            # frequency with which the output file is written (not used anymore)
    stop_when_converged  True               # stops the emcee when convergence has been reached
    autocorr_factor     100                 # 'autocorr_factor' times the estimated autocorrelation time. If this estimate is changed by less than 1%, we’ll consider things converged
    resume              False               # resumes the chain from the last position, requires previous chain to be present

where only ``exclude``, which should list the numbers of columns excluded from the likelihood evaluation (counting from 0), is 
optional. The total number of MCMC samples will be equal to :math:`n_\mathrm{walkers}\cdot n_\mathrm{steps}`, after which the 
pipeline will stop running. The autocorrelation times are stored in the header file for reference.


Analytical covariance estimation
*************

There is an additional section, ``covariance``, which includes the input parameters needed for the calculation of the anayltical covariance
using the ``covariance.covariance`` module. It uses the covariance matrix calculation as described in the Appendix A. of Dvornik et al. 2018
paper. The required parameters in this section are: ::

    [covariance]
    pi_max              100.0               # maximal projection length as used in clutering measurements to obtain wp
    area                180.0               # area of a survey in deg^2, assumes square geometry
    eff_density         8.53                # effective density in gal/arcmin^2
    variance_squared    0.082               # variance of the galaxy ellipticity (as defined in KiDS papers)
    mean_survey_redshift    0.6             # mean survey redshift
    gauss               True                # calculate the gaussian part of the covariance? [bool]
    non_gauss           True                # calculate the non-gaussian part of the covariance? [bool]
    ssc                 True                # calculate the super-sample part of the covariance? [bool]
    cross               True                # calculate the cross terms between observables? [bool]
    subtract_randoms    False               # is the signal from random points subtracted from the data? [bool]
    threads             4                   # number of threads to use to calculate the covariance matrix (independt from sampler, uses different paralellisation method)
    output              covariance_test.txt # output filename

*NOTE*: All other sections for the halo model have to be specified as well to run the covariance code succesfully!
Currently the sampler setup is also required, but only the required parameters have to be specified, and won't be actually used.

Future improvements
*******************

* Custom relations should be written in a user-supplied file rather than in the pipeline source code.
* Adding a ``module_path`` optional entry to each section would easily allow custom files: the pipeline could simply add that path to ``sys.path`` and import from there.

  * There is the pickling problem however. Need to check if the above would allow for multi-thread runs.
* Might want to add a ``model`` section proper, in case the above is implemented but more generally for any future changes


.. include:: ../reference-links.rst
