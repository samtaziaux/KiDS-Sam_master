Installation
============

**a)** Contact Cristóbal Sifón (sifon@astro.princeton.edu) to become a
member of the KiDS-WL repository and join the KiDS-GGL team

**b)** Download `the latest stable version of the KiDS-GGL
pipeline <https://github.com/KiDS-WL/KiDS-GGL/releases/latest>`__ and
unpack,

::

       $ cd path/to/kids_ggl_folder
       $ tar xvf KiDS-GGL-<version>.tar.gz

or, if you chose to download the ``.zip`` file,

::

       $ unzip KiDS-GGL-<version>.zip

where for instance ``<version>=1.0.0``.

**c)** From within the same folder, run

::

       $ python setup.py install [--user]

where the ``--user`` flag is recommended so as not to require root
privileges, and the package will typically be installed in
``$HOME/.local/lib``. If instead you want to install it in a
non-standard place, then type

::

       $ python setup.py install --prefix=path/to/installation/

The commands above will also install some additional packages required
by the pipeline: ``astropy>=1.1.0`` for astronomical utilities (e.g.,
constants and units), ``emcee>=2.1.0`` for MCMC, ``hmf>=3.0.1`` for halo
mass function utilities, ``mpmath>=0.19`` for mathematical utilities,
and ``numpy>=1.5.0``. Optionally, if you want to use the CAMB transfer
functions in your halo models (preffered method), you should also instal
CAMB using the following command:

::

       $ pip install --egg camb

**d)** After the setup script has finished, you should have a copy of
the ``kids_ggl`` executable somewhere in your ``$PATH``, which means you
can run it out-of-the-box from anywhere in your computer. To make sure,
type

::

       $ which kids_ggl

If you do not see any message it means the place where the executable is
is not part of your ``$PATH``. To fix this, look through the
installation messages where the file was placed (in my Macbook this
place is
``/Library/Frameworks/Python.framework/Versions/2.7/bin/kids_ggl``) and
add the following statement to your ``~/.bashrc`` or ``~/.bash_profile``
if your terminal shell is ``bash``:

::

       export PATH=${PATH}:<path_to_kids_ggl_folder>

or the following in your ``~/.cshrc`` or ``~/.tcshrc`` if your shell is
set to ``csh`` or ``tcsh``:

::

       setenv PATH ${PATH}:<path_to_kids_ggl_folder>

where, in my case,
``<path_to_kids_ggl_folder>=/Library/Frameworks/Python.framework/Versions/2.7/bin``.

**e)** You can run an example by typing (see below for details)

::

    $ kids_ggl -c demo/ggl_demo_nfw_stack.txt --sampler --demo

This should show three panels with data points and lines resembling the
Early Science satellite galaxy-galaxy lensing results of `Sifón et al.
(2015)`_ and, after closing it, show the 3x3x14 covariance matrix.

.. _Sifón et al. (2015): http://adsabs.harvard.edu/abs/2015MNRAS.454.3938S


KiDS-GGL as a python module
===========================

At this point you are also able to import any component of the KiDS-GGL
pipeline as ``python`` modules for use within your own code. Step **c)**
above should ensure all the components are accessible by your ``python``
installation. For instance, to access the different density profiles
(currently all based on modifications of the NFW profile) and their
associated quantities (such as mass, lensing signal, etc), you may type

::

    > from kids_ggl_pipeline.halomodel import nfw

To get more information on what's available, type

::

    > from kids_ggl_pipeline import halomodel
    > help(halomodel)

Note that the use of ``kids_ggl_pipeline`` as a python module has not been tested 
thoroughly and problems may arise. If so, please report them in ``github``.


Configuration file
==================

See ``demo/ggl_demo_nfw_stack.txt`` and ``demo/ggl_model_demo.txt`` for
guidance. The former is intended for a simple average NFW modelling of
the signal and only works with ``kids_ggl<2.0.0``, while the latter is a
full halo model and requires ``kids_ggl>=2.0.0``.

Running
=======

There are two major things the pipeline can do for you:

**a)** Measure the lensing signal. To do this, type:

::

        $ kids_ggl -c <config_file> --esd

**b)** Estimate halo model parameters from the lensing signal. To do
this, type:

::

        $ kids_ggl -c <config_file> --sampler

The sampler module has a demo option which you should always try before
running a particular model on the data. To this end, type

::

        $ kids_ggl -c <config_file> --sampler --demo

This option will generate the ESD(s) for your chosen set of initial
parameters, print the chi2/dof on screen, overplot the model to the data
points and, once you close this plot, will display the full covariance
matrix.

*New in version 2:* By default, if the output file exists the pipeline will
ask the user whether they want to overwrite it. This can be skipped by
setting the ``-f`` flag in the command line:

::

        $ kids_ggl -c <config_file> --sampler -f


**Some suggestions to make the best of this pipeline:**

-  Make sure you have tested your model using the demo before you set up
   a full run!
-  After running a demo, run an MCMC chain of your model with as few
   steps as possible to make sure that the output looks the way you want
   it to look. Fix anything that you can and report any possible bugs.
-  Always check how many cores are available in your machine before
   running in parallel.
-  **Contribute!**


Questions
^^^^^^^^^

If you have any questions, please (preferably) raise an issue in
``github``, or contact Andrej Dvornik (dvornik@strw.leidenuniv.nl),
Margot Brouwer (brouwer@strw.leidenuniv.nl) and/or Cristóbal Sifón
(sifon@astro.princeton.edu).
