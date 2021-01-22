=================
 Getting Started
=================

Installation
************

To get started, clone or fork ``kids_ggl`` from the github repository: ::

    git clone https://github.com/KiDS-WL/KiDS-GGL.git

or download `the latest stable version <https://github.com/KiDS-WL/KiDS-GGL/releases/latest>`_ and unpack, ::

    tar xvf KiDS-GGL-<version>.tar.gz

or ::

    unzip KiDS-GGL-<version>.zip

where, for instance, ``<version>=2.0.0``.

Enter the folder where the pipeline is located and run ::

    pip install .

If you are not using Anaconda, you may have to add a ``--user`` flag to the above if you do not have root privileges (though we 
absolutely recommend using Anaconda).

The commands above will also install some additional packages required by the pipeline: ``astropy>=1.2.0`` for astronomical 
utilities (e.g., constants and units), ``emcee>=2.1.0`` for MCMC, ``hmf>=3.0.1`` for halo mass function utilities, ``mpmath>=0.19`` 
for mathematical utilities, and ``numpy>=1.5.0``. Optionally, if you want to use the CAMB transfer functions in your halo models 
(preferred method), you should also install CAMB using the following command: ::

    pip install --egg camb

After the setup script has finished, you should have a copy of the ``kids_ggl`` executable somewhere in your ``PATH``, which means 
you can run it out-of-the-box from anywhere in your computer. To make sure, type ::

    which kids_ggl

If you do not see any message it means the place where the executable is is not part of your ``PATH``. To fix this, look through the 
installation messages where the file was placed and add the following statement to your ``~/.bashrc`` or ``~/.bash_profile`` if 
your terminal shell is ``bash``: ::

        export PATH=${PATH}:<path_to_kids_ggl_folder>

or the following in your ``~/.cshrc`` or ``~/.tcshrc`` if your shell is set to ``csh`` or ``tcsh``: ::

        setenv PATH ${PATH}:<path_to_kids_ggl_folder>

You can run a quick example by typing ::

    kids_ggl -c demo/ggl_halomodel.config --sampler --demo

This should show three panels with data points and lines resembling the Early Science satellite galaxy-galaxy lensing results of 
Sifon et al. (2015) and, after closing it, show the 3x3x14x14 covariance matrix.


.. -------------------------------------------------------------------
   -------------------------------------------------------------------


Usage
*****

``kids_ggl`` requires a configuration file to run, and there are two kinds of configuration files required, depending on the task 
you ask from it: one for the `production of a lensing signal <esd-production/index.html>`_ and one if you want to `calculate and 
sample halo model predictions <halomodel/index.html>`_.


Measure the lensing signal
--------------------------

You can use ``kids_ggl`` to measure the lensing signal from the KiDS data simply by typing ::

    kids_ggl -c <esd_config_file> --esd

In order to do this, you need either the `right set of files in your computer <esd-production/input-data.html>`_ or to do it from 
one of the `KiDS servers <esd-production/input-data.html#servers>`_.

Fit the data with a halo model
------------------------------

You can also use ``kids_ggl`` to generate halo model predictions for the lensing signal and to fit lensing data with a halo model. 
This functionality is activated with the ``--sampler`` flag. In order to fit a halo model to data, type ::

    kids_ggl -c <model_config_file> --sampler

while if you simply want to generate halo model predictions for a single set of halo model parameters (those provided in the 
configuration file), type ::

    kids_ggl -c <model_config_file> --sampler --demo

The line above will generate the ESD(s) for your chosen set of initial parameters, print :math:`\chi^2/\mathrm{dof}` on screen, 
overplot the model to the data points and, once you close this plot, will display the full covariance matrix.

Sometimes, after running the demo a couple times, you don't really need to see the covariance matrix every time you run the demo. 
You can run the demo without plotting the covariance with ::

    kids_ggl -c <config_file> --sampler --demo --no-cov

the ``--no-cov`` option is ignored if ``--demo`` is not present.

By default, if the MCMC output file exists ``kids_ggl`` will ask the user whether they want to overwrite the output. Since version 
``2.0.0``, you can force overwrite by writing ::

    kids_ggl -c <config_file> --sampler -f

Generate mock data
------------------

Finally, you can also use ``kids_ggl`` to generate mock data, based on the parameters entered in the configuration file, by typing 
::

    kids_ggl -c <config_file> --mock

For now, doing this will save the output into a ``mock/`` folder, with the file names containing some information about the 
quantities calculated. These files are ready to be fed into ``kids_ggl`` for halo model fitting, for instance.

*IMPORTANT*: mock covariance calculation has not yet been implemented. The ``kids_ggl`` MCMC sampler requires a covariance 
matrix to work.


Python modules
--------------

After following the installation steps above you will also be able to import any component of the KiDS-GGL pipeline as ``python`` 
modules for use within your own code. For instance, to access the different density profiles (currently all based on modifications 
of the NFW profile) and their associated quantities (such as mass, lensing signal, etc), you may type ::

    from kids_ggl_pipeline.halomodel import nfw

Most functionality should have a working help page, accessed by typing in a ``python`` shell (or Jupyter notebook, etc): ::

    help(nfw)


Acknowledgements
****************

See `References <references.html>`_.
