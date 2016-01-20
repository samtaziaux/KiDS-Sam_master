# KiDS-GGL
Galaxy-Galaxy Lensing pipeline production

---
Update log:

2016 Jan 20 - Added easy_install capability (Cristóbal Sifón)

2015 Nov 3 - Added detail on how to install hmf (Andrej Dvornik)

2015 Sep 25 - Finished a first complete version (Cristóbal Sifón)

2015 Aug 11 - Created but left incomplete (Cristóbal Sifón)

---

*This Readme contains all the instructions needed to install, set up and run
the KiDS galaxy-galaxy lensing pipeline, which takes current KiDS and GAMA
catalogs and produces (or reads) an ESD and a covariance matrix and runs a
fitting module (i.e., halo model) with a given sampling technique.*

1. Installation

    a) Contact Cristóbal Sifón (sifon@strw.leidenuniv.nl) to become a member
       of the KiDS-WL repository and join the KiDS-GGL team


    b) From the folder where you want the package to be located, run:

            git clone git@github.com:KiDS-WL/KiDS-GGL.git

    c) From within the same folder, run

            python setup.py install

    If you don't have administrator privileges or simply want to install it in a non-standard place (e.g., your home directory, then type

            python setup.py install --prefix=path/to/installation/

    Either of these will install some additional packages required by the pipeline: `emcee>=2.1.0` for MCMC, `hmf==1.7.0` for halo mass function utilities, `mpmath` for mathematical utilities, and `numpy>=1.5.0`.

    After the setup script has finished, you should have a copy of the `kids_ggl` executable somewhere in your $PATH.

2. Set up your configuration file. See `help/ggl_demo.config` for guidance.

3. Run! There are two major things the pipeline can do for you:

    a) Measure the lensing signal. To do this, type:

        kids_ggl -c <config_file> --esd

    b) Estimate halo model parameters from the lensing signal. To do this, type:

        kids_ggl -c <config_file> --sampler

    The sampler module has a demo option which you should always try before running a particular model on the data. To this end, type

        kids_ggl -c <config_file> --sampler --demo

    This option will generate the ESD(s) for your chosen set of initial parameters, print the chi2/dof on screen, overplot the model to the data points and, once you close this plot, will display the full covariance matrix.

**Some suggestions to make the best of this pipeline:**

- Make sure you have tested your model using the demo before you set up a full run!
- After running a demo, run an MCMC chain of your model with as few steps as possible to make sure that the output looks the way you want it to look. Fix anything that you can and report any possible bugs.
- Always check how many cores are available in your machine before running in parallel.
- **Contribute!**
