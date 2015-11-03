# KiDS-GGL
Galaxy-Galaxy Lensing pipeline production

---
Update log:

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

    c) In order to run the halo model (halo.py) and the sampler there are some additional Python packages needed. You will need hmf v1.7.0 and emcee. Running
            
            pip install hmf==1.7.0
            
    should install both, as hmf requires emcee in the first place (The latest version of hmf is 1.8.0, but the halo model has not been tested against it). If not
    
            pip install emcee
            
    will do. hmf specific documentation and source can be found at: https://github.com/steven-murray/hmf

2. Set up your configuration file. See `help/ggl_demo.config` for guidance.

3. Run! There are two major things the pipeline can do for you:

    a) Measure the lensing signal. To do this, type:

        python kids-ggl.py -c <config_file> --esd

    b) Estimate halo model parameters from the lensing signal. To do this, type:

        python kids-ggl.py -c <config_file> --sampler

    The sampler module has a demo option which you should always try before running a particular model on the data. To this end, type

        python kids-ggl.py -c <config_file> --sampler --demo

    This option will generate the ESD(s) for your chosen set of initial parameters, print the chi2/dof on screen, overplot the model to the data points and, once you close this plot, will display the full covariance matrix.

**Some suggestions to make the best of this pipeline:**

- Make sure you have tested your model using the demo before you set up a full run!
- After running a demo, run an MCMC chain of your model with as few steps as possible to make sure that the output looks the way you want it to look. Fix anything that you can and report any possible bugs.
- Always check how many cores are available in your machine before running in parallel.
- **Contribute!**
