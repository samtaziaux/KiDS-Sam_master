#!/usr/bin/env python
import argparse

# KiDS-GGL modules
import esd_production
import halomodel
import sampling

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_file')
    # options to turn on and off the data production or halo model?

    # ESD data production
    esd_production.run(parser.config_file)

    # Choose and set up a halo model
    hm_options = halomodel.hm_utils.read_config(parser.config_file)
    model, params, param_types, prior_types, \
        val1, val2, val3, val4, hm_functions, \
        starting, meta_names, fits_format = hm_options
    halomodel.config()

    # Setup and run MCMC sampler
    sampling_options = sampling.sampling_utils.read_config(parser.config_file)
    sampling.sampler.emcee(sampling_options, hm_options)

    return

main()
