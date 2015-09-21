#!/usr/bin/env python
import argparse
import os

# KiDS-GGL modules
#import esd_production
#import halomodel
#import sampling
#import sampler, sampling_utils
#import hm_utils
from esd_production import shearcode
from sampling import sampler, sampling_utils
from halomodel import hm_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_file')
    parser.add_argument('--demo', dest='demo', action='store_true',
                        help='do a demo run with the input parameters')
    args = parser.parse_args()
    # options to turn on and off the data production or halo model?

    # ESD data production
    #shearcode.run_esd(args.config_file)
    #return

    # Choose and set up a halo model
    hm_options = hm_utils.read_config(args.config_file)
    # Setup and run MCMC sampler
    sampling_options = sampling_utils.read_config(args.config_file)
    sampler.run_emcee(hm_options, sampling_options, args)

    return

main()
