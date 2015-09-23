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
    parser.add_argument('--demo', dest='demo', action='store_true',
                        help='do a demo MCMC run with the input parameters')
    parser.add_argument('--esd', dest='esd', action='store_true',
                        help='run the ESD production module')
    parser.add_argument('--sampler', dest='sampler', action='store_true',
                        help='run the MCMC sampler')
    parser.add_argument('-c', '--config', dest='config_file')
    args = parser.parse_args()
    # options to turn on and off the data production or halo model?

    # ESD data production
    if args.esd:
        shearcode.run_esd(args.config_file)

    if args.sampler or args.demo:
        # Choose and set up a halo model
        hm_options = hm_utils.read_config(args.config_file)
        # Setup and run MCMC sampler
        sampling_options = sampling_utils.read_config(args.config_file)
        sampler.run_emcee(hm_options, sampling_options, args)

    return

main()
