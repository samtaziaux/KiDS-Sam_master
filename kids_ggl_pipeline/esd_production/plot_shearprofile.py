#!/usr/bin/python

"Part of the module to determine the shear as a function of radius from a galaxy."

# Import the necessary libraries
import pyfits
import numpy as np
import distance
import sys
import os
import time
import shearcode_modules as shear
from astropy import constants as const, units as u

# Important constants
inf = np.inf # Infinity
nan = np.nan # Not a number


def main():


    filenames = np.loadtxt('/brouwer/environment_project/filenames_enviroment.txt')


    # Input parameters
    Nsplit, Nsplits, centering, ranks, lensid_file, lens_binning, binnum, \
            lens_selection, binname, Nobsbins, src_selection, path_Rbins, name_Rbins, Runit, path_output, \
            path_splits, path_results, purpose, O_matter, O_lambda, Ok, h, \
            filename_addition, Ncat, splitslist, blindcats, blindcat, blindcatnum, \
            path_kidscats, path_gamacats = shear.input_variables()


        # Plotting the data for the separate observable bins
        plottitle = shear.define_plottitle(purpose, centering, ranks, lens_selection, binname, Nobsbins, src_selection)
        
        # What plotting style is used (lin or log)
        if 'random' in purpose:
            plotstyle = 'lin'
        else:
            plotstyle = 'log'
        
        if 'No' not in binname:
            plotlabel = ylabel
        else:
            plotlabel = r'%.3g $\leq$ %s $\textless$ %.3g (%i lenses)'%(binmin, binname.replace('_', ''), binmax, len(galIDs_matched))

        try:
            shear.define_plot(plotname, plotlabel, plottitle, plotstyle, Nobsbins, xlabel, ylabel, binnum)
        except:
            pass

    # Writing and showing the plot
    try:
        shear.write_plot(plotname, plotstyle)
    except:
        pass
        
    return

main()
