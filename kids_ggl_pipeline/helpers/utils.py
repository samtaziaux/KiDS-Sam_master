from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as pl
import curses


def read_esdfiles(esdfiles):
    # Reads in ESD files and applies the multiplicative bias correction
    
    data = np.loadtxt(esdfiles[0]).T
    data_x = data[0]
    
    data_y = np.zeros(len(data_x))
    error_h = np.zeros(len(data_x))
    error_l = np.zeros(len(data_x))
    
    print('Imported ESD profiles: %i'%len(esdfiles))
    
    for f in xrange(len(esdfiles)):
        # Load the text file containing the stacked profile
        data = np.loadtxt(esdfiles[f]).T
        
        bias = data[4]
        bias[bias==-999] = 1
        
        datax = data[0]
        datay = data[1]/bias
        datay[datay==-999] = np.nan
        
        errorh = (data[3])/bias # covariance error
        errorl = (data[3])/bias # covariance error
        errorh[errorh==-999] = np.nan
        errorl[errorl==-999] = np.nan
        
        
        data_y = np.vstack([data_y, datay])
        error_h = np.vstack([error_h, errorh])
        error_l = np.vstack([error_l, errorl])
    
    data_y = np.delete(data_y, 0, 0)
    error_h = np.delete(error_h, 0, 0)
    error_l = np.delete(error_l, 0, 0)
    
    return data_x, data_y, error_h, error_l


def make2dcov(covfile, covcols, Nobsbins, Nrbins, exclude_bins=None):
    # Creates 2d covariance out of KiDS-GGL pipeline output.
    # Useful for combining measurements in conjunction with make4dcov
    # Return cov2d a 2D covariance and a per bin representation (cov)
    
    cov = np.loadtxt(covfile, usecols=[covcols[0]])
    if len(covcols) == 2:
        cov /= np.loadtxt(covfile, usecols=[covcols[1]])
    # 4-d matrix
    if exclude_bins is None:
        nexcl = 0
    else:
        nexcl = len(exclude_bins)
    cov = cov.reshape((Nobsbins,Nobsbins,Nrbins+nexcl,Nrbins+nexcl))
    cov2d = cov.transpose(0,2,1,3)
    cov2d = cov2d.reshape((Nobsbins*(Nrbins+nexcl),
                           Nobsbins*(Nrbins+nexcl)))

    if exclude_bins is not None:
        for b in exclude_bins[::-1]:
            cov = np.delete(cov, b, axis=3)
            cov = np.delete(cov, b, axis=2)

    # switch axes to have the diagonals aligned consistently to make it
    # a 2d array
    cov2d = cov.transpose(0,2,1,3)
    cov2d = cov2d.reshape((Nobsbins*Nrbins,Nobsbins*Nrbins))
    
    return cov2d


def make4dcov(cov2d, nbins, nrbins):
    # Creates 4d covariance out of standard 2d covariance, given bins.
    # Useful for combining measurements in conjunction with make2dcov
    # Returns flattened and per bin representation (cov)

    cov4d_i = cov2d.reshape((nbins,nrbins,nbins,nrbins))
    cov4d = cov4d_i.transpose(2,0,3,1).flatten()
    cov = cov4d.reshape((nbins,nbins,nrbins,nrbins))
    
    return cov4d, cov


def make_block_diag_cov(*covs):
        
    import scipy.linalg
    cov = scipy.linalg.block_diag(*covs)

    return cov


def key(screen, var):
    while True:
        char = screen.getch()
        if char == 27:
            raise SystemExit()
        elif char == curses.KEY_ENTER or char == 10 or char == 13:
            break
        elif char == 8 or char == 127 or char == curses.KEY_BACKSPACE:
            screen.addstr('\b \b')
            try:
                del var[-1]
            except IndexError:
                continue
        elif char == curses.KEY_RIGHT:
            var.append('right')
            screen.addstr('→')
        elif char == curses.KEY_LEFT:
            var.append('left')
            screen.addstr('←')
        elif char == curses.KEY_UP:
            var.append('up')
            screen.addstr('↑')
        elif char == curses.KEY_DOWN:
            var.append('down')
            screen.addstr('↓')
        else:
            var.append(curses.keyname(char).decode('utf-8'))
            screen.addch(char)
    return ''.join(var)


def interactive(args):
    
    config_file = []
    method = []
    screen = curses.initscr()
    num_rows, num_cols = screen.getmaxyx()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)
    screen.scrollok(True)
    
    message = 'Welcome to the KiDS-GGL interactive shell!\n\n\n'
    message_len = int(len(message) / 2)
    middle = int(num_cols / 2)
    x_pos = middle - message_len
    screen.addstr(0, x_pos, message)
    screen.addstr('KiDS-pipeline$ ')
    screen.refresh()
 
    try:
        while True:
            var = []
            var = key(screen, var)
            if var == '':
                pass
            elif var == 'setup':
                screen.addstr('\nPlease enter path to the config file: ')
                config_file = key(screen, config_file)
                screen.addstr('\nPlease enter the method: ')
                method = key(screen, method)
                screen.addstr('\nPipeline configured with config file: {0}, and running in {1} mode. Closing console...'.format(config_file, method))
                screen.refresh()
                curses.napms(5000)
                break
            elif var == 'upupdowndownleftrightleftrightab':
                curses.napms(1000)
                screen.addstr('\n ___ __________________________________ ______ ______\n|   |                                  |______|      |\n|   | Nintendo®                        |      |      |\n|   | ENTERTAINMENT SYSTEM™            |      |      |\n|   |__________________________________|______|      |\n|______________________________________|______|______|\n|   |    _____   _____  |              | 1  2 |      |\n \  | o [POWER] [RESET] |              ||\ |\ |     /\n  \ |___________________|              ||_||_||    /\n   \___________________________________|______|___/\n\n')
            
                screen.addstr('\nKiDS-pipeline$ Congratulations you found the NES!')
            elif var == 'exit()' or var == 'exit':
                break
            else:
                screen.addstr('\nKiDS-pipeline$ Input not recognised, try with setup...')
            screen.addstr('\nKiDS-pipeline$ ')
            screen.refresh()
            
                        
    finally:
        curses.nocbreak()
        screen.keypad(0)
        #curses.echo()
        curses.endwin()
        
    args.config_file = config_file
    if 'cov' in method:
        args.cov = True
    if 'demo' in method:
        args.demo = True
    if 'sampler' in method:
        args.sampler = True
    if 'mock' in method:
        args.mock = True
    if 'esd' in method:
        args.esd = True
    return args


if __name__ == '__main__':
    print(0)














