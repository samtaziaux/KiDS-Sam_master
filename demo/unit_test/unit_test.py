#!/usr/bin/python

"""
This utility function calculates the radial diameter distance 'Da' from the
redshift 'z' for a given matter and lambda density 'Om' and 'Ol'). Multiplying
Da by an opening angle 'theta' on the sky will give its actual size 'x'.

"""

import astropy.io.fits as pyfits
from scipy import integrate
import numpy as np
import sys
import glob
from astropy import constants as const, units as u
from kids_ggl_pipeline.esd_production import distance
import matplotlib.pyplot as pl

c = const.c.value
G = const.G.value
G_Mpc = G = const.G.to('pc3/Msun s2')


def write_gama_like_cat(ra, dec, z):
    
    id = 1
    header = ['ID', 'RA', 'DEC', 'Z', 'lim']
    data = [np.array([id]), np.array([ra]), np.array([dec]), np.array([z]), np.array([1.0])]

    cols = []
    for i in xrange(len(header)):
        if header[i] == 'ID':
            cols.append(
                    pyfits.Column(name='%s'%(header[i]), format='J', array=data[i])
                    )
        else:
            cols.append(
                    pyfits.Column(name='%s'%(header[i]), format='D', array=data[i])
                    )


    print('\nWriting GAMA like catalog ...')

    new_cols = pyfits.ColDefs(cols)
    hdu = pyfits.BinTableHDU.from_columns(new_cols)
    hdu.writeto('unit_test_GAMA_cat.fits', clobber=True)
    

    return 0


def write_kids_like_cat(RA, DEC, Z_B, g1, g2, RA_cen, DEC_cen):
    
    RA, DEC, Z_B, g1, g2 = np.array([RA]).T, np.array([DEC]).T, np.array([Z_B]).T, np.array([g1]).T, np.array([g2]).T

    dim = len(RA)
    
    
    theli = 'KIDS_%.1f_%.1f'%(RA_cen, DEC_cen)
    theli = (theli.replace('.','p')).replace('-','m')
    tile = np.array([theli]*dim)

    SeqNr = np.arange(dim)
    weight=np.ones(dim)

    SN=np.empty(dim)
    SN.fill(100.)

    MASK=np.empty(dim)
    MASK.fill(0)

    cols = []
    cols.append(pyfits.Column(name='SeqNr', format='D', array=SeqNr))
    cols.append(pyfits.Column(name='ALPHA_J2000', format='D', array=RA))
    cols.append(pyfits.Column(name='DELTA_J2000', format='D', array=DEC))
    cols.append(pyfits.Column(name='e1_A', format='D', array=g1))
    cols.append(pyfits.Column(name='e2_A', format='D', array=g2))
    cols.append(pyfits.Column(name='e1_B', format='D', array=g1))
    cols.append(pyfits.Column(name='e2_B', format='D', array=g2))
    cols.append(pyfits.Column(name='e1_C', format='D', array=g1))
    cols.append(pyfits.Column(name='e2_C', format='D', array=g2))
    cols.append(pyfits.Column(name='weight_A', format='D', array=weight))
    cols.append(pyfits.Column(name='weight_B', format='D', array=weight))
    cols.append(pyfits.Column(name='weight_C', format='D', array=weight))
    cols.append(pyfits.Column(name='Z_B', format='D', array=Z_B))
    cols.append(pyfits.Column(name='model_SNratio', format='D', array=SN))
    cols.append(pyfits.Column(name='MASK', format='D', array=MASK))
    cols.append(pyfits.Column(name='THELI_NAME', format='16A', array=tile))

    print('\nWriting KiDS like catalog, containing %i objects ...'%len(SeqNr))
    
    new_cols = pyfits.ColDefs(cols)
    hdu_new = pyfits.BinTableHDU.from_columns(new_cols)
    outhdu = pyfits.BinTableHDU.from_columns(new_cols)
    #outhdu.name='OBJECTS'
    hdulist = pyfits.HDUList([pyfits.PrimaryHDU(), outhdu, hdu_new])
    hdulist.insert(3, outhdu)
    hdulist.writeto('KiDS_cat/unit_test_KiDS_cat.fits', clobber=True)
    
    return 0


def sis(sigma, lens_args, source_args, d_s, d_ls, d_l):
    
    print('\nCreating one SIS lens ...')
    
    xpos, ypos, redshift = lens_args
    source_xpos, source_ypos, source_redshift = source_args
    
    theta = np.sqrt((source_xpos-xpos)**2.0 + (source_ypos-ypos)**2.0)
    angle = np.arctan((source_ypos-ypos)/(source_xpos-xpos))
    
    d_ls = d_ls/(1.0 + source_redshift-redshift)
    d_s = d_s/(1.0 + source_redshift)
    distances_ratio = d_ls / d_s
    
    deg_per_radian = 180./np.pi


    profile = lambda x: (4.0*np.pi*sigma**2.0 * distances_ratio)/(c**2.0 * 2.0 * x)
    gamma_t = profile(theta) * deg_per_radian
    
    out = -gamma_t*np.exp(2.j*angle)
    
    # Plotting and saving analytical forms to compare the pipeline results to.
    
    lin_range = 10.0**np.linspace(np.log10(0.01), np.log10(20),len(theta))
    
    gamma_t_int = (4.0*np.pi*sigma**2.0 * np.median(distances_ratio))/(c**2.0* 2.0 * lin_range) * deg_per_radian * 60.0 # in arcmins!
    delta_sigma = (sigma * 3.241 * 10**(-20))**2.0 / (2.0 * G_Mpc.value * lin_range) # in pc^2/Msun, comoving - as usual.
    
    # To physical units, as this is what pipeline returns!
    delta_sigma = delta_sigma*(1.0+redshift)**2.0
    
    pl.plot(lin_range, gamma_t_int)
    #index = np.argsort(theta)
    #pl.plot(theta[index], gamma_t[index])
    pl.xscale('log')
    pl.yscale('log')
    pl.savefig('gamma_t_sis.pdf')
    #pl.show()
    pl.close()
    #quit()
    
    pl.plot(lin_range, delta_sigma)
    pl.xscale('log')
    pl.yscale('log')
    pl.savefig('delta_sigma_sis.pdf')
    #pl.show()
    pl.close()
    
    shear_data = np.vstack((lin_range, gamma_t_int)).T
    sigma_data = np.vstack((lin_range, delta_sigma)).T
    np.savetxt('gamma_t_sis.dat', shear_data)
    np.savetxt('delta_sigma_sis.dat', sigma_data)

    return out


def create_kids_field(n_sources, center_of_field, lens_args):
    
    print('\nCreating KiDS field ...')
    
    lenses_xpos, lenses_ypos, r = lens_args
    
    sources_xpos = np.random.uniform(-0.5 + center_of_field[0], 0.5 + center_of_field[0], n_sources)
    sources_ypos = np.random.uniform(-0.5 + center_of_field[1], 0.5 + center_of_field[1], n_sources)
    nz = np.genfromtxt('data_nz.txt')[:,0]
    nz = nz/nz.sum()
    sources_redshift = np.random.choice(np.linspace(0.0, 3.5, len(nz)), p=nz, size=n_sources)
    
    # This was here to exclude the strong lensing regime. Not really needed though ...
    
    #strong = np.where(np.sum(np.sqrt((sources_xpos-lenses_xpos)**2.0 +
    #           (sources_ypos-lenses_ypos)**2.0) < 2.0*3600.0*2.0, axis=0) > 0.)
    #sources_xpos = np.delete(sources_xpos, strong)
    #sources_ypos = np.delete(sources_ypos, strong)
    #sources_redshift = np.delete(sources_redshift, strong)
    
    
    return sources_xpos, sources_ypos, sources_redshift


def shear_map(sigma, lens_args, source_args, cosmo):
    
    print('\nCalculating shear map ...')
    
    O_matter, O_lambda, h = cosmo
    
    xpos, ypos, redshift = lens_args
    source_xpos, source_ypos, source_redshift = source_args
    
    source_comoving = np.array([distance.comoving(y, O_matter, O_lambda, h)
                                for y in source_redshift])
    lens_source_comoving = np.array([distance.comoving(y, O_matter, O_lambda, h)
                                     for y in source_redshift-redshift])
    lens_comoving = distance.comoving(redshift, O_matter, O_lambda, h)
    lens_angular = lens_comoving/(1.0+redshift)

    
    ellip = sis(sigma, lens_args, source_args, source_comoving, l
                ens_source_comoving, lens_angular)
    

    ellip_x = 2.0*np.real(np.sqrt(np.absolute(ellip) * ellip))
    ellip_y = 2.0*np.imag(np.sqrt(np.absolute(ellip) * ellip))
    
    ellip_x[np.isnan(ellip_x)] = 0.0
    ellip_y[np.isnan(ellip_y)] = 0.0
    
    
    N = np.sqrt(ellip_x**2.0 + ellip_y**2.0)
    
    pl.quiver(source_xpos, source_ypos, ellip_x, ellip_y, headwidth=0,
              headlength=0, headaxislength=0, pivot='middle')
    pl.title('Shear map')
    pl.xlabel('x [deg]')
    pl.ylabel('y [deg]')
    pl.show()
    pl.close()
    
    
    return source_xpos, source_ypos, source_redshift, ellip_x, -ellip_y


if __name__ == '__main__':

    n_sources = 100700 # around KiDS density, but not really important.

    sigma = 600.0 * 1000.0 # in m/s
    
    omega_m = 0.315
    omega_v = 0.685
    h = 1.0
    cosmo = omega_m, omega_v, h
    
    ra = 0.0
    dec = 0.0
    z = 0.2
    
    center_of_field = 180.0, 1.5
    
    # What is missing here is the translations on the sphere, so we can properly test
    # out the implementation of speherical geometry in the pipeline
    
    lens_args = center_of_field[0]+ra, center_of_field[1]+dec, z
    
    source_args = create_kids_field(n_sources, center_of_field, lens_args)
    
    RA, DEC, Z_B, g1, g2 = shear_map(sigma, lens_args, source_args, cosmo)

    write_kids = write_kids_like_cat(RA, DEC, Z_B, g1, g2,
                                     center_of_field[0]+ra,
                                     center_of_field[1]+dec)
    write_gama = write_gama_like_cat(center_of_field[0], center_of_field[1], z)

    quit()
    
