# -*- coding: utf-8 -*-

###############################################################################
#-------------------SECTION ZERO: IMPORTING PACKAGES--------------------------#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

# importing astropy packages
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize

# importing photutils packages
from photutils.aperture import SkyCircularAperture
from photutils import CircularAperture, CircularAnnulus
from photutils import DAOStarFinder, aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.datasets import load_spitzer_image, load_spitzer_catalog

# path-type packages
import os
import glob
from pathlib import Path

# misc packages
import calibrimbore as cal 
from copy import deepcopy

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury/MSc Astronomy/MSc 2021/ASTR480 Research/ASTR480 Code/01 Data Reduction Pipeline/DataReductionPipeline/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from drp_funcs import *
from asp_funcs import *

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury/MSc Astronomy/MSc 2021/ASTR480 Research/ASTR480 Code/02 Data Analysis/Flux-Photometry-Analysis/src"
os.chdir(code_home_path) #from now on, we are in this directory

###############################################################################
#--------------------SECTION TWO: BACKGROUND DETECTION------------------------#
###############################################################################

def get_bkg_info(data):
   # creating a background object
   sigma_clip = SigmaClip(sigma=3.)
   bkg_estimator = MedianBackground()
   bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                      sigma_clip=sigma_clip, bkg_estimator=bkg_estimator) 
   return bkg

###############################################################################
#------------------------SECTION FOUR: APERTURES------------------------------#
###############################################################################

def postions_and_apertures(sources):
    # creating list of sources
    try:
        positions = (sources['xcentroid'], sources['ycentroid'])
        the_lst = []
        x_vals = sources['xcentroid']
        y_vals = sources['ycentroid']
        for index in range(len(x_vals)):
            to_return = (x_vals[index],y_vals[index])
            the_lst.append(to_return)
        positions = (sources['xcentroid'], sources['ycentroid'])
        apertures = CircularAperture(the_lst, r=20.0)
        positions = the_lst
    except KeyError:
        positions = (sources['ra'], sources['dec'])
        the_lst = []
        x_vals = sources['ra']
        y_vals = sources['dec']
        for index in range(len(x_vals)):
            to_return = (x_vals.to_numpy()[index],y_vals.to_numpy()[index])
            the_lst.append(to_return)
        positions = (sources['ra'], sources['dec'])
        apertures = CircularAperture(the_lst, r=20.0)
        positions = the_lst
    return apertures, positions

def pixels_to_ra_dec(image,sources):
    f = fits.open(image)
    w = WCS(f[0].header)
    ra_dec_sources = []
    for i in range(len(sources['xcentroid'])):
        sky = w.pixel_to_world(sources['xcentroid'][i],sources['ycentroid'][i])
        sources['xcentroid'][i] = (sky.ra * u.deg).value
        sources['ycentroid'][i] = (sky.dec * u.deg).value
    return sources

def skymapper_sources(my_target_coords):
    coords = my_target_coords 
    c = SkyCoord(coords,unit=(u.hourangle, u.deg))
    try:
        sm_sources = cal.get_skymapper_region(c.ra.deg,c.dec.deg)
    except:
        sm_sources = cal.get_skymapper_region(c.ra.deg,c.dec.deg,size=0.4*60**2)
    
    ind = (np.isfinite(sm_sources.r.values) & np.isfinite(sm_sources.i.values) 
           & np.isfinite(sm_sources.z.values) & (sm_sources['g'].values < 19) & (sm_sources['g'].values > 13))
    sm_sources = sm_sources.iloc[ind]
    return sm_sources

def sm_to_moa_transform(sm_sources,moa_sources):
    s2 = deepcopy(sm_sources)
    
    dra = moa_sources['xcentroid'][:,np.newaxis] - s2['ra'].values[np.newaxis,:]
    ddec = moa_sources['ycentroid'][:,np.newaxis] - s2['dec'].values[np.newaxis,:]
    dist = np.sqrt(dra**2 + ddec**2)
    min_value = np.nanmin(dist,axis=0)
    min_index = np.argmin(dist,axis=0)

    t1 = moa_sources[min_index]
    
    good_indices = []
    for i in range(len(t1)):
        if np.abs(t1['xcentroid'][0] - s2['ra'].values[0]) >= min_value[i]:
            good_indices.append(i)

    new_t1 = t1[good_indices]
    new_s2 = s2.iloc[good_indices]
    
    # calculating zero point
    zp = new_s2['MOA_R_est'].values - new_t1['flux']
    # zp = new_t1['flux'] - new_s2['MOA_R_est'].values
    
    # converting flux to mag
    for i in range(len(new_t1)):
        new_t1['mag'] = zp[i] - (2.5*np.log10(new_t1['flux']))
    
    final_calibrated_mags = new_s2['MOA_R_est'].values - new_t1['mag']
    
    return new_t1, new_s2, zp, final_calibrated_mags

###############################################################################
###############################----PLOTS----###################################
###############################################################################

def diagnostic_plots(data,image_name,apertures,annulus_apertures,ZP,phot_table,outputs_path):
    # Display the image and overlay the apertures and anulli - zoom in to check that they are OK 
    plt.figure()
    norm = simple_norm(data, 'sqrt', percent=96) # another way to "stretch" the image display
    plt.imshow(data, norm=norm,origin='lower')
    plt.colorbar()
    apertures.plot(color='deeppink', lw=2)
    annulus_apertures.plot(color='red', lw=2)
    plt.xlim(1500, 1600)
    plt.ylim(2000, 2100)
    plt.title("Zoomed-In Apertures for {}".format(image_name))
    plt.savefig(outputs_path/"apertures-{}.jpeg".format(image_name),dpi=900)
    plt.show()

    # plotting stats
    plt.figure()
    plt.hist(phot_table['mag'],color="darkviolet")
    plt.grid(b=True,which='both',axis='both')
    plt.xlabel("Apparent Magnitude")
    plt.ylabel("Number of Stars")
    plt.title("Histogram of Apparent Magnitudes for Stars with ZP {} for {}".format(ZP,image_name))
    plt.savefig(outputs_path/"mags_with_zp_for_{}.jpeg".format(image_name),dpi=900)
    plt.show()
    
    plt.plot(phot_table['flux'],phot_table['mag'],'.',color="darkviolet")
    plt.grid(b=True,which='both',axis='both')
    plt.xlabel("Flux")
    plt.ylabel("Apparent Magnitude")
    plt.title("Flux vs Magnitude of Apertures for {}".format(image_name))
    plt.savefig(outputs_path/"flux_vs_mag_for_{}.jpeg".format(image_name),dpi=900)
    plt.show()
    
    plt.hist(phot_table['ycenter'].value,bins=50,color="darkviolet")
    plt.grid(b=True,which='both',axis='both')
    plt.xlabel("Flux")
    plt.ylabel("frequency")
    plt.title("Flux Values for {}".format(image_name))
    plt.savefig(outputs_path/"flux_vals_for_{}.jpeg".format(image_name),dpi=900)
    plt.show()
    
    # getting S/N ratios
    xcenter_sn_ratios = []
    for i in phot_table['xcenter'].value:
        xcenter_sn_ratios.append(i/np.sqrt(i))
       
    ycenter_sn_ratios = []
    for i in phot_table['ycenter'].value:
        ycenter_sn_ratios.append(i/np.sqrt(i))
    
    plt.hist(xcenter_sn_ratios,label="xcenter",color="deeppink")
    plt.grid(b=True,which='both',axis='both')
    plt.hist(ycenter_sn_ratios,alpha=0.5,label="ycenter",color="darkviolet")
    plt.title("S/N Ratios of Point Sources in {}".format(image_name))
    plt.xlabel("S/N Ratio")
    plt.ylabel("frequency")
    plt.legend()
    plt.savefig(outputs_path/"sn_ratio_hist_for_{}.jpeg".format(image_name),dpi=900)
    plt.show()