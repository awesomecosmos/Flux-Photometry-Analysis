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

# misc packages
import calibrimbore as cal 
from copy import deepcopy

# path-type packages
import os
import glob
from glob import glob
from pathlib import Path

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury/MSc Astronomy/MSc 2021/ASTR480 Research/ASTR480 Code/01 Data Reduction Pipeline/DataReductionPipeline/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from drp_funcs import *
from asp_funcs import *

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury/MSc Astronomy/MSc 2021/ASTR480 Research/ASTR480 Code/02 Data Analysis/Flux-Photometry-Analysis/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from flux_and_photometry_funcs import *

###############################################################################
#----------------------SECTION ONE: INITIALISATION----------------------------#
###############################################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# reading in image files from Reduced ALERTS folder
images_path = Path("//spcsfs/ave41/astro/ave41/ObsData-2021-02-13/ALERT/Reduced ALERT/WCS Calibrated/Sidereally Stacked Images")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# filtering data files
lst_of_images = [str(images_path) +"/" + n 
                  for n in os.listdir(images_path) if (n.endswith('fit')) and n.__contains__('-aligned-')]
image_names = [n for n in os.listdir(images_path) if (n.endswith('fit')) and n.__contains__('-aligned-')]
outputs_path = path_checker(images_path,'Flux and Photometry Outputs')


for i in range(len(lst_of_images)):
    # Read the image
    data, hdr = fits.getdata(lst_of_images[i], header=True)
    with fits.open(lst_of_images[i], "append") as img_hdul:
        img_hdr1 = img_hdul[0].header
        target_ra = img_hdr1['RA      '].strip(' ')
        target_dec = img_hdr1['DEC     '].strip(' ')
        my_target_coords = [str(target_ra) + " " + str(target_dec)]
    
    #--------------------SECTION TWO: BACKGROUND DETECTION------------------------#
    
    # Estimate the sky background level and its standard deviation
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
    bkg = get_bkg_info(data) 
    
    #---------------------SECTION THREE: SOURCE DETECTION-------------------------#
    
    # Start up the DAOStarFinder object and detect stars
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)    
    moa_sources = daofind(data - median)

    #------------------------SECTION FOUR: APERTURES------------------------------#
    
    apertures, positions = postions_and_apertures(moa_sources)
    
    # Set up a set of circular apertures (one for each position) with a radius 
    # of 5 pixels and annuli with
    # inner and outer radii of 10 and 15 pixels.
    apertures = CircularAperture(positions, r=5)
    annulus_apertures = CircularAnnulus(positions, r_in=10, r_out=15)
    
    #---------------------SECTION FIVE: FLUX CALIBRATION--------------------------#
    
    moa_sources = pixels_to_ra_dec(lst_of_images[i],moa_sources)
    
    # RYAN'S CODE
    R_filter = 'moared.txt' #wavelengths, ?
    R_fit = cal.sauron(band=R_filter,system='skymapper',gr_lims=[-.5,0.8],plot=True,cubic_corr=False)
    
    sm_sources = skymapper_sources(my_target_coords)
    R_estimates = R_fit.estimate_mag(mags = sm_sources)
    sm_sources['MOA_R_est'] = R_estimates
    
    new_t1, new_s2, zp, final_calibrated_mags = sm_to_moa_transform(sm_sources,moa_sources)
    
    # PLOTS
    plt.figure()
    plt.plot(sm_sources['g']-sm_sources['r'],sm_sources['MOA_R_est']-sm_sources['r'],'.',color="darkviolet")
    plt.xlabel("g-r")
    plt.ylabel("MOA-R - r")
    plt.grid("both")
    plt.title("(g-r) vs (MOA-R - r)")
    plt.savefig(outputs_path/"sm_sources_colour-{}.jpeg".format(image_names[i]),dpi=900)
    plt.show()
    
    plt.hist(zp,bins=100,color="darkviolet")
    plt.grid("both")
    plt.xlabel("zero points")
    plt.ylabel("frequency")
    plt.title("Calibrated Zero Points of Sources")
    plt.savefig(outputs_path/"cal_zp-{}.jpeg".format(image_names[i]),dpi=900)
    plt.show()
        
    plt.hist(new_t1['mag'],bins=100,color="darkviolet")
    plt.grid("both")
    plt.xlabel("apparent magnitudes")
    plt.ylabel("frequency")
    plt.title("Calibrated Apparent Magnitudes of Sources")
    plt.savefig(outputs_path/"hist_cal_mags-{}.jpeg".format(image_names[i]),dpi=900)
    plt.show()
    
    plt.plot(new_t1['flux'],new_t1['mag'],'.',color="darkviolet")
    plt.grid("both")
    plt.xlabel("flux")
    plt.ylabel("magnitude")
    plt.title("Calibrated Apparent Magnitudes of Sources")
    plt.savefig(outputs_path/"flux_vs_cal_mags-{}.jpeg".format(image_names[i]),dpi=900)
    plt.show()
    
    plt.hist(final_calibrated_mags,bins=50,color="darkviolet")
    plt.grid("both")
    plt.xlabel("apparent magnitudes")
    plt.ylabel("frequency")
    plt.title("Final Calibrated Magnitudes of Sources")
    plt.savefig(outputs_path/"final_cal_mags-{}.jpeg".format(image_names[i]),dpi=900)
    plt.show()
    
    plt.figure()
    plt.plot(new_s2['g']-new_s2['r'],final_calibrated_mags,'.',color="darkviolet")
    plt.xlabel("g-r")
    plt.ylabel("magnitudes")
    plt.grid("both")
    plt.title("(g-r) vs magnitudes")
    plt.savefig(outputs_path/"final_cal_colour-{}.jpeg".format(image_names[i]),dpi=900)
    plt.show()