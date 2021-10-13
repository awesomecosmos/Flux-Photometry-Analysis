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
    return apertures, positions


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