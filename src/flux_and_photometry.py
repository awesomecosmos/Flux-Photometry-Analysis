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

# filtering data files
image_names = [n for n in os.listdir(images_path) if (n.endswith('fit')) and n.__contains__('-aligned-')]

# lst_of_images = []
# for n in os.listdir(images_path):
#     if (n.endswith('fit')) and n.__contains__('-aligned-'):
#         lst_of_images.append(str(images_path) +"/" + n)

outputs_path = path_checker(images_path,'Flux and Photometry Outputs')

# put in loop[]
# Read the image
data, hdr = fits.getdata(lst_of_images[0], header=True)

#%%
###############################################################################
#--------------------SECTION TWO: BACKGROUND DETECTION------------------------#
###############################################################################

# Estimate the sky background level and its standard deviation
mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
bkg = get_bkg_info(data) 

#%%
###############################################################################
#---------------------SECTION THREE: SOURCE DETECTION-------------------------#
###############################################################################

# Start up the DAOStarFinder object and detect stars
daofind = DAOStarFinder(fwhm=5.0, threshold=5.*std)    
sources = daofind(data - median)

#%%
###############################################################################
#------------------------SECTION FOUR: APERTURES------------------------------#
###############################################################################

apertures, positions = postions_and_apertures(sources)

# Set up a set of circular apertures (one for each position) with a radius of 5 pixels and annuli with
# inner and outer radii of 10 and 15 pixels.
apertures = CircularAperture(positions, r=5)
annulus_apertures = CircularAnnulus(positions, r_in=10, r_out=15)

#%%
###############################################################################
#------------------------SECTION FIVE: PHOTOMETRY-----------------------------#
###############################################################################

# Measure the total flux in both the aperture and annulus for each star. 
apers = [apertures, annulus_apertures]
phot_table = aperture_photometry(data - bkg.background, apers, method='subpixel',
                                 subpixels=5)

# Calculate the mean flux per pixel in the annulus
sky_mean = phot_table['aperture_sum_1'] / annulus_apertures.area

# Multiply this by the number of pixels in the aperture and subtract from the aperture flux measurement.
# Put the result in a new column of the table.
aperture_sky_sum = sky_mean * apertures.area
phot_table['flux'] = phot_table['aperture_sum_0'] - aperture_sky_sum

# setting zero point and adjusting magnitudes
# Magnitude zero point is arbitrary
ZP = 30
phot_table['mag'] = ZP - 2.5*np.log10(phot_table['flux'])

print(phot_table)
#%%
# getting plots
diagnostic_plots(data,image_names[0],apertures,annulus_apertures,ZP,phot_table,outputs_path)

#%%
# annulus stuff
annulus_masks = annulus_apertures.to_mask(method='center')
plt.imshow(annulus_masks[0], interpolation='nearest')
plt.colorbar()
annulus_data = annulus_masks[0].multiply(data)
mask = annulus_masks[0].data
annulus_data_1d = annulus_data[mask > 0]
print(annulus_data_1d.shape)
_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
print(median_sigclip)  

# background subtraction stuff
bkg_median = []
for mask in annulus_masks:
    annulus_data = mask.multiply(data)
    annulus_data_1d = annulus_data[mask.data > 0]
    _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
    bkg_median.append(median_sigclip)
bkg_median = np.array(bkg_median)

# Spitzer catalog stuff
hdu = load_spitzer_image()  
data = u.Quantity(hdu.data, unit=hdu.header['BUNIT'])  
wcs = WCS(hdu.header)  
catalog = load_spitzer_catalog() 
positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')  
aperture = SkyCircularAperture(positions, r=4.8 * u.arcsec) 

# error stuff
error = 0.1 * data

#%%
phot_table = aperture_photometry(data, apertures, error=error, wcs=wcs)
phot_table['annulus_median'] = bkg_median
phot_table['aper_bkg'] = bkg_median * apertures.area
phot_table['aper_sum_bkgsub'] = phot_table['aperture_sum'].value - phot_table['aper_bkg']
factor = (1.2 * u.arcsec) ** 2 / u.pixel
fluxes_catalog = catalog['f4_5']  
converted_aperture_sum = (phot_table['aperture_sum'] * factor).to(u.mJy / u.pixel) 
print(phot_table)

#%%





























