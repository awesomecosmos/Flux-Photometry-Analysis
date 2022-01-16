# -*- coding: utf-8 -*-

###############################################################################
#-------------------SECTION ZERO: IMPORTING PACKAGES--------------------------#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

# importing astropy packages
from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from astropy.visualization import SqrtStretch, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import SigmaClip, sigma_clipped_stats, gaussian_fwhm_to_sigma

# importing photutils packages
from photutils.utils import calc_total_error
from photutils.aperture import SkyCircularAperture
from photutils.background import Background2D, MedianBackground
from photutils.datasets import load_spitzer_image, load_spitzer_catalog
from photutils import CircularAperture, CircularAnnulus, DAOStarFinder, aperture_photometry
from photutils.segmentation import SourceCatalog, deblend_sources, detect_threshold, make_source_mask, detect_sources

# misc packages
from copy import deepcopy
import calibrimbore as cal 
from astroquery.jplhorizons import Horizons

# path-type packages
import os
import glob
from glob import glob
from pathlib import Path

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury//ASTR480 Research/ASTR480 Code/01 Data Reduction Pipeline/DataReductionPipeline/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from drp_funcs import *
from asp_funcs import *

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury//ASTR480 Research/ASTR480 Code/02 Data Analysis/Flux-Photometry-Analysis/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from flux_cal_funcs import *

###############################################################################
#----------------------SECTION ONE: INITIALISATION----------------------------#
###############################################################################
#%%
all_mags = []
#%%
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# reading in image files from Reduced ALERTS folder
images_path = Path("//spcsfs/ave41/astro/ave41/ObsData-2021-02-18/ALERT/Reduced ALERT/WCS Calibrated/Sidereally Stacked Images")
# 

# 11, 12: fit
# 18: fits
# images_path = Path("//spcsfs/ave41/astro/ave41/ObsData-2021-02-13/ALERT/Reduced ALERT/WCS Calibrated/Sidereally Stacked Images/Images for Flux Calibration")

d = np.genfromtxt('2021-02-18.txt',delimiter=",")
x_pixel_locs = d[0,:]
y_pixel_locs = d[1,:]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

outputs_path = path_checker(images_path,'Flux and Photometry Outputs')

# filtering data files
# for proper image list
# lst_of_images = [[str(images_path) +"/" + n 
#                   for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')][-1]]
# image_names = [[n for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')][-1]]

lst_of_images = [str(images_path) +"/" + n 
                  for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]
image_names = [n for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]

target_name = 'C/2021 A7'

obsDates = []
mags_of_comet = []
flux_errs = []


for i in range(len(lst_of_images)):
    # Read the image
    x_pixel_loc = x_pixel_locs[i]
    y_pixel_loc = y_pixel_locs[i]
    
    data, hdr = fits.getdata(lst_of_images[i], header=True)
    with fits.open(lst_of_images[i], "append") as img_hdul:
        img_hdr1 = img_hdul[0].header
        target_ra = img_hdr1['RA      '].strip(' ')
        target_dec = img_hdr1['DEC     '].strip(' ')
        exptime = img_hdr1['EXPTIME ']
        obsDate = '20' + img_hdr1['DATE    '].strip(' ')
        target_location = '474'
        my_target_coords = [str(target_ra) + " " + str(target_dec)]
    
    #--------------------SECTION TWO: BACKGROUND DETECTION------------------------#
    
    # Estimate the sky background level and its standard deviation
    # mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
    bkg = get_bkg_info(data) 
    
    mask = make_source_mask(data, nsigma=2, npixels=5, dilate_size=11)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    
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
    
    
    apers = [apertures, annulus_apertures]
    phot_table = aperture_photometry(data - bkg.background, apers, method='subpixel',
                                     subpixels=5)
    
    moa_sources = pixels_to_ra_dec(lst_of_images[i],phot_table)

    # Calculate the mean flux per pixel in the annulus
    sky_mean = np.abs(moa_sources['aperture_sum_1'] / annulus_apertures.area)
    
    # Multiply this by the number of pixels in the aperture and subtract from the aperture flux measurement.
    # Put the result in a new column of the table.
    aperture_sky_sum = sky_mean * apertures.area
    moa_sources['flux'] = np.abs(moa_sources['aperture_sum_0'] - aperture_sky_sum)
    
    # flux_err = []
    # effective_gain = exptime
    # for k in range(len(moa_sources)):
    #     error = np.mean(calc_total_error(moa_sources['flux'][k], bkg.background_rms, effective_gain))
    #     moa_sources['err'] = error
    #     flux_err.append(error)
    
    #---------------------SECTION FIVE: FLUX CALIBRATION--------------------------#
    
    R_filter = 'moared.txt' #wavelengths, ?
    R_fit = cal.sauron(band=R_filter,system='skymapper',gr_lims=[-.5,0.8],plot=False,cubic_corr=False)
    
    sm_sources = skymapper_sources(my_target_coords)
    R_estimates = R_fit.estimate_mag(mags = sm_sources)
    sm_sources['MOA_R_est'] = R_estimates
    
    plt.hist(R_estimates,bins=20,color="darkviolet")
    plt.grid("both")
    plt.xlabel("apparent magnitudes")
    plt.ylabel("frequency")
    plt.title("Estimated Magnitudes of Sources in MOA-R")
    # plt.savefig(outputs_path/"est_mags-{}.jpeg".format(img_name),dpi=900)
    plt.show()
    
    def sm_to_moa_transform(sm_sources,moa_sources):
        s2 = deepcopy(sm_sources)

        dra = moa_sources['xcenter'].value[:,np.newaxis] - s2['ra'].values[np.newaxis,:]
        ddec = moa_sources['ycenter'].value[:,np.newaxis] - s2['dec'].values[np.newaxis,:]
        dist = np.sqrt(dra**2 + ddec**2)
        min_value = np.nanmin(dist,axis=0)
        min_index = np.argmin(dist,axis=0)
        
        # finding closest sources to SM sources in MOA sources
        t1 = moa_sources[min_index]
        good_indices = []
        for j in range(len(t1)):
            diff = np.abs(t1['xcenter'].value[j] - s2['ra'].values[j])
            if diff <= min_value[j]:
                good_indices.append(j)
        
        # MOA and SM source lists are now filtered to only include the matching sources
        new_t1 = t1[good_indices]
        new_s2 = s2.iloc[good_indices]
        
        m_moa = (-2.5 * np.log10(new_t1['flux']))
        # calculating zero point
        zp = new_s2['MOA_R_est'].values - m_moa
        print(np.nanmean(zp))
        
        # converting flux to mag
        for a in range(len(new_t1)):
            new_t1['mag'] = (-2.5 * np.log10(new_t1['flux'])) + zp[a]
        
        # final_calibrated_mags = new_s2['MOA_R_est'].values - new_t1['mag']
        final_calibrated_mags = new_t1['mag'] - new_s2['MOA_R_est'].values
        
        return new_t1, new_s2, zp, final_calibrated_mags
    
    new_t1, new_s2, zp, final_calibrated_mags = sm_to_moa_transform(sm_sources,moa_sources)
   
    # plt.hist(final_calibrated_mags,color="darkviolet")
    # plt.grid("both")
    # plt.xlabel("apparent magnitudes")
    # plt.ylabel("frequency")
    # plt.title("Final Calibrated Magnitudes of Sources")
    # # plt.savefig(outputs_path/"final_cal_mags-{}.jpeg".format(img_name),dpi=900)
    # plt.show()
    #%%
    # PLOTS
    plotting_funcs_flux_cal(image_names[i],sm_sources,zp,new_t1,new_s2,
                                final_calibrated_mags,outputs_path)
    
    diagnostic_plots(data,image_names[i],apertures,annulus_apertures,zp,new_t1,outputs_path)
    #%%
    # getting S/N ratios
    # xcenter_sn_ratios = []
    # for i in new_t1['xcenter'].value:
    #     xcenter_sn_ratios.append(np.abs(i)/np.sqrt(np.abs(i)))
       
    # ycenter_sn_ratios = []
    # for i in new_t1['ycenter'].value:
    #     ycenter_sn_ratios.append(np.abs(i)/np.sqrt(np.abs(i)))
        
    flux_sn_ratios = []
    for l in new_t1['flux']:
        flux_sn_ratios.append(np.abs(l)/np.sqrt(np.abs(l)))
        # flux_sn_ratios.append(l/np.sqrt(l))
    #%%
    plt.hist(flux_sn_ratios,label="flux S/N",color="darkviolet",bins=25)
    plt.axvline(3, color='deeppink', linestyle='dashed', linewidth=1,label="S/N=3")
    plt.grid(b=True,which='both',axis='both')
    # plt.title("S/N Ratios of Point Sources in {}".format(image_names[i]))
    plt.title("S/N Ratios of Point Sources")
    plt.xlabel("S/N Ratio")
    plt.ylabel("frequency")
    plt.legend()
    plt.savefig(outputs_path/"sn_ratio_hist_for_{}.jpeg".format(image_names[i]),dpi=900)
    plt.show()
    #%%
    
    #---------------------SECTION SIX: COMET PHOTOMETRY---------------------------#
    
    # do jpl horizons stuff here to find coords of comet
    t = Time(obsDate, format='fits', scale='utc')
    jd_obsDate = t.jd
    
    obj = Horizons(id=target_name, location=target_location,epochs=i)#i
    eph = obj.ephemerides()
    
    observations = Horizons(id=target_name, location=target_location,epochs=jd_obsDate)
    jpl_data = observations.ephemerides()
    
    # these are the predicted postions of  comet in this img
    comet_actual_ra = jpl_data['RA'][0]
    comet_actual_dec = jpl_data['DEC'][0]
    comet_geocentric_distance = jpl_data['delta'][0] #au
    
    # calculating arcsec size of comet to use for FWHM/aperture ring size
    epoch_comet_ang_diameter = comet_ang_size(comet_geocentric_distance,diameter=20000) # arcsec
    fwhm_diameter_in_pixels = arcsec_to_pixel(epoch_comet_ang_diameter)
    print("epoch_comet_ang_diameter: ",epoch_comet_ang_diameter)
    print("fwhm_diameter_in_pixels: ",fwhm_diameter_in_pixels)

    f = fits.open(lst_of_images[i])
    w = wcs.WCS(f[0].header)
    newf = fits.PrimaryHDU()
    
    newf.data = f[0].data[100:-100,100:-100]
    newf.header = f[0].header
    newf.header.update(w[100:-100,100:-100].to_header())
    
    
    # c = SkyCoord(comet_actual_ra, comet_actual_dec, unit="deg") 
    # jpl_ra_pixels, jpl_dec_pixels = w.world_to_pixel(c)
    # jpl_ra_pixels = (sky_ra.ra * u.deg).value
    # jpl_dec_pixels = (sky.dec * u.deg).value
    
    
    comet_pixel_location_x = x_pixel_loc
    comet_pixel_location_y = y_pixel_loc
    
    x_diffs = []
    x_diffs = [0]
    for c in range(len(x_pixel_locs)):
        try:
            # if i == 0:
            #     x_diffs.append(i)
            # else:
            # x_diffs.append(np.abs(x_pixel_locs[c]-x_pixel_locs[c+1]))
            x_diffs.append(x_pixel_locs[c]-x_pixel_locs[c+1])
        except:
            pass
    
    y_diffs = []
    y_diffs = [0]
    for d in range(len(y_pixel_locs)):
        try:
            # if i == 0:
            #     x_diffs.append(i)
            # else:
            # y_diffs.append(np.abs(y_pixel_locs[d]-y_pixel_locs[d+1]))
            y_diffs.append(y_pixel_locs[d]-y_pixel_locs[d+1])
        except:
            pass
    
    
    
    position = (comet_pixel_location_x, comet_pixel_location_y)
    
    shape = (100, 100)
    cutout = Cutout2D(f[0].data, position, shape, wcs=w)
    f.data = cutout.data
    # Update the FITS header with the cutout WCS
    f[0].header.update(cutout.wcs.to_header())
    
    # norm = simple_norm(data, 'sqrt', percent=80)
    # plt.imshow(data, norm=norm, origin='lower')
    # plt.colorbar()

    mean, median, std = sigma_clipped_stats(cutout.data, sigma=3.0)   
    bkg = get_bkg_info(cutout.data) 
    # daofind = DAOStarFinder(fwhm=fwhm_diameter_in_pixels*2, threshold=5.*std) 
    daofind = DAOStarFinder(fwhm=fwhm_diameter_in_pixels, threshold=5.*std) 
    comet_sources = daofind(cutout.data - median)
 
    centre_postion = (50+x_diffs[1],50+y_diffs[1])#postions_and_apertures(comet_sources)[1] # #misleading var name - change this to like comet_sources_positions
    apertures = CircularAperture(centre_postion, r=fwhm_diameter_in_pixels)
    # apertures = CircularAperture(centre_postion, r=fwhm_diameter_in_pixels)
    annulus_apertures = CircularAnnulus(centre_postion, r_in=fwhm_diameter_in_pixels*2, r_out=fwhm_diameter_in_pixels*3)
    # annulus_apertures = CircularAnnulus(centre_postion, r_in=fwhm_diameter_in_pixels*2, r_out=fwhm_diameter_in_pixels*3)

    #%%
    norm = simple_norm(cutout.data, 'sqrt', percent=80)
    plt.imshow(cutout.data, norm=norm, origin='lower')
    # norm = simple_norm(data, 'sqrt', percent=80)
    # plt.imshow(data, norm=norm, origin='lower')
    plt.colorbar()
    apertures.plot(color='white', lw=2)
    annulus_apertures.plot(color='red', lw=2)
    plt.title("Cutout of {} on {}".format(target_name,obsDate))
    # plt.savefig(outputs_path/"cutout_with_annulus_{}".format(i))
    plt.show()
    #%%
    
    # for i in range(len(centre_postion)):
    #     coords_to_check = apertures.positions[i]
    #     if 
    apers = [apertures, annulus_apertures]
    phot_table = aperture_photometry(cutout.data - bkg.background, apers, method='subpixel',
                                      subpixels=5)
    
    comet_sources2 = pixels_to_ra_dec(lst_of_images[i],phot_table)

    # Calculate the mean flux per pixel in the annulus
    sky_mean = np.abs(comet_sources2['aperture_sum_1'] / annulus_apertures.area)
    
    # Multiply this by the number of pixels in the aperture and subtract from the aperture flux measurement.
    # Put the result in a new column of the table.
    aperture_sky_sum = sky_mean * apertures.area
    comet_sources2['flux'] = np.abs(comet_sources2['aperture_sum_0'] - aperture_sky_sum)
    
    
    
    # dra = comet_sources['xcentroid'][:,np.newaxis] - comet_pixel_location_x
    # ddec = comet_sources['ycentroid'][:,np.newaxis] - comet_pixel_location_y
    dra = comet_sources2['xcenter'][:,np.newaxis].value - comet_pixel_location_x#comet_sources['xcentroid'][:,np.newaxis][:]
    ddec = comet_sources2['ycenter'][:,np.newaxis].value - comet_pixel_location_y#comet_sources['ycentroid'][:,np.newaxis][:]
    dist = np.sqrt(dra**2 + ddec**2)
    min_value = np.nanmin(dist,axis=0)
    min_index = np.argmin(dist,axis=0)
    
    comet_index = min_index
    # comet_max_flux = comet_sources['flux'][comet_index]

    # comet_mag = comet_sources['mag'][comet_index]
    # mag_of_comet_cal = comet_mag + np.nanmean(zp)
    # print(mag_of_comet_cal)
    
    effective_gain = exptime
    for k in range(len(comet_sources2)):
        error = np.mean(calc_total_error(comet_sources2['flux'][k], bkg.background_rms, effective_gain))
        comet_sources2['err'] = error
    
    
    
    # moa_mag =  (-2.5 * np.log10(comet_sources2['flux'][comet_index]))
    # moa_mag_uncert = (-2.5 * np.log10(comet_sources['err'][comet_index])) + np.nanmean(zp)
    flux_of_comet = comet_sources2['flux'][comet_index]
    flux_err_of_comet = comet_sources2['err'][comet_index]
    mag_of_comet = (-2.5 * np.log10(comet_sources2['flux'][comet_index])) + np.nanmedian(zp)
    moa_mag_uncert = np.abs(-2.5*(mag_of_comet * np.sqrt((flux_err_of_comet/flux_of_comet)**2))) #via error propogation)
    #(-2.5 * np.log10(comet_sources2['err'][comet_index])) + np.nanmedian(zp)
    print("mag_of_comet: ",mag_of_comet[0],"+/-",moa_mag_uncert[0])
    
    
    # plt.plot(obsDate,mag_of_comet)

    obsDates.append(obsDate)
    mags_of_comet.append(mag_of_comet[0])
    flux_errs.append(moa_mag_uncert[0])
    
all_mags.append([obsDates,mags_of_comet,flux_errs])   


# plt.plot(obsDates,mags_of_comet,'.')

# uncert = np.std(mags_of_comet) / np.sqrt(len(mags_of_comet))

# for i in range(len(flux_errs)):
#     plt.errorbar(obsDates,mags_of_comet,flux_errs[i][0],fmt='.')
#     # plt.errorbar(obsDates,mags_of_comet,uncert,fmt='.',color="deeppink")
    
    


#%%


# new_lst = [all_mags[0],all_mags[1],all_mags[10],all_mags[11]]

fig, ax = plt.subplots(1,1) #plt.subplots(figsize=(15, 10))
# for m in new_lst:
#     plt.plot(m[0],m[1],'.',label=m[0][1][:10])
#     # plt.errorbar(m[0],m[1],flux_errs[0][0],fmt='.')
for m in all_mags:
    # plt.plot(m[0],m[1],'.',label=m[0][1][:10])
    plt.errorbar(m[0],m[1],m[2][0][0],fmt='.',label=m[0][1][:10])
ax.grid("both")
# ax.invert_xaxis()
plt.xticks(rotation=90)
plt.legend(loc=0)
plt.xlabel("epoch")
plt.ylabel("apparent magnitude")
plt.title("Light Curve of C/2021 A7")
plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=1)
plt.savefig("final_lightcurve.jpeg",dpi=900)
    