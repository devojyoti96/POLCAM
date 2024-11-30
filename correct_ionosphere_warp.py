import os, glob, logging, gc, warnings, psutil
from casatasks import imsubimage, exportfits
from astropy.io import fits
from basic_func import make_stokes_cube
from joblib import Parallel, delayed
os.system("rm -rf casa*log")
warnings.filterwarnings("ignore")


def estimate_warp_map(imagename, allsky_cat="GGSM.fits"):
    """
    Parameters
    ----------
    imagename : str
        Name of the image to calculate ionospheric warp screen
    allsky_cat : str
        All sky fits file name
    Returns
    -------
    str
        Unwarped catalog fits file
    """
    image_prefix = imagename.split(".fits")[0]
    if os.path.exists(image_prefix + "_I.image"):
        os.system("rm -rf " + image_prefix + "_I.image")
    imsubimage(
        imagename=imagename, outfile=image_prefix + "_I.image", stokes="I", dropdeg=True
    )
    if os.path.exists(image_prefix + "_I.fits"):
        os.system("rm -rf " + image_prefix + "_I.fits")
    exportfits(
        imagename=image_prefix + "_I.image",
        fitsimage=image_prefix + "_I.fits",
        dropdeg=True,
    )
    os.system("rm -rf " + image_prefix + "_I.image")
    I_imagename = image_prefix + "_I.fits"
    I_image_prefix = I_imagename.split(".fits")[0]
    print("#########################")
    print("Estimating noise map using BANE...\n")
    bane_cmd = "BANE " + I_imagename
    print(bane_cmd + "\n")
    print("#########################")
    os.system(bane_cmd + '>tmp')
    rms_image = I_image_prefix + "_rms.fits"
    bkg_image = I_image_prefix + "_bkg.fits"
    print("#########################")
    print("Source finding using AEGEAN...\n")
    aegean_cmd = (
        "aegean --noise "
        + rms_image
        + " --background "
        + bkg_image
        + " "
        + I_imagename
        + " --table "
        + image_prefix
        + "_catalog.fits"
    )
    print(aegean_cmd + "\n")
    print("#########################")
    os.system(aegean_cmd + '>tmp')
    print("#########################")
    print("Ionospheric correction using fits_warp...\n")
    source_catalog = image_prefix + "_catalog_comp.fits"
    fitswarp_cmd = (
        "fits_warp.py --incat "
        + source_catalog
        + " --refcat "
        + allsky_cat
        + " --xmcat "
        + image_prefix
        + "_xm.fits --infits "
        + I_imagename
    )
    print(fitswarp_cmd + "\n")
    print("#########################")
    os.system(fitswarp_cmd + '>tmp')
    os.system("rm -rf tmp")
    xm_fits = image_prefix + "_xm.fits"
    os.system(
        "rm -rf "
        + source_catalog
        + " "
        + bkg_image
        + " "
        + rms_image
        + " "
        + I_image_prefix
        + "*"
    )
    gc.collect()
    return xm_fits


def correct_warp(imagename, xmfits, ncpu=-1, keep_original=True):
    """
    Parameters
    ----------
    imagename : str
        Name of the fits image
    xmfits : str
        Unwarped catalog fits file
    ncpu : int
        Number of cpu threads to use    
    keep_original : bool
        Keep original image or replace it
    Returns
    -------
    str
        Unwraped fits image
    """
    if ncpu<0:
        ncpu=int(psutil.cpu_cpunt()*(100-psutil.cpu_percent())/100.0)
    def run_fits_warp(xmfits,temp_image,ncpu):
        image_prefix=temp_image.split('.fits')[0]
        print(
            "Ionospheric correction using fits_warp: "
            + os.path.basename(xmfits)
            + " on image: "
            + os.path.basename(temp_image)
            + "\n"
        )
        fitswarp_cmd = (
            "fits_warp.py --xm " + xmfits + " --infits " + temp_image + " --cores "+str(ncpu)+" --suffix unwarped"
        )
        os.system(fitswarp_cmd+ '>tmp')
        unwarped_file = image_prefix+ "_unwarped.fits"
        os.system("rm -rf tmp "+temp_image)
        return unwarped_file
    image_prefix = imagename.split(".fits")[0]
    header=fits.getheader(imagename)
    if header['NAXIS3']==4 or header['NAXIS4']==4: 
        os.system("rm -rf "+image_prefix+'-I-image.image '+image_prefix+'-I-image.fits') 
        imsubimage(imagename=imagename,outfile=image_prefix+'-I-image.image',stokes='I',dropdeg=False)
        exportfits(imagename=image_prefix+'-I-image.image',fitsimage=image_prefix+'-I-image.fits',dropdeg=False)
        os.system("rm -rf "+image_prefix+'-I-image.image')
        os.system("rm -rf "+image_prefix+'-Q-image.image '+image_prefix+'-Q-image.fits') 
        imsubimage(imagename=imagename,outfile=image_prefix+'-Q-image.image',stokes='Q',dropdeg=False)
        exportfits(imagename=image_prefix+'-Q-image.image',fitsimage=image_prefix+'-Q-image.fits',dropdeg=False)
        os.system("rm -rf "+image_prefix+'-Q-image.image')
        os.system("rm -rf "+image_prefix+'-U-image.image '+image_prefix+'-U-image.fits') 
        imsubimage(imagename=imagename,outfile=image_prefix+'-U-image.image',stokes='U',dropdeg=False)
        exportfits(imagename=image_prefix+'-U-image.image',fitsimage=image_prefix+'-U-image.fits',dropdeg=False)
        os.system("rm -rf "+image_prefix+'-U-image.image')
        os.system("rm -rf "+image_prefix+'-V-image.image '+image_prefix+'-V-image.fits') 
        imsubimage(imagename=imagename,outfile=image_prefix+'-V-image.image',stokes='V',dropdeg=False)
        exportfits(imagename=image_prefix+'-V-image.image',fitsimage=image_prefix+'-V-image.fits',dropdeg=False)
        os.system("rm -rf "+image_prefix+'-V-image.image')
        print ('########################\n')
        os.environ["JOBLIB_TEMP_FOLDER"] = os.path.dirname(image_prefix) + "/tmp"
        if ncpu<4:
            n_jobs=ncpu
        else:
            n_jobs=4    
        wsclean_images = Parallel(n_jobs=n_jobs, backend='threading')(delayed(run_fits_warp)(xmfits,image_prefix+'-'+pol+'-image.fits',ncpu) for pol in ['I','Q','U','V'])
        output_image = make_stokes_cube(
                    wsclean_images,
                    image_prefix+"_unwarped",
                    imagetype="fits",
                    keep_wsclean_images=False,
                ) 
        if keep_original == False:
            os.system("rm -rf " + imagename)
            os.system("mv " + output_image + " " + imagename)
        else:
            imagename=output_image
        os.system("rm -rf "+os.path.dirname(image_prefix) + "/tmp")    
    else:
        print ('########################\n')
        unwarped_file=run_fits_warp(xmfits,imagename,ncpu)
        if keep_original == False:
            os.system("rm -rf " + imagename)
            os.system("mv " + unwarped_file + " " + imagename)
        else:
            imagename=unwarped_file              
    return imagename
    
