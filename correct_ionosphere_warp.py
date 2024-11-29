import os, glob, logging, gc, warnings
from casatasks import imsubimage, exportfits

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
    os.system(bane_cmd + " > " + os.path.dirname(imagename) + "/tmp_bane")
    os.system("rm -rf " + os.path.dirname(imagename) + "/tmp_bane")
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
    os.system(aegean_cmd + " >" + os.path.dirname(imagename) + "/tmp_aegen")
    os.system("rm -rf " + os.path.dirname(imagename) + "/tmp_aegen")
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
        + "_xm.fits --plot --infits "
        + I_imagename
    )
    print(fitswarp_cmd + "\n")
    print("#########################")
    os.system(fitswarp_cmd + " > " + os.path.dirname(imagename) + "/tmp_fitswarp")
    os.system("rm -rf " + os.path.dirname(imagename) + "/tmp_fitswarp")
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


def correct_warp(imagename, xmfits, keep_original=True):
    """
    Parameters
    ----------
    imagename : str
        Name of the fits image
    xmfits : str
        Unwarped catalog fits file
    keep_original : bool
        Keep original image or replace it
    Returns
    -------
    str
        Unwraped fits image
    """
    image_prefix = imagename.split(".fits")[0]
    print("#########################\n")
    print(
        "Ionospheric correction using fits_warp: "
        + xmfits
        + " on image: "
        + imagename
        + "\n"
    )
    fitswarp_cmd = (
        "fits_warp.py --xm " + xmfits + " --infits " + imagename + " --suffix warped"
    )
    print(fitswarp_cmd + "\n")
    os.system(fitswarp_cmd)
    unwarped_file = image_prefix + "_unwarped.fits"
    if keep_original == False:
        os.system("rm -rf " + imagename)
        os.system("mv " + unwarped_file + " " + imagename)
        return imagename
    else:
        return unwarped_file
