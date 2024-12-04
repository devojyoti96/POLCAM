import os, glob, logging, gc, warnings, psutil
from casatasks import imsubimage, exportfits
from astropy.io import fits
from basic_func import make_stokes_cube, make_bkg_rms_image
from joblib import Parallel, delayed
from optparse import OptionParser

os.system("rm -rf casa*log")
warnings.filterwarnings("ignore")


def estimate_warp_map(imagename, outdir="", ncpu =-1, allsky_cat="GGSM.fits"):
    """
    Parameters
    ----------
    imagename : str
        Name of the image to calculate ionospheric warp screen
    outdir : str
        Output directory name
    allsky_cat : str
        All sky fits file name
    Returns
    -------
    str
        Background image
    str
        rms image
    str
        Unwarped catalog fits file
    """
    image_prefix = imagename.split(".fits")[0]
    if os.path.exists(image_prefix + "_I.image"):
        os.system("rm -rf " + image_prefix + "_I.image")
    imsubimage(
        imagename=imagename,
        outfile=image_prefix + "_I.image",
        stokes="I",
        dropdeg=True,
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
    original_stdout = os.dup(1)  # File descriptor for stdout
    original_stderr = os.dup(2)  # File descriptor for stderr
    if (
        os.path.exists(image_prefix + "_rms.fits") == False
        or os.path.exists(image_prefix + "_bkg.fits") == False
    ):
        print("Estimating noise map using BANE...\n")
        if ncpu>0:
            bane_cmd = "BANE --noclobber --cores " +str(ncpu)+" "+ I_imagename
        else:
            bane_cmd = "BANE --noclobber " + I_imagename
        print(bane_cmd + "\n")
        with open(os.devnull, 'w') as fnull:
            os.dup2(fnull.fileno(), 1)  # Redirect stdout
            os.dup2(fnull.fileno(), 2)  # Redirect stderr
        os.system(bane_cmd)
        if I_image_prefix != image_prefix:
            os.system(
                "mv " + I_image_prefix + "_bkg.fits " + image_prefix + "_bkg.fits"
            )
            os.system(
                "mv " + I_image_prefix + "_rms.fits " + image_prefix + "_rms.fits"
            )
    rms_image = image_prefix + "_rms.fits"
    bkg_image = image_prefix + "_bkg.fits"
    os.dup2(original_stdout, 1)  # Restore stdout
    os.dup2(original_stderr, 2)  # Restore stderr
    print("Source finding using AEGEAN...\n")
    if ncpu>0:
        aegean_cmd = (
        "aegean --cores "
        +str(ncpu)
        +" --noise "
        + rms_image
        + " --background "
        + bkg_image
        + " "
        + I_imagename
        + " --table "
        + image_prefix
        + "_catalog.fits"
        )
    else:
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
    with open(os.devnull, 'w') as fnull:
        os.dup2(fnull.fileno(), 1)  # Redirect stdout
        os.dup2(fnull.fileno(), 2)  # Redirect stderr
    os.system(aegean_cmd)
    os.dup2(original_stdout, 1)  # Restore stdout
    os.dup2(original_stderr, 2)  # Restore stderr
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
    with open(os.devnull, 'w') as fnull:
        os.dup2(fnull.fileno(), 1)  # Redirect stdout
        os.dup2(fnull.fileno(), 2)  # Redirect stderr
    os.system(fitswarp_cmd)
    os.dup2(original_stdout, 1)  # Restore stdout
    os.dup2(original_stderr, 2)  # Restore stderr
    os.close(original_stdout)
    os.close(original_stderr)
    xm_fits = image_prefix + "_xm.fits"
    if outdir == "":
        outdir = os.path.dirname(imagename) + "/warps"
    if os.path.isdir(outdir) == False:
        os.makedirs(outdir)
    os.system("mv " + bkg_image + " " + rms_image + " " + xm_fits + " " + outdir)
    bkg_image = outdir + "/" + os.path.basename(bkg_image)
    rms_image = outdir + "/" + os.path.basename(rms_image)
    xm_fits = outdir + "/" + os.path.basename(xm_fits)
    os.system("rm -rf " + source_catalog+" "+I_imagename)
    gc.collect()
    return bkg_image, rms_image, xm_fits


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
    if ncpu < 0:
        ncpu = int(psutil.cpu_cpunt() * (100 - psutil.cpu_percent()) / 100.0)

    def run_fits_warp(xmfits, temp_image, ncpu):
        image_prefix = temp_image.split(".fits")[0]
        print(
            "Ionospheric correction using fits_warp: "
            + os.path.basename(xmfits)
            + " on image: "
            + os.path.basename(temp_image)
            + "\n"
        )
        fitswarp_cmd = (
            "fits_warp.py --xm "
            + xmfits
            + " --infits "
            + temp_image
            + " --cores "
            + str(ncpu)
            + " --suffix unwarped"
        )
        os.system(fitswarp_cmd + ">tmp")
        unwarped_file = image_prefix + "_unwarped.fits"
        os.system("rm -rf tmp " + temp_image)
        return unwarped_file

    image_prefix = imagename.split(".fits")[0]
    header = fits.getheader(imagename)
    if header["NAXIS3"] == 4 or header["NAXIS4"] == 4:
        os.system(
            "rm -rf "
            + image_prefix
            + "-I-image.image "
            + image_prefix
            + "-I-image.fits"
        )
        imsubimage(
            imagename=imagename,
            outfile=image_prefix + "-I-image.image",
            stokes="I",
            dropdeg=False,
        )
        exportfits(
            imagename=image_prefix + "-I-image.image",
            fitsimage=image_prefix + "-I-image.fits",
            dropdeg=False,
        )
        os.system("rm -rf " + image_prefix + "-I-image.image")
        os.system(
            "rm -rf "
            + image_prefix
            + "-Q-image.image "
            + image_prefix
            + "-Q-image.fits"
        )
        imsubimage(
            imagename=imagename,
            outfile=image_prefix + "-Q-image.image",
            stokes="Q",
            dropdeg=False,
        )
        exportfits(
            imagename=image_prefix + "-Q-image.image",
            fitsimage=image_prefix + "-Q-image.fits",
            dropdeg=False,
        )
        os.system("rm -rf " + image_prefix + "-Q-image.image")
        os.system(
            "rm -rf "
            + image_prefix
            + "-U-image.image "
            + image_prefix
            + "-U-image.fits"
        )
        imsubimage(
            imagename=imagename,
            outfile=image_prefix + "-U-image.image",
            stokes="U",
            dropdeg=False,
        )
        exportfits(
            imagename=image_prefix + "-U-image.image",
            fitsimage=image_prefix + "-U-image.fits",
            dropdeg=False,
        )
        os.system("rm -rf " + image_prefix + "-U-image.image")
        os.system(
            "rm -rf "
            + image_prefix
            + "-V-image.image "
            + image_prefix
            + "-V-image.fits"
        )
        imsubimage(
            imagename=imagename,
            outfile=image_prefix + "-V-image.image",
            stokes="V",
            dropdeg=False,
        )
        exportfits(
            imagename=image_prefix + "-V-image.image",
            fitsimage=image_prefix + "-V-image.fits",
            dropdeg=False,
        )
        os.system("rm -rf " + image_prefix + "-V-image.image")
        print("########################\n")
        if ncpu < 4:
            n_jobs = ncpu
        else:
            n_jobs = 4  
        with Parallel(n_jobs=n_jobs, backend="multiprocessing") as parallel:
            wsclean_images = parallel(
                delayed(run_fits_warp)(
                    xmfits, image_prefix + "-" + pol + "-image.fits", ncpu
                )
                for pol in ["I", "Q", "U", "V"]
            )
        del parallel
        output_image = make_stokes_cube(
            wsclean_images,
            image_prefix + "_unwarped",
            imagetype="fits",
            keep_wsclean_images=False,
        )
        if keep_original == False:
            os.system("rm -rf " + imagename)
            os.system("mv " + output_image + " " + imagename)
        else:
            imagename = output_image
    else:
        print("########################\n")
        unwarped_file = run_fits_warp(xmfits, imagename, ncpu)
        if keep_original == False:
            os.system("rm -rf " + imagename)
            os.system("mv " + unwarped_file + " " + imagename)
        else:
            imagename = unwarped_file
    return imagename


def main():
    usage = "Perform correction of direction dependent ionosphere"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--imagename",
        dest="imagename",
        default=None,
        help="Name of the image",
        metavar="String",
    )
    parser.add_option(
        "--do_correction",
        dest="do_correction",
        default=True,
        help="Perform ionospheric warp correction or only make warp surface",
        metavar="Boolean",
    )
    parser.add_option(
        "--warp_cat",
        dest="warp_cat",
        default="",
        help="Perform ionospheric warp correction using this warp catalog",
        metavar="String",
    )
    parser.add_option(
        "--keep_original",
        dest="keep_original",
        default=True,
        help="Keep original image or overwrite it",
        metavar="Boolean",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Numbers of CPU threads to use",
        metavar="Integer",
    )
    (options, args) = parser.parse_args()
    if options.imagename == None or os.path.exists(options.imagename) == False:
        print("Please provide correct imagename.")
        gc.collect()
        return 1
    try:
        if options.warp_cat == "" or os.path.exists(options.warp_cat) == False:
            print("Determining ionospheric warp surface...")
            warp_cat = estimate_warp_map(options.imagename)
        else:
            warp_cat = options.warp_cat
        if eval(str(options.do_correction)) == False:
            print("Ionospheric warp surface: " + warp_cat)
            gc.collect()
            return 0
        else:
            print(
                "Ionospheric warp correction using: "
                + os.path.basename(warp_cat)
                + "\n"
            )
            outfile = correct_warp(
                options.imagename,
                warp_cat,
                ncpu=int(options.ncpu),
                keep_original=eval(str(options.keep_original)),
            )
            header = fits.getheader(outfile)
            data = fits.getdata(outfile)
            header["UNWARP"] = "Y"
            fits.writeto(outfile, data, header, overwrite=True)
            print("Ionospheric warp corrected image: " + outfile)
            gc.collect()
            return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        return 1


if __name__ == "__main__":
    result = main()
    os._exit(result)
