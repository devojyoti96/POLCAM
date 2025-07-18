from basic_func import *
from optparse import OptionParser
from astropy.io import fits
import os, gc, traceback, numpy as np
from correct_pb import *
from correct_ionosphere_warp import *
from joblib import Parallel, delayed


def correct_leakage_surface(
    imagename,
    q_surface_poly,
    u_surface_poly,
    v_surface_poly,
    outdir="",
):
    """
    Correct Stokes I to other Stokes leakages
    Parameters
    ----------
    imagename : str
        Imagename
    q_surface_poly : numpy.array
        Q leakage surface polynomial
    u_surface_poly : numpy.array
        U leakage surface polynomial
    v_surface_poly : numpy.array
        V leakage surface polynomials
    outdir : str
        Output directory name
    Returns
    -------
    str
        Output imagename
    """
    if outdir == "":
        outdir = os.path.dirname(imagename)
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    data = fits.getdata(imagename)
    header = fits.getheader(imagename)
    if header["CTYPE3"] == "FREQ":
        freq = header["CRVAL3"] / 10**6
    elif header["CTYPE4"] == "FREQ":
        freq = header["CTYPE4"] / 10**6
    else:
        print("No frequency information available in image: ", i)
        return
    q_surface_data = np.polyval(q_surface_poly, freq)
    u_surface_data = np.polyval(u_surface_poly, freq)
    v_surface_data = np.polyval(v_surface_poly, freq)
    if header["CTYPE3"] == "STOKES":
        data[0, 1, ...] = data[0, 1, ...] - (
            q_surface_data[0, 0, ...] * data[0, 0, ...]
        )
        data[0, 2, ...] = data[0, 2, ...] - (
            q_surface_data[0, 0, ...] * data[0, 0, ...]
        )
        data[0, 3, ...] = data[0, 3, ...] - (
            q_surface_data[0, 0, ...] * data[0, 0, ...]
        )
    else:
        data[1, 0, ...] = data[1, 0, ...] - (
            q_surface_data[0, 0, ...] * data[0, 0, ...]
        )
        data[2, 0, ...] = data[2, 0, ...] - (
            q_surface_data[0, 0, ...] * data[0, 0, ...]
        )
        data[3, 0, ...] = data[3, 0, ...] - (
            q_surface_data[0, 0, ...] * data[0, 0, ...]
        )
    fits.writeto(
        outdir
        + "/"
        + os.path.basename(imagename).split(".fits")[0]
        + "_leakagecor.fits",
        data,
        header,
        overwrite=True,
    )
    print("Leakage surface correction is done for: ", os.path.basename(imagename))
    return (
        outdir
        + "/"
        + os.path.basename(imagename).split(".fits")[0]
        + "_leakagecor.fits"
    )


def get_polycoeff_leakage_surface(leakage_surfaces=[], poldeg=3):
    """
    Get polynomial coefficients of leakage surface across frequency (in MHz)
    Parameters
    ----------
    leakage_surfaces : list
        List surface image list
    poldeg : int
        Polynomial degree
    Returns
    -------
    numpy.array
        Polynomial coefficient array for each pixels (array shape: poldeg+1 , n_Xpix, n_Ypix)
    """
    freqs = []
    surface_data = []
    for i in leakage_surfaces:
        data = fits.getdata(i)
        surface_data.append(data)
        header = fits.getheader(i)
        if header["CTYPE3"] == "FREQ":
            freq = header["CRVAL3"] / 10**6
        elif header["CTYPE4"] == "FREQ":
            freq = header["CTYPE4"] / 10**6
        else:
            print("No frequency information available in image: ", i)
            return
        freqs.append(freq)
    freqs = np.array(freqs)
    pos = np.argsort(freqs)
    freqs = freqs[pos]
    surface_data = np.array(surface_data)
    surface_data = surface_data[pos]
    reshaped_surface_data = surface_data.reshape(surface_data.shape[0], -1)
    coeffs = np.polyfit(freqs, reshaped_surface_data, deg=poldeg)
    coeffs = coeffs.reshape(
        coeffs.shape[0], surface_data.shape[1], surface_data.shape[2]
    )
    return coeffs


def apply_all_ddcal(imagedir, leakage_dir, warps_dir, metafits, ncpu=-1, mem=-1):
    ######################################
    # Primary beam correction
    ######################################
    print("#######################")
    print("Correcting for primary beams ....")
    print("#######################")
    pbcor_image_dir, pb_dir, total_images = correctpb_spectral_images(
        imagedir + "/images",
        metafits,
        interpolate=True,
        ncpu=ncpu,
        mem=mem,
    )
    pbcor_images = glob.glob(pbcor_image_dir + "/*pbcor.fits")
    q_leakage_surfaces = glob.glob(leakage_dir + "/*q_surface*.fits")
    u_leakage_surfaces = glob.glob(leakage_dir + "/*u_surface*.fits")
    v_leakage_surfaces = glob.glob(leakage_dir + "/*v_surface*.fits")
    q_surface_poly = get_polycoeff_leakage_surface(
        leakage_surfaces=q_leakage_surfaces, poldeg=3
    )
    u_surface_poly = get_polycoeff_leakage_surface(
        leakage_surfaces=u_leakage_surfaces, poldeg=3
    )
    v_surface_poly = get_polycoeff_leakage_surface(
        leakage_surfaces=v_leakage_surfaces, poldeg=3
    )
    ###########################################
    # Estimating number of parallel jobs
    ###########################################
    if ncpu == -1:
        ncpu = psutil.cpu_count(logical=True)
    available_mem = psutil.virtual_memory().available / 1024**3
    if mem == -1:
        mem = available_mem
    elif mem > available_mem:
        mem = available_mem
    file_size = os.path.getsize(pbcor_images[0]) / (1024**3)
    max_jobs = int(mem / file_size)
    if ncpu < max_jobs:
        n_jobs = ncpu
    else:
        n_jobs = max_jobs
    #############################################
    # Correcting leakage surfaces
    #############################################
    print("#######################")
    print("Correcting residual leakages ....")
    print("#######################")
    outdir = imagedir + "/images/leakage_cor"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    with Parallel(n_jobs=n_jobs) as parallel:
        final_images = parallel(
            delayed(correct_leakage_surface)(
                imagename, q_surface_poly, u_surface_poly, v_surface_poly, outdir=outdir
            )
            for imagename in pbcor_images
        )
    del parallel
    ############################################
    # Correcting ionospheric warp
    ############################################
    print("#######################")
    print("Correcting for ionospheric warps ....")
    print("#######################")
    outdir = imagedir + "/images/final_images"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    n_jobs = int(n_jobs / 4)
    if n_jobs < 1:
        n_jobs = 1
    with Parallel(n_jobs=n_jobs) as parallel:
        final_images = parallel(
            delayed(correct_warp)(imagename, warps_dir, ncpu=4, outdir=outdir)
            for imagename in final_images
        )
    del parallel
