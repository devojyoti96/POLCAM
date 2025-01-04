from basic_func import *
from optparse import OptionParser
from astropy.io import fits
import os, gc, traceback
from correct_pb import *
from correct_ionosphere_warp import *


def fit_leakage_poly(dataq, datai):
    """
    Fit leakage surface polynominal
    Parameters
    ----------
    dataq : numpy.array
        Polarization data
    datai : numpy.array
        Stokes I data
    Returns
    -------
    numpy.array
        Leakage surface array
    """
    # Calculate q_by_i and filter valid indices
    q_by_i = dataq / datai
    valid_indices = ~np.isnan(q_by_i)
    # Extract x, y, and z from valid indices
    x, y = np.where(valid_indices)
    z = q_by_i[valid_indices]
    z[np.abs(z) > 1] = 1
    # Create the design matrix for the least squares fit
    q_stack = np.column_stack((x, y, z))
    AQ = np.c_[
        np.ones(q_stack.shape[0]),
        q_stack[:, :2],
        np.prod(q_stack[:, :2], axis=1),
        q_stack[:, :2] ** 2,
    ]
    # Solve for coefficients using least squares
    CQ, _, _, _ = scipy.linalg.lstsq(AQ, q_stack[:, 2])
    # Generate the surface values
    k_indices, l_indices = np.meshgrid(
        range(dataq.shape[0]), range(dataq.shape[1]), indexing="ij"
    )
    data_backup = (
        CQ[4] * k_indices**2
        + CQ[5] * l_indices**2
        + CQ[3] * k_indices * l_indices
        + CQ[1] * k_indices
        + CQ[2] * l_indices
        + CQ[0]
    )
    return data_backup


def leakage_surface(imagename, outdir="", threshold=5, bkg_image="", rms_image=""):
    """
    Make Stokes I to other stokes leakage surface
    Parameters
    ----------
    imagename : str
        Name of the image
    outdir : str
        Output directory name
    threshold : float
        Threshold to choose sources from Stokes I
    bkg_image : str
        Background image
    rms_image : str
        rms image
    Returns
    -------
    int
        Success message
    str
        Output directory name
    str
        Q surface image
    str
        U surface image
    str
        V surface image
    """
    if (
        bkg_image == ""
        or rms_image == ""
        or os.path.exists(bkg_image) == False
        or os.path.exists(rms_image) == False
    ):
        bkg_image, rms_image = make_bkg_rms_image(imagename)
    data = fits.getdata(imagename)
    header = fits.getheader(imagename)
    if header["NAXIS"] != 4 or (data.shape[0] != 4 and data.shape[1] != 4):
        print(
            "This image: "
            + imagename
            + " is not a full stokes image. Please provide full Stokes image."
        )
        return 1, None, None, None, None
    try:
        if header["CTYPE3"] == "STOKES":
            q_surface = fit_leakage_poly(data[0, 1, ...], data[0, 0, ...])
            u_surface = fit_leakage_poly(data[0, 2, ...], data[0, 0, ...])
            v_surface = fit_leakage_poly(data[0, 3, ...], data[0, 0, ...])
        elif header["CTYPE4"] == "STOKES":
            q_surface = fit_leakage_poly(data[1, 0, ...], data[0, 0, ...])
            u_surface = fit_leakage_poly(data[2, 0, ...], data[0, 0, ...])
            v_surface = fit_leakage_poly(data[3, 0, ...], data[0, 0, ...])
        else:
            print("Stokes axis is not present.")
            print("Could not make leakage surface.")
            return 1, None, None, None, None

        imagename_split = os.path.basename(imagename).split("-")
        index = imagename_split.index("iquv")
        header["BUNIT"] = "FRAC"

        if outdir == "":
            outdir = os.path.dirname(imagename) + "/leakage_surfaces"
        if os.path.isdir(outdir) == False:
            os.makedirs(outdir)

        imagename_split[index] = "q_surface"
        q_surface_name = "-".join(imagename_split)
        fits.writeto(outdir + "/" + q_surface_name, q_surface, header, overwrite=True)

        imagename_split[index] = "u_surface"
        u_surface_name = "-".join(imagename_split)
        fits.writeto(outdir + "/" + u_surface_name, u_surface, header, overwrite=True)

        imagename_split[index] = "v_surface"
        v_surface_name = "-".join(imagename_split)
        fits.writeto(outdir + "/" + v_surface_name, v_surface, header, overwrite=True)

        return (
            0,
            outdir,
            outdir + "/" + q_surface_name,
            outdir + "/" + u_surface_name,
            outdir + "/" + v_surface_name,
        )
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print(
            "#####################\nLeakage surface estimation failed.\n#####################\n"
        )
        return 1, None, None, None, None


def perform_leakage_cal(imagedir, metafits, source_model_fits, ncpu=-1, mem=-1):
    """
    Perform leakage calibration
    Parameters
    ----------
    imagedir : str
        Image directory
    metafits : str
        Metafits file name
    source_model_fits : str
        Source model fits table
    ncpu : int
        Number of CPU threads
    mem : float
        Memory to use in GB
    Returns
    -------
    str
        A file with output directory names
    """
    save_dirname = metafits.split(".metafits")[0] + "_ddcal_dirs.npy"
    os.system(
        "rm -rf "
        + imagedir
        + "/images/*_I.fits "
        + imagedir
        + "/images/*_comp.fits "
        + imagedir
        + "/images/*_bkg.fits "
        + imagedir
        + "/images/*_rms.fits"
    )
    image_list = glob.glob(imagedir + "/images/*.fits")
    filtered_image_list = []
    for image in image_list:
        if "MFS" not in os.path.basename(image):
            filtered_image_list.append(image)
    image_list = filtered_image_list
    ddcal_dir_list = []
    ######################################
    # Primary beam correction
    ######################################
    print("#######################")
    print("Estimating primary beams ....")
    print("#######################")
    pbcor_image_dir, pb_dir, total_images = correctpb_spectral_images(
        imagedir + "/images",
        metafits,
        interpolate=True,
        ncpu=ncpu,
        mem=mem,
    )
    ddcal_dir_list.append(pbcor_image_dir)
    ddcal_dir_list.append(pb_dir)
    ######################################
    # Ionospheric warp surface
    ######################################
    warp_outdir = imagedir + "/warps"
    if os.path.isdir(warp_outdir) == False:
        os.makedirs(warp_outdir)
    print("#######################")
    print("Estimating ionosphere warp surfaces....")
    print("#######################")
    if ncpu == -1:
        nthread = psutil.cpu_count(logical=True)
    else:
        nthread = ncpu
    available_mem = psutil.virtual_memory().available / 1024**3
    if mem == -1:
        absmem = available_mem
    elif mem > available_mem:
        absmem = available_mem
    else:
        absmem = mem
    file_size = os.path.getsize(image_list[0]) / (1024**3)
    max_jobs = int(absmem / file_size)
    if nthread < max_jobs:
        n_jobs = nthread
    else:
        n_jobs = max_jobs
    print("Total parallel jobs: " + str(n_jobs) + "\n")
    with Parallel(n_jobs=n_jobs, backend="multiprocessing") as parallel:
        results = parallel(
            delayed(estimate_warp_map)(
                imagename,
                outdir=warp_outdir,
                allsky_cat=source_model_fits,
            )
            for imagename in image_list
        )
    ddcal_dir_list.append(warp_outdir)
    #########################################
    # Polconversion estimation
    #########################################
    print("#######################")
    print("Estimating ionosphere warp surfaces....")
    print("#######################")
    leakage_surface_outdir = imagedir + "/leakage_surfaces"
    if os.path.isdir(leakage_surface_outdir) == False:
        os.makedirs(leakage_surface_outdir)
    bkg_image_list = []
    rms_image_list = []
    for imagename in image_list:
        bkg_image = glob.glob(
            warp_outdir + "/" + os.path.basename(imagename).split(".fits")[0] + "*bkg*"
        )
        if len(bkg_image) > 0:
            bkg_image = bkg_image[0]
        else:
            bkg_image = ""
        rms_image = glob.glob(
            warp_outdir + "/" + os.path.basename(imagename).split(".fits")[0] + "*rms*"
        )
        if len(rms_image) > 0:
            rms_image = rms_image[0]
        else:
            rms_image = ""
        bkg_image_list.append(bkg_image)
        rms_image_list.append(rms_image)
    with Parallel(n_jobs=n_jobs, backend="multiprocessing") as parallel:
        results = parallel(
            delayed(leakage_surface)(
                image_list[i],
                outdir=leakage_surface_outdir,
                threshold=5,
                bkg_image=bkg_image_list[i],
                rms_image=rms_image_list[i],
            )
            for i in range(len(image_list))
        )
    del parallel
    ddcal_dir_list.append(leakage_surface_outdir)
    np.save(save_dirname, ddcal_dir_list)
    return save_dirname


def main():
    usage = "Determine direction-dependent pol-conversion leakage surfaces"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--imagedir",
        dest="imagedir",
        default=None,
        help="Name of the image directory",
        metavar="String",
    )
    parser.add_option(
        "--metafits",
        dest="metafits",
        default=None,
        help="Name of the metafits file",
        metavar="String",
    )
    parser.add_option(
        "--source_model",
        dest="source_model",
        default=None,
        help="Name of the source model fits table",
        metavar="String",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Numbers of CPU threads to use",
        metavar="Integer",
    )
    parser.add_option(
        "--absmem",
        dest="absmem",
        default=-1,
        help="Memory in GB to be used",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if options.imagedir == None or os.path.exists(options.imagedir) == False:
        print("Please provide a valid image directory name.\n")
        return 1
    elif options.metafits == None or os.path.exists(options.metafits) == False:
        print("Please provide a valid metafits name.\n")
        return 1
    elif options.source_model == None or os.path.exists(options.source_model) == False:
        print("Please provide a valid source model fits table.\n")
        return 1
    else:
        try:
            dirname = perform_leakage_cal(
                options.imagedir,
                options.metafits,
                options.source_model,
                ncpu=int(options.ncpu),
                mem=float(options.absmem),
            )
            print("Direction-dependent calibration directories are saved in: ", dirname)
            return 0
        except Exception as e:
            traceback.print_exc()
            gc.collect()
            return 1


if __name__ == "__main__":
    result = main()
    os._exit(result)
