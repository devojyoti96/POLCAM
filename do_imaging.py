import os, resource, gc, psutil
from basic_func import *
from optparse import OptionParser
from joblib import Parallel, delayed

os.system("rm -rf casa*log")


def perform_spectral_imaging(
    msname,
    nchan,
    imagedir="",
    multiscale_scales="",
    weight="briggs",
    robust=0.0,
    pol="IQUV",
    FWHM=True,
    threshold=5,
    minuv_l=-1,
    ncpu=-1,
    mem=-1,
):
    """
    Performing spectral imaging
    Parameters
    ----------
    msname : str
        Name of the measurement set
    nchan : int
        Number of spectral channel
    imagedir : str
        Imaging directory
    multiscale_scales : list
        Multiscale scales
    weight : str
        Image weighting
    robust : str
        Robust parameters for briggs weighting
    pol : str
        Polarization to image
    FWHM : bool
        Image upto FWHM or first null
    threshold : float
        Auto-threshold
    minuv_l : float
        Minimum uv-range in lambda
    ncpu : int
        Number of CPU threads to use
    mem : float
        Memory in GB
    Returns
    -------
    int
        Success message
    str
        Image directory
    str
        Image prefix name
    """
    pwd = os.getcwd()
    msname = os.path.abspath(msname)
    msmd = msmetadata()
    msmd.open(msname)
    max_chan = msmd.nchan(0)
    msmd.close()
    if nchan > max_chan:
        nchan = max_chan
    if imagedir == "":
        workdir = (
            os.path.dirname(os.path.abspath(msname))
            + "/imagedir_MFS_ch_"
            + str(nchan)
            + "_"
            + os.path.basename(msname).split(".ms")[0]
        )
    else:
        workdir = (
            imagedir
            + "/imagedir_MFS_ch_"
            + str(nchan)
            + "_"
            + os.path.basename(msname).split(".ms")[0]
        )
    if os.path.exists(workdir) == False:
        os.makedirs(workdir)
    else:
        os.system("rm -rf " + workdir + "/*")
    prefix = (
        workdir
        + "/"
        + os.path.basename(msname).split(".ms")[0]
        + "_nchan_"
        + str(nchan)
    )
    cwd = os.getcwd()
    os.chdir(workdir)
    cellsize = calc_cellsize(msname, 3)
    imsize = calc_imsize(msname, 3, FWHM=FWHM)
    if weight == "briggs":
        weight += " " + str(robust)
    if minuv_l < 0:
        uvrange = get_calibration_uvrange(msname)
        minuv_l = uvrange.split("~")[0]
    wsclean_args = [
        "-scale " + str(cellsize) + "asec",
        "-size " + str(imsize) + " " + str(imsize),
        "-no-dirty",
        "-weight " + weight,
        "-name " + prefix,
        "-pol " + str(pol),
        "-niter 2000",
        "-mgain 0.85",
        "-nmiter 5",
        "-gain 0.1",
        "-auto-threshold " + str(threshold) + " -auto-mask " + str(threshold + 0.1),
        "-minuv-l " + str(minuv_l),
        "-use-wgridder",
        "-channels-out " + str(nchan),
        "-temp-dir " + workdir,
        "-join-channels",
    ]
    if multiscale_scales != "":
        wsclean_args.append("-multiscale")
        wsclean_args.append("-multiscale-gain 0.1")
        wsclean_args.append("-multiscale-scales " + multiscale_scales)
    if ncpu > 0:
        wsclean_args.append("-j " + str(ncpu))
    if mem > 0:
        wsclean_args.append("-abs-mem " + str(mem))
    if pol == "QU":
        wsclean_cmd = (
            "wsclean " + " ".join(wsclean_args) + " -join-polarizations " + msname
        )
    else:
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
    print("Starting imaging of ms: " + msname + "\n")
    print(wsclean_cmd + "\n")
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    total_chunks = nchan * 4 * 4
    if total_chunks > soft_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE, (total_chunks, hard_limit))
    os.system(wsclean_cmd)  # + " > tmp_wsclean")
    os.system("rm -rf tmp_wsclean")
    os.chdir(pwd)
    return 0, workdir, os.path.basename(prefix)


def final_image_cubes(imagedir, image_prefix, imagetype="image", ncpu=-1, mem=-1):
    """
    Make final Stokes image cubes
    Parameters
    ----------
    imagedir : str
        Name of the image directory
    image_prefix : str
        Image prefix name
    ncpu : int
        Number of CPU threads to use
    mem : float
        Absolute memory to use in GB
    Returns
    -------
    list
        List of image cubes
    """

    def get_stokes_cube(image_prefix, imagetype, i_image, q_image, u_image, v_image):
        ch = str(
            i_image.split(image_prefix + "-")[-1].split("-I-" + imagetype + ".fits")[0]
        )
        header = fits.getheader(i_image)
        freq_MHz = float(header["CRVAL3"]) / 10**6
        coarse_chan = freq_to_MWA_coarse(freq_MHz)
        outfile_name = (
            image_prefix
            + "-ch-"
            + str(ch)
            + "-coch-"
            + str(coarse_chan)
            + "-iquv-"
            + imagetype
            + ".fits"
        )
        wsclean_images = [i_image, q_image, u_image, v_image]
        output_image = make_stokes_cube(
            wsclean_images,
            outfile_name,
            imagetype="fits",
            keep_wsclean_images=False,
        )
        gc.collect()
        return output_image

    s = time.time()
    pwd = os.getcwd()
    os.chdir(imagedir)
    ####################################
    # Making Stokes cubes
    ###################################
    i_images = sorted(glob.glob(image_prefix + "-*I-" + imagetype + ".fits"))
    q_images = sorted(glob.glob(image_prefix + "-*Q-" + imagetype + ".fits"))
    u_images = sorted(glob.glob(image_prefix + "-*U-" + imagetype + ".fits"))
    v_images = sorted(glob.glob(image_prefix + "-*V-" + imagetype + ".fits"))

    for i in range(len(i_images)):
        if "MFS" in i_images[i]:
            i_images.remove(i_images[i])
            q_images.remove(q_images[i])
            u_images.remove(u_images[i])
            v_images.remove(v_images[i])
    if ncpu == -1:
        ncpu = psutil.cpu_count(logical=True)
    available_mem = psutil.virtual_memory().available / 1024**3
    if mem == -1:
        mem = available_mem
    elif mem > available_mem:
        mem = available_mem
    file_size = os.path.getsize(i_images[0]) / (1024**3)
    max_jobs = int(mem / (4 * file_size))
    if ncpu < max_jobs:
        n_jobs = ncpu
    else:
        n_jobs = max_jobs
    print("Total parallel jobs: " + str(n_jobs) + "\n")
    final_images = Parallel(n_jobs=n_jobs)(
        delayed(get_stokes_cube)(
            image_prefix, imagetype, i_images[i], q_images[i], u_images[i], v_images[i]
        )
        for i in range(len(i_images))
    )
    os.chdir(pwd)
    time.sleep(2)
    gc.collect()
    print("Total time taken: " + str(round(time.time() - s, 2)) + "s.\n")
    return final_images


################################
def main():
    usage = "Perform spectral imaging"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
        metavar="String",
    )
    parser.add_option(
        "--nchan",
        dest="nchan",
        default=1,
        help="Number of spectral channels",
        metavar="Integer",
    )
    parser.add_option(
        "--imagedir",
        dest="imagedir",
        default="",
        help="Image directory",
        metavar="String",
    )
    parser.add_option(
        "--multiscale_scales",
        dest="multiscale_scales",
        default="",
        help="Multiscale scales",
        metavar="String",
    )
    parser.add_option(
        "--weight",
        dest="weight",
        default="natural",
        help="Image weighting",
        metavar="String",
    )
    parser.add_option(
        "--robust",
        dest="robust",
        default=0.0,
        help="Robust parameter for briggs weighting",
        metavar="Float",
    )
    parser.add_option(
        "--threshold",
        dest="threshold",
        default=6.0,
        help="Auto threshold for CLEANing",
        metavar="Float",
    )
    parser.add_option(
        "--pol",
        dest="pol",
        default="IQUV",
        help="Polarizations to image",
        metavar="String",
    )
    parser.add_option(
        "--FWHM",
        dest="FWHM",
        default=True,
        help="Image upto FWHM or first null",
        metavar="Boolean",
    )
    parser.add_option(
        "--minuv_l",
        dest="minuv_l",
        default=-1,
        help="Minimum uv-range in lambda",
        metavar="Float",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Numbers of CPU threads to be used",
        metavar="Integer",
    )
    parser.add_option(
        "--mem",
        dest="mem",
        default=-1,
        help="Memory in GB to be used",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if options.msname == None:
        print("Please provide the measurement set name.\n")
        return 1
    msg, imagedir, image_prefix = perform_spectral_imaging(
        options.msname,
        int(options.nchan),
        imagedir=options.imagedir,
        multiscale_scales=options.multiscale_scales,
        weight=options.weight,
        robust=float(options.robust),
        threshold=float(options.threshold),
        minuv_l=float(options.minuv_l),
        pol=options.pol,
        FWHM=eval(str(options.FWHM)),
        ncpu=int(options.ncpu),
        mem=float(options.mem),
    )
    final_images = final_image_cubes(
        imagedir,
        image_prefix,
        imagetype="image",
        ncpu=int(options.ncpu),
        mem=float(options.mem),
    )
    final_models = final_image_cubes(
        imagedir,
        image_prefix,
        imagetype="model",
        ncpu=int(options.ncpu),
        mem=float(options.mem),
    )
    final_residuals = final_image_cubes(
        imagedir,
        image_prefix,
        imagetype="residual",
        ncpu=int(options.ncpu),
        mem=float(options.mem),
    )
    os.system("rm -rf " + imagedir + "/*psf.fits")
    if os.path.exists(imagedir + "/images") == False:
        os.makedirs(imagedir + "/images")
    if os.path.exists(imagedir + "/models") == False:
        os.makedirs(imagedir + "/models")
    if os.path.exists(imagedir + "/residuals") == False:
        os.makedirs(imagedir + "/residuals")    
    os.system("mv " + imagedir + "/*image*.fits " + imagedir + "/images")
    os.system("mv " + imagedir + "/*model*.fits " + imagedir + "/models")
    os.system("mv " + imagedir + "/*residual*.fits " + imagedir + "/residuals")
    print("Images are saved in : ", imagedir)
    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)
