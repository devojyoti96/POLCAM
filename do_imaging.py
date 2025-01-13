import os, resource, gc, psutil
from basic_func import *
from optparse import OptionParser
from joblib import Parallel, delayed

os.system("rm -rf casa*log")


def perform_spectrotemporal_imaging(
    msname,
    freqres=-1,
    timeres=-1,
    nchan=1,
    ntime=1,
    imagedir="",
    use_multiscale=True,
    multiscale_scales="",
    weight="briggs",
    robust=0.0,
    pol="IQUV",
    FWHM=True,
    imsize=None,
    threshold=3,
    niter=100000,
    minuv_l=-1,
    savemodel=False,
    ncpu=-1,
    mem=-1,
):
    """
    Performing spectral imaging
    Parameters
    ----------
    msname : str
        Name of the measurement set
    freqres : float
        Frequency resolution in kHz (If specified, nchan will be ignored)
    timeres : float
        Temporal resolution in seconds (If specified, ntime will be ignored)
    nchan : int
        Numbers of spectral channels
    ntime : int
        Numbers of temporal slices
    imagedir : str
        Imaging directory
    use_multiscale : bool
        Use multiscale or not
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
    imsize : int
        Image size in degree. If it is given, FWHM parameter will be ignored
    threshold : float
        Auto-threshold
    niter: int
        Number of iterations
    minuv_l : float
        Minimum uv-range in lambda
    savemodel : bool
        Save model to modelcolumn or not
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
    max_time = len(msmd.timesforspws(0))
    if freqres > 0:
        freqres = freqres * 10**3
        ms_freqres = msmd.chanres(0)[0]
        if freqres < ms_freqres:
            print(
                "Intended frequency resolution: "
                + str(round(freqres / 1000.0, 2))
                + " kHz is smaller than ms frequency resolution: "
                + str(round(ms_freqres / 1000.0, 2))
                + "\n"
            )
            freqres = ms_freqres
        ms_freqs = msmd.chanfreqs(0)
        bw = np.max(ms_freqs) - np.min(ms_freqs)  # In Hz
        nchan = int(bw / freqres)
    if timeres > 0:
        ms_times = msmd.timesforspws(0)
        ms_timeres = abs(ms_times[1] - ms_times[0])
        total_time = abs(np.max(ms_times) - np.min(ms_times))
        if timeres < ms_timeres:
            print(
                "Intended time resolution: "
                + str(round(timeres, 2))
                + " seconds is smaller than ms time resolution: "
                + str(round(ms_timeres, 2))
                + "\n"
            )
            timeres = ms_timeres
        ntime = int(total_time / timeres)
    msmd.close()
    if nchan > max_chan:
        nchan = max_chan
    elif nchan < 1:
        nchan = 1
    if ntime > max_time:
        ntime = max_time
    elif ntime < 1:
        ntime = 1
    if imagedir == "":
        workdir = (
            os.path.dirname(os.path.abspath(msname))
            + "/imagedir_MFS_ch_"
            + str(nchan)
            + "_t_"
            + str(ntime)
            + "_pol_"
            + pol
            + "_"
            + os.path.basename(msname).split(".ms")[0]
        )
    else:
        workdir = (
            imagedir
            + "/imagedir_MFS_ch_"
            + str(nchan)
            + "_t_"
            + str(ntime)
            + "_pol_"
            + pol
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
        + "_ntime_"
        + str(ntime)
    )
    cwd = os.getcwd()
    os.chdir(workdir)
    cellsize = calc_cellsize(msname, 3)
    if imsize == None and imsize < 32:
        imsize = calc_imsize(msname, 3, FWHM=FWHM)
    else:
        imsize = int(imsize * 3600.0 / cellsize)
        pow2 = round(np.log2(imsize / 10.0), 0)
        imsize = int((2**pow2) * 10)
    if weight == "briggs":
        weight += " " + str(robust)
    if float(minuv_l) < 0:
        uvrange = get_calibration_uvrange(msname)
        minuv_l = float(uvrange.split("~")[0])
    wsclean_args = [
        "-scale " + str(cellsize) + "asec",
        "-size " + str(imsize) + " " + str(imsize),
        "-no-dirty",
        "-weight " + weight,
        "-name " + prefix,
        "-pol " + str(pol),
        "-niter " + str(niter),
        "-mgain 0.85",
        "-nmiter 5",
        "-gain 0.1",
        "-join-channels",
        "-auto-threshold 1 -auto-mask " + str(threshold),
        "-minuv-l " + str(minuv_l),
        "-channels-out " + str(nchan),
        "-intervals-out " + str(ntime),
        "-temp-dir " + workdir,
    ]
    if savemodel == False:
        wsclean_args.append("-no-update-model-required")
    if use_multiscale:
        wsclean_args.append("-multiscale")
        wsclean_args.append("-multiscale-gain 0.1")
    if multiscale_scales != "":
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
    os.system(wsclean_cmd + " > " + prefix + "_wsclean.log")
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

    def get_outfile_name(image_prefix, imagename, imagetype, pol):
        t_ch_split = os.path.basename(imagename).split(image_prefix)[1].split("-")[1]
        try:
            ch = str(int(t_ch_split))
        except:
            ch = "MFS"
        header = fits.getheader(imagename)
        freq_MHz = float(header["CRVAL3"]) / 10**6
        coarse_chan = freq_to_MWA_coarse(freq_MHz)
        dateobs = header["DATE-OBS"]
        t_str = (
            "".join(dateobs.split("T")[0].split("-"))
            + "_"
            + "".join(dateobs.split("T")[-1].split(":"))
        )
        outfile_name = (
            image_prefix
            + "-t-"
            + str(t_str)
            + "-f-"
            + str(round(freq_MHz, 3))
            + "-ch-"
            + str(ch)
            + "-coch-"
            + str(coarse_chan)
            + "-"
            + pol
            + "-"
            + imagetype
            + ".fits"
        )
        return outfile_name

    def get_stokes_cube(image_prefix, imagetype, pol, imagelist=[]):
        outfile_name = get_outfile_name(image_prefix, imagelist[0], imagetype, pol)
        output_image = make_stokes_cube(
            imagelist,
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
    # Making image list for stokes cube
    ###################################
    i_images = sorted(glob.glob(image_prefix + "-*I-" + imagetype + ".fits"))
    q_images = sorted(glob.glob(image_prefix + "-*Q-" + imagetype + ".fits"))
    u_images = sorted(glob.glob(image_prefix + "-*U-" + imagetype + ".fits"))
    v_images = sorted(glob.glob(image_prefix + "-*V-" + imagetype + ".fits"))
    pol = ""
    pol_list = []
    filtered_i_images = []
    mfs_i_images = []
    filtered_q_images = []
    mfs_q_images = []
    filtered_u_images = []
    mfs_u_images = []
    filtered_v_images = []
    mfs_v_images = []
    if len(i_images) > 0:
        pol += "i"
        pol_list.append("i")
        for i in range(len(i_images)):
            if "MFS" not in i_images[i]:
                filtered_i_images.append(i_images[i])
            else:
                mfs_i_images.append(i_images[i])
    if len(q_images) > 0:
        pol += "q"
        pol_list.append("q")
        for i in range(len(q_images)):
            if "MFS" not in q_images[i]:
                filtered_q_images.append(q_images[i])
            else:
                mfs_q_images.append(q_images[i])
    if len(u_images) > 0:
        pol += "u"
        pol_list.append("u")
        for i in range(len(u_images)):
            if "MFS" not in u_images[i]:
                filtered_u_images.append(u_images[i])
            else:
                mfs_u_images.append(u_images[i])
    if len(v_images) > 0:
        pol += "v"
        pol_list.append("v")
        for i in range(len(v_images)):
            if "MFS" not in v_images[i]:
                filtered_v_images.append(v_images[i])
            else:
                mfs_v_images.append(v_images[i])
    ####################################
    # Making image list of polarizations
    ####################################
    final_image_list = []
    final_mfs_image_list = []
    n_images = np.nanmax(
        [
            len(filtered_i_images),
            len(filtered_q_images),
            len(filtered_u_images),
            len(filtered_v_images),
        ]
    )
    n_mfs_images = np.nanmax(
        [len(mfs_i_images), len(mfs_q_images), len(mfs_u_images), len(mfs_v_images)]
    )
    for i in range(n_images):
        temp_list = []
        if "i" in pol_list:
            temp_list.append(filtered_i_images[i])
        if "q" in pol_list:
            temp_list.append(filtered_q_images[i])
        if "u" in pol_list:
            temp_list.append(filtered_u_images[i])
        if "v" in pol_list:
            temp_list.append(filtered_v_images[i])
        final_image_list.append(temp_list)
    for i in range(n_mfs_images):
        temp_list = []
        if "i" in pol_list:
            temp_list.append(mfs_i_images[i])
        if "q" in pol_list:
            temp_list.append(mfs_q_images[i])
        if "u" in pol_list:
            temp_list.append(mfs_u_images[i])
        if "v" in pol_list:
            temp_list.append(mfs_v_images[i])
        final_mfs_image_list.append(temp_list)

    #################################
    if ncpu == -1:
        ncpu = psutil.cpu_count(logical=True)
    available_mem = psutil.virtual_memory().available / 1024**3
    if mem == -1:
        mem = available_mem
    elif mem > available_mem:
        mem = available_mem
    file_size = os.path.getsize(final_image_list[0][0]) / (1024**3)
    max_jobs = int(mem / (4 * file_size))
    if ncpu < max_jobs:
        n_jobs = ncpu
    else:
        n_jobs = max_jobs
    print("Total parallel jobs: " + str(n_jobs) + "\n")
    with Parallel(n_jobs=n_jobs) as parallel:
        final_images = parallel(
            delayed(get_stokes_cube)(
                image_prefix, imagetype, pol, imagelist=final_image_list[i]
            )
            for i in range(len(final_image_list))
        )
    del parallel
    with Parallel(n_jobs=n_jobs) as parallel:
        final_mfs_images = parallel(
            delayed(get_stokes_cube)(
                image_prefix,
                imagetype,
                pol,
                imagelist=final_mfs_image_list[i],
            )
            for i in range(len(final_mfs_image_list))
        )
    del parallel
    for image in final_mfs_images:
        final_images.append(image)
    os.chdir(pwd)
    time.sleep(2)
    gc.collect()
    print("Total time taken: " + str(round(time.time() - s, 2)) + "s.\n")
    return final_images


################################
def main():
    usage = "Perform spectral polarimetric snapshot imaging\nImages will be saved in : {imagedir}+'/imagedir_MFS_ch_{nchan}_t_{ntime}_pol_{pol}_{obsid}\nImages name format: {obsid}_nchan_{nchan}_ntime_{ntime}-t-{yyyymmdd}_{hhmmss.ff}-f-{freqMHz}-ch-{ch_number}-coch-{coarse_chan_number}-iquv-image.fits"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
        metavar="String",
    )
    parser.add_option(
        "--freqres",
        dest="freqres",
        default=-1,
        help="Spectral resolution of image in kHz (If specified, nchan will be ignored)",
        metavar="Float",
    )
    parser.add_option(
        "--nchan",
        dest="nchan",
        default=1,
        help="Number of spectral channels",
        metavar="Integer",
    )
    parser.add_option(
        "--timeres",
        dest="timeres",
        default=-1,
        help="Temporal resolution of image in seconds (If specified, ntime will be ignored)",
        metavar="Float",
    )
    parser.add_option(
        "--ntime",
        dest="ntime",
        default=1,
        help="Number of temporal slices",
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
        "--use_multiscale",
        dest="use_multiscale",
        default=True,
        help="Use multiscale or not",
        metavar="Boolean",
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
        default=3.0,
        help="Auto threshold for CLEANing",
        metavar="Float",
    )
    parser.add_option(
        "--niter",
        dest="niter",
        default=100000,
        help="Number of iterations",
        metavar="Integer",
    )
    parser.add_option(
        "--pol",
        dest="pol",
        default="IQUV",
        help="Polarizations to image (valid modes: 'IQUV', 'XXYY', 'RRLL', 'I', 'QU', 'IV','IQ')",
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
        "--savemodel",
        dest="savemodel",
        default=False,
        help="Save model to modelcolumn or not",
        metavar="Boolean",
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
    if options.pol not in [
        "IQUV",
        "XXYY",
        "RRLL",
        "I",
        "QU",
        "IV",
        "IQ",
        "iquv",
        "xxyy",
        "rrll",
        "i",
        "qu",
        "iv",
        "iq",
    ]:
        print(
            "Given polarization mode: "
            + options.pol
            + " is not a valid combination. Choose a correct combination.\n"
        )
        return 1
    msg, imagedir, image_prefix = perform_spectrotemporal_imaging(
        options.msname,
        freqres=float(options.freqres),
        timeres=float(options.timeres),
        nchan=int(options.nchan),
        ntime=int(options.ntime),
        imagedir=options.imagedir,
        use_multiscale=eval(str(options.use_multiscale)),
        multiscale_scales=options.multiscale_scales,
        weight=options.weight,
        robust=float(options.robust),
        threshold=float(options.threshold),
        niter=int(options.niter),
        minuv_l=float(options.minuv_l),
        pol=options.pol,
        FWHM=eval(str(options.FWHM)),
        savemodel=eval(str(options.savemodel)),
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
