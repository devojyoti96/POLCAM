import os, glob, gc, psutil, time, traceback, warnings
from joblib import Parallel, delayed
from optparse import OptionParser
from correct_ionosphere_warp import estimate_warp_map

warnings.filterwarnings("ignore")
os.system("rm -rf casa*log")
MWA_PB_file = "mwa_full_embedded_element_pattern.h5"
sweetspot_file = "MWA_sweet_spots.npy"


def run_cmd(cmd):
    imagename = os.path.basename(cmd.split("--imagename ")[-1].split(" ")[0])
    print("Correcting primary beam for image: " + imagename + "\n")
    result = os.system(cmd)
    gc.collect()
    return result


def correctpb_spectral_images(
    imagedir, image_prefix, metafits, interpolate=True, ncpu=-1, mem=-1
):
    """
    Perform primary beam corrections for all Stokes cube images
    Parameters
    ----------
    imagedir : str
        Image directory name
    image_prefix : str
        Image prefix of all images
    metafits : str
        Metafits file
    interpolate : bool
        Spatial interpolation or not
    ncpu : int
        Number of CPU threads to be used
    mem : float
        Amount of memory to be used in GB
    Returns
    -------
    str
        Primary beam corrected image directory
    int
        Numbers of primary beam corrected images
    """
    s = time.time()
    images = sorted(glob.glob(imagedir + "/" + image_prefix + "*image.fits"))
    if os.path.exists(imagedir + "/pbcor_images") == False:
        os.makedirs(imagedir + "/pbcor_images")
    if os.path.exists(imagedir + "/pbs") == False:
        os.makedirs(imagedir + "/pbs")
    cmd_list_1 = []
    cmd_list_2 = []
    pb_coch = []
    if ncpu == -1:
        ncpu = int(
            psutil.cpu_count(logical=True) * (100 - psutil.cpu_percent()) / 100.0
        )
    available_mem = psutil.virtual_memory().available / 1024**3
    if mem == -1:
        mem = available_mem
    elif mem > available_mem:
        mem = available_mem
    file_size = os.path.getsize(images[0]) / (1024**3)
    max_jobs = int(mem / (3 * file_size))
    if ncpu < max_jobs:
        n_jobs = ncpu
    else:
        n_jobs = max_jobs
    per_job_cpu = int(ncpu / n_jobs)
    if per_job_cpu < 1:
        per_job_cpu = 1
    #############################
    # Making ionosphere warp correction catalog
    #############################
    warp_cat_list = []
    for i in range(len(images)):
        imagename = images[i]
        if "MFS" in os.path.basename(imagename):
            warp_cat = estimate_warp_map(imagename, allsky_cat="GGSM.fits")
            warp_cat_list.append(warp_cat)
    ##############################
    # Warp correction and primary beam correction
    #############################
    for i in range(len(images)):
        imagename = images[i]
        if "MFS" not in os.path.basename(imagename):
            coch = os.path.basename(imagename).split("-coch-")[-1].split("-")[0]
            outfile = os.path.basename(imagename).split(".fits")[0] + "_pbcor"
            cmd = (
                "python3 mwapb.py --MWA_PB_file "
                + str(MWA_PB_file)
                + " --sweetspot_file "
                + str(sweetspot_file)
                + " --imagename "
                + imagename
                + " --outfile "
                + outfile
                + " --metafits "
                + metafits
                + " --IAU_order False --image_conv fits --num_threads "
                + str(per_job_cpu)
                + " --verbose False --interpolated "
                + str(interpolate)
            )
            if len(warp_cat_list) > 0:
                for warp_cat in warp_cat_list:
                    temp_image_prefix = os.path.basename(imagename).split("-ch")[0]
                    if temp_image_prefix in os.path.basename(warp_cat):
                        cmd += " --warp_cat " + warp_cat
            if coch in pb_coch:
                cmd += " --pb_jones_file " + imagedir + "/pbs/pbfile_" + coch + ".npy"
                cmd_list_2.append(cmd)
            else:
                pb_coch.append(coch)
                cmd += " --save_pb " + imagedir + "/pbs/pbfile_" + coch
                cmd_list_1.append(cmd)
    print("Maximum numbers of parallel jobs: " + str(n_jobs) + "\n")
    if os.path.exists(imagedir + "/tmp") == False:
        os.makedirs(imagedir + "/tmp")
    os.environ["JOBLIB_TEMP_FOLDER"] = imagedir + "/tmp"
    with Parallel(n_jobs=n_jobs) as parallel:
        msgs = parallel(delayed(run_cmd)(cmd) for cmd in cmd_list_1)
    with Parallel(n_jobs=n_jobs) as parallel:
        msgs = parallel(delayed(run_cmd)(cmd) for cmd in cmd_list_2)
    os.system("mv " + imagedir + "/*pbcor.fits " + imagedir + "/pbcor_images/")
    print("Total time taken : " + str(round(time.time() - s, 2)) + "s.\n")
    os.system("rm -rf " + imagedir + "/tmp")
    total_images = len(glob.glob(imagedir + "/pbcor_images/*"))
    gc.collect()
    return imagedir + "/pbcor_images/", total_images


################################
def main():
    usage = "Perform correction of direction dependent effects (ionosphere, primary beam and direction dependent leakages)"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--imagedir",
        dest="imagedir",
        default=None,
        help="Name of the image directory",
        metavar="String",
    )
    parser.add_option(
        "--image_prefix",
        dest="image_prefix",
        default=None,
        help="Image prefix name",
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
        "--interpolate",
        dest="interpolate",
        default=True,
        help="Spatial interpolation or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Numbers of CPU threads to use",
        metavar="Integer",
    )
    parser.add_option(
        "--mem",
        dest="mem",
        default=-1,
        help="Amount of memory in GB to use",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if (
        options.imagedir == None
        or options.image_prefix == None
        or options.metafits == None
    ):
        print("Please provide necessary input parameters.\n")
        return 1
    try:
        pbcor_image_dir, total_images = correctpb_spectral_images(
            options.imagedir,
            options.image_prefix,
            options.metafits,
            interpolate=eval(str(options.interpolate)),
            ncpu=int(options.ncpu),
            mem=float(options.mem),
        )
        print(
            "Total primary beam corrected images are made: "
            + str(total_images)
            + " and saved in: "
            + str(pbcor_image_dir)
            + "\n"
        )
        return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        return 1


if __name__ == "__main__":
    result = main()
    os._exit(result)
