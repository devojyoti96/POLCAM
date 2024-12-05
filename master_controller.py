import glob, os, psutil, traceback, gc, tempfile, shutil
from basic_func import *
from optparse import OptionParser
from correct_pb import *
from correct_ionosphere_warp import *
from correct_polcon import *

os.system("rm -rf casa*log")

beamfile = os.getcwd() + "/mwa_full_embedded_element_pattern.h5"
source_model_file = os.getcwd() + "/GGSM.txt"
source_model_fits = os.getcwd() + "/GGSM.fits"


def perform_model_import(msdir, basedir, cpu_percent=10, mem_percent=20):
    """
    Perform model import in all ms
    Parameters
    ----------
    msdir : str
        MS directory
    basedir : str
        Base directory
    cpu_percent : float
        Free CPU percentage
    mem_percent : float
        Free memory percentage
    Returns
    -------
    int
        Success message (0 or 1)
    """
    print("Model import jobs are being started ....\n")
    try:
        os.system("rm -rf " + basedir + "/.Finished_model*")
        mslist = glob.glob(msdir + "/*.ms")
        trial_ms = mslist[0]
        mssize = get_column_size(trial_ms, "DATA")  # In GB
        total_memory = psutil.virtual_memory().available / (1024**3)  # In GB
        max_jobs = int(total_memory / mssize)
        total_cpus = psutil.cpu_count()
        ncpu = int(total_cpus / max_jobs)
        if ncpu < 1:
            ncpu = 1
            max_jobs=total_cpus    
        print("Maximum numbers of jobs to spawn at once:", max_jobs)
        count = 0
        free_jobs = -1
        for ms in mslist:
            metafits = (
                os.path.dirname(os.path.abspath(ms))
                + "/"
                + os.path.basename(ms).split(".ms")[0].split("_")[0]
                + ".metafits"
            )
            cmd = (
                "python3 hyperdrive_model.py --msname "
                + ms
                + " --metafits "
                + metafits
                + " --beamfile "
                + beamfile
                + " --sourcelist "
                + source_model_file
                + " --ncpu "
                + str(ncpu)
            )
            basename = "model_" + os.path.basename(ms).split(".ms")[0] + "_hyperdrive"
            batch_file = create_batch_script_nonhpc(cmd, basedir, basename)
            os.system("bash " + batch_file)
            print("Spawned command: " + cmd + "\n")
            count += 1
            if free_jobs > 0:
                free_jobs -= 1
            if count >= max_jobs or free_jobs == 0:
                free_jobs = wait_for_resources(
                    basedir + "/.Finished_model",
                    cpu_threshold=cpu_percent,
                    memory_threshold=mem_percent,
                )
                print("Freeded jobs: ", free_jobs)
        while True:
            finished_files = glob.glob(basedir + "/.Finished_model*")
            if len(finished_files) >= count:
                break
        print(
            "#####################\nModel import jobs are finished successfully.\n#####################\n"
        )
        gc.collect()
        return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print(
            "#####################\nModel import jobs are finished unsuccessfully.\n#####################\n"
        )
        return 1


def perform_all_calibration(
    msdir, basedir, refant=1, do_kcross=True, cpu_percent=10, mem_percent=20
):
    """
    Perform bandpass and crosshand phase calibration for all ms
    Parameters
    ----------
    msdir : str
        MS directory
    basedir : str
        Base directory
    refant : int
        Reference antenna index
    do_kcross : bool
        Perform crosshand phase calibration
    cpu_percent : float
        Free CPU percentage
    mem_percent : float
        Free memory percentage
    Returns
    -------
    int
        Success message (0 or 1)
    """
    print("Calibration jobs are being started ....\n")
    try:
        os.system("rm -rf " + basedir + "/.Finished_calibrate*")
        mslist = glob.glob(msdir + "/*.ms")
        caldir = basedir + "/caldir"
        trial_ms = mslist[0]
        mssize = get_column_size(trial_ms, "DATA")  # In GB
        total_memory = psutil.virtual_memory().available / (1024**3)  # In GB
        max_jobs = int(total_memory / mssize)
        total_cpus = psutil.cpu_count()
        ncpu = int(total_cpus / max_jobs)
        if ncpu < 1:
            ncpu = 1
            max_jobs=total_cpus 
        print("Maximum numbers of jobs to spawn at once:", max_jobs)
        count = 0
        free_jobs = -1
        for ms in mslist:
            cmd = (
                "python3 calibrate.py --msname "
                + ms
                + " --refant "
                + str(refant)
                + " --do_kcross "
                + str(do_kcross)
                + " --caldir "
                + caldir
            )
            basename = (
                "calibrate_" + os.path.basename(ms).split(".ms")[0] + "_bcal_kcross"
            )
            batch_file = create_batch_script_nonhpc(cmd, basedir, basename)
            os.system("bash " + batch_file)
            print("Spawned command: " + cmd + "\n")
            count += 1
            if count >= max_jobs:
                free_jobs = wait_for_resources(
                    basedir + "/.Finished_calibrate",
                    cpu_threshold=cpu_percent,
                    memory_threshold=mem_percent,
                )
                total_memory = psutil.virtual_memory().available / (1024**3)  # In GB
                max_jobs = int(total_memory / mssize)
                count = 0
                print("Maximum freed jobs: ", max_jobs)
        while True:
            finished_files = glob.glob(basedir + "/.Finished_calibrate*")
            if len(finished_files) >= count:
                break
        print(
            "#####################\nCalibration jobs are finished successfully.\n#####################\n"
        )
        gc.collect()
        return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print(
            "#####################\nCalibration jobs are finished unsuccessfully.\n#####################\n"
        )
        return 1


def perform_all_applycal(
    msdir, bcaldir, kcrossdir, basedir, do_flag=True, cpu_percent=10, mem_percent=20
):
    """
    Apply calibration solutions of all target ms
    Parameters
    ----------
    msdir : str
        Name of the target ms directory
    bcaldir : str
        Name of bandpass calibration table directory
    kcrossdir : str
        Name of crosshand phase calibration table directory
    basedir : str
        Base directory
    do_flag : bool
        Do flagging or not
    cpu_percent : float
        Free CPU percentage
    mem_percent : float
        Free memory percentage
    Returns
    -------
    int
        Success message (0 or 1)
    """
    print("Apply calibration solution jobs are being started ....\n")
    try:
        os.system("rm -rf " + basedir + "/.Finished_applycal*")
        mslist = glob.glob(msdir + "/*.ms")
        bcal_tables = glob.glob(bcaldir + "/*.bcal")
        kcross_tables = glob.glob(kcrossdir + "/*.kcross")
        trial_ms = mslist[0]
        mssize = get_column_size(trial_ms, "DATA")  # In GB
        total_memory = psutil.virtual_memory().available / (1024**3)  # In GB
        max_jobs = int(total_memory / (2 * mssize))
        total_cpus = psutil.cpu_count()
        ncpu = int(total_cpus / max_jobs)
        if ncpu < 1:
            ncpu = 1
            max_jobs=total_cpus 
        print("Maximum numbers of jobs to spawn at once:", max_jobs)
        count = 0
        free_jobs = -1
        for ms in mslist:
            ms_obsid = int(os.path.basename(ms).split(".ms")[0].split("_")[0])
            mssize = get_column_size(ms, "DATA")  # In GB
            tb = table()
            tb.open(ms + "/SPECTRAL_WINDOW")
            freq = tb.getcol("CHAN_FREQ")
            tb.close()
            start_coarse_chan = freq_to_MWA_coarse(freq[0] / 10**6)
            end_coarse_chan = freq_to_MWA_coarse(freq[-1] / 10**6)
            coarse_chan_str = str(start_coarse_chan) + "_" + str(end_coarse_chan)
            bcal = ""
            kcross = ""
            selected_bcals = []
            for caltable in bcal_tables:
                if coarse_chan_str in caltable:
                    selected_bcals.append(caltable)
            if (
                len(selected_bcals) > 1
            ):  # Selecting nearest time bcal tables if has many bcal tables
                bcal_obsids = np.array(
                    [int(os.path.basename(i).split("_ref")[0]) for i in selected_bcals]
                )
                pos = np.argmin(np.abs(bcal_obsids - ms_obsid))
                bcal = selected_bcals[pos]
            else:
                bcal = selected_bcals[0]
            selected_kcrosss = []
            for caltable in kcross_tables:
                if coarse_chan_str in caltable:
                    selected_kcrosss.append(caltable)
            if (
                len(selected_kcrosss) > 1
            ):  # Selecting nearest time kcross tables if has many kcross tables
                kcross_obsids = np.array(
                    [
                        int(os.path.basename(i).split("_ref")[0])
                        for i in selected_kcrosss
                    ]
                )
                pos = np.argmin(np.abs(kcross_obsids - ms_obsid))
                kcross = selected_kcrosss[pos]
            else:
                kcross = selected_kcrosss[0]
            if bcal == "" and kcross == "":
                print("Caltable(s) for the same coarse channels do(es) not exist.\n")
            else:
                cmd = (
                    "python3 apply_solutions.py --msname "
                    + ms
                    + " --do_flag "
                    + str(do_flag)
                )
                if bcal != "":
                    cmd += " --bandpass_table " + str(bcal)
                if kcross != "":
                    cmd += " --kcross_table " + str(kcross)
                basename = (
                    "applycal_" + os.path.basename(ms).split(".ms")[0] + "_bcal_kcross"
                )
                batch_file = create_batch_script_nonhpc(cmd, basedir, basename)
                os.system("bash " + batch_file)
                print("Spawned command: " + cmd + "\n")
                count += 1
                if free_jobs > 0:
                    free_jobs -= 1
                if count >= max_jobs or free_jobs == 0:
                    free_jobs = wait_for_resources(
                        basedir + "/.Finished_applycal",
                        cpu_threshold=cpu_percent,
                        memory_threshold=mem_percent,
                    )
                    print("Freed jobs: ", free_jobs)
        while True:
            finished_files = glob.glob(basedir + "/.Finished_applycal*")
            if len(finished_files) >= count:
                break
        print(
            "#####################\nApply calibration solution jobs are finished successfully.\n#####################\n"
        )
        gc.collect()
        return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print(
            "#####################\nApply calibration solution jobs are finished unsuccessfully.\n#####################\n"
        )
        return 1


def perform_all_spectral_imaging(
    msdir,
    basedir,
    nchan,
    multiscale_scales=[],
    weight="briggs",
    robust=0.0,
    threshold=6,
    pol="IQUV",
    FWHM=True,
    minuv_l=-1,
    cpu_percent=10,
    mem_percent=20,
):
    print("Imaging jobs are started ....\n")
    try:
        os.system("rm -rf " + basedir + "/.Finished_imaging*")
        mslist = glob.glob(msdir + "/*.ms")
        trial_ms = mslist[0]
        mssize = get_column_size(trial_ms, "DATA")  # In GB
        total_memory = psutil.virtual_memory().available / (1024**3)  # In GB
        max_jobs = int(total_memory / (2 * mssize))
        print("Maximum numbers of jobs to spawn at once:", max_jobs)
        count = 0
        free_jobs = -1
        absmem = total_memory / max_jobs
        available_cpu = int(psutil.cpu_count() * (100 - psutil.cpu_percent()) / 100.0)
        ncpu = int(available_cpu / max_jobs)
        if ncpu < 1:
            ncpu = 1
            max_jobs=total_cpus     
        scales = ",".join([str(i) for i in multiscale_scales])
        for ms in mslist:
            cmd = (
                "python3 do_imaging.py --msname "
                + ms
                + " --nchan "
                + str(nchan)
                + " --imagedir "
                + str(basedir)
                + " --weight "
                + weight
                + " --robust "
                + str(robust)
                + " --threshold "
                + str(threshold)
                + " --pol "
                + str(pol)
                + " --FWHM "
                + str(FWHM)
                + " --ncpu "
                + str(ncpu)
                + " --mem "
                + str(absmem)
            )
            if len(multiscale_scales) > 0:
                cmd += " --multiscale_scales " + scales
            if minuv_l > 0:
                cmd += " --minuv_l " + str(minuv_l)
            basename = (
                "imaging_"
                + os.path.basename(ms).split(".ms")[0]
                + "_nchan_"
                + str(nchan)
            )
            batch_file = create_batch_script_nonhpc(cmd, basedir, basename)
            os.system("bash " + batch_file)
            print("Spawned command: " + cmd + "\n")
            count += 1
            if free_jobs > 0:
                free_jobs -= 1
            if count >= max_jobs or free_jobs == 0:
                free_jobs = wait_for_resources(
                    basedir + "/.Finished_imaging",
                    cpu_threshold=cpu_percent,
                    memory_threshold=mem_percent,
                )
                print("Freed jobs: ", free_jobs)
        while True:
            finished_files = glob.glob(basedir + "/.Finished_imaging*")
            if len(finished_files) >= count:
                break
        print(
            "#####################\nImaging jobs are finished successfully.\n#####################\n"
        )
        gc.collect()
        return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print(
            "#####################\nImaging jobs are finished unsuccessfully.\n#####################\n"
        )
        return 1

def make_all_ddcal(msdir, image_basedir, ncpu=-1, mem=-1):
    """
    Estimate all direction dependent calibration in image plane
    Parameters
    ----------
    msdir : str
        Name of the measurement set directory
    image_basedir : str
        Image base directory name
    ncpu : int
        Number of CPU threads to use
    mem : float
        Memory to use in GB
    Returns
    -------
    dict
        A dictionary containing information of solution directories for all measurement sets
    """       
    msmd = msmetadata()
    mslist = glob.glob(msdir + "/*.ms")
    ddcal_dirs = {}
    with tempfile.TemporaryDirectory(dir=image_basedir) as tempdir:
        os.environ["JOBLIB_TEMP_FOLDER"] = tempdir
        try:
            for ms in mslist:
                ddcal_dir_list = []
                ms_obsid = os.path.basename(ms).split(".ms")[0]
                metafits = msdir + "/" + ms_obsid + ".metafits"
                msmd.open(ms)
                total_coarsechan = int(msmd.bandwidths(0) / (1280*1000))
                msmd.close()
                total_coarsechan=10
                imagedir = glob.glob(
                    image_basedir + "/imagedir_MFS_ch_" + str(total_coarsechan) + "_" + str(ms_obsid)
                )
                if len(imagedir)==0:
                    print ("No suitable image directory with coarse channel: "+str(total_coarsechan)+" and ObsID: "+str(ms_obsid))
                else:
                    imagedir=imagedir[0]
                    os.system("rm -rf "+imagedir+"/*_I*")    
                    image_list = glob.glob(imagedir + "/images/*.fits")
                    filtered_image_list=[]
                    for image in image_list:
                        if 'MFS' not in os.path.basename(image):
                            filtered_image_list.append(image)
                    image_list=filtered_image_list 
                    if len(image_list)==0:
                        print ("No images in image direcory: "+imagedir + "/images/")
                    else:       
                        ######################################
                        # Primary beam correction
                        ######################################
                        print("#######################")
                        print("Estimating primary beams ....")
                        print("#######################")
                        pbcor_image_dir, pb_dir, total_images = correctpb_spectral_images(
                            imagedir + "/images", metafits, interpolate=True, ncpu=ncpu, mem=mem
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
                        with Parallel(n_jobs=n_jobs, backend='multiprocessing') as parallel:
                            results= parallel(delayed(estimate_warp_map)(imagename, outdir=warp_outdir,allsky_cat=source_model_fits) for imagename in image_list)
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
                                warp_outdir
                                + "/"
                                + os.path.basename(imagename).split(".fits")[0]
                                + "*bkg*"
                            )
                            if len(bkg_image) > 0:
                                bkg_image = bkg_image[0]
                            else:
                                bkg_image = ""
                            rms_image = glob.glob(
                                warp_outdir
                                + "/"
                                + os.path.basename(imagename).split(".fits")[0]
                                + "*rms*"
                            )
                            if len(rms_image) > 0:
                                rms_image = rms_image[0]
                            else:
                                rms_image = ""
                            bkg_image_list.append(bkg_image)
                            rms_image_list.append(rms_image)
                        with Parallel(n_jobs=n_jobs, backend='multiprocessing') as parallel:
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
                        ddcal_dirs[ms_obsid] = ddcal_dir_list
            gc.collect()
            return 0,ddcal_dirs        
        except Exception as e:
            traceback.print_exc()
            gc.collect()
            print(
                "#####################\nDirection dependent calibration jobs are finished unsuccessfully.\n#####################\n"
            )
            return 1, ddcal_dirs
        finally:
            gc.collect()
            return 0,ddcal_dirs     
       
    


################################
def main():
    usage = "Master controller for MWA Polcal pipeline"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--calms_dir",
        dest="calms_dir",
        default=None,
        help="Name of the calibrator measurement set directory",
        metavar="String",
    )
    parser.add_option(
        "--bcal_dir",
        dest="bcal_dir",
        default=None,
        help="Name of the bandpass calibration solutions directory",
        metavar="String",
    )
    parser.add_option(
        "--kcross_dir",
        dest="kcross_dir",
        default=None,
        help="Name of the crosshand phase calibration solutions directory",
        metavar="String",
    )
    parser.add_option(
        "--targetms_dir",
        dest="targetms_dir",
        default=None,
        help="Name of the target measurement set directory",
        metavar="String",
    )
    parser.add_option(
        "--basedir",
        dest="basedir",
        default=None,
        help="Name of the base working directory",
        metavar="String",
    )
    parser.add_option(
        "--refant",
        dest="refant",
        default=1,
        help="Reference antenna",
        metavar="Integer",
    )
    parser.add_option(
        "--do_kcross",
        dest="do_kcross",
        default=True,
        help="Perform crosshand phase calibration or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--do_target_flag",
        dest="do_target_flag",
        default=True,
        help="Perform target flagging or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--import_model",
        dest="import_model",
        default=True,
        help="Import model or not",
        metavar=True,
    )
    parser.add_option(
        "--free_cpu_percent",
        dest="cpu_percent",
        default=10,
        help="Amount of free CPU percentage",
        metavar="Float",
    )
    parser.add_option(
        "--free_mem_percent",
        dest="mem_percent",
        default=20,
        help="Amount of free memory percentage",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if (
        options.calms_dir == None
        and options.bcal_dir == None
        and options.kcross_dir == None
    ):
        print("No calibrator observations or solutions are provided.\n")
        return 1
    if options.basedir == None:
        print("Please provide a base directory name.\n")
        return 1
    elif os.path.exists(options.basedir) == False:
        os.makedirs(options.basedir)

    if options.calms_dir != None:
        os.system("rm -rf " + options.calms_dir + "/*model.ms*")
        os.system("rm -rf " + options.calms_dir + "/*.bcal")
        if eval(str(options.import_model)) == True:
            msg = perform_model_import(
                options.calms_dir,
                options.basedir,
                cpu_percent=float(options.cpu_percent),
                mem_percent=float(options.mem_percent),
            )
            gc.collect()
        else:
            msg = 0
        if msg == 1:
            return 1
        else:
            msg = perform_all_calibration(
                options.calms_dir,
                options.basedir,
                refant=int(options.refant),
                do_kcross=eval(str(options.do_kcross)),
                cpu_percent=float(options.cpu_percent),
                mem_percent=float(options.mem_percent),
            )
            gc.collect()
            if msg == 1:
                return 1
            elif options.targetms_dir != None:
                if options.bcal_dir == None:
                    bcaldir = options.basedir + "/caldir"
                else:
                    bcaldir = options.bcal_dir
                if options.kcross_dir == None:
                    kcrossdir = options.basedir + "/caldir"
                else:
                    kcrossdir = options.kcross_dir
                msg = perform_all_applycal(
                    options.targetms_dir,
                    bcaldir,
                    kcrossdir,
                    options.basedir,
                    do_flag=eval(str(options.do_target_flag)),
                    cpu_percent=float(options.cpu_percent),
                    mem_percent=float(options.mem_percent),
                )
                gc.collect()
                if msg == 1:
                    return 1
                else:
                    return 0
            else:
                return 0
    elif options.targetms_dir != None:
        msg = perform_all_applycal(
            options.targetms_dir,
            options.bcal_dir,
            options.kcross_dir,
            options.basedir,
            do_flag=eval(str(options.do_target_flag)),
            cpu_percent=float(options.cpu_percent),
            mem_percent=float(options.mem_percent),
        )
        gc.collect()
        if msg == 1:
            return 1
        else:
            return 0


if __name__ == "__main__":
    result = main()
    os._exit(result)
    gc.collect()
