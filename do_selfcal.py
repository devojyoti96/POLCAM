import os, gc, traceback
from casatasks import gaincal, applycal, delmod, imstat, split
from casatools import table
from do_imaging import perform_spectrotemporal_imaging
from basic_func import get_calibration_uvrange
from optparse import OptionParser


def perform_single_selfcal(
    msname,
    selfcaldir,
    iteration=0,
    calmode="p",
    gaintype="T",
    solint="int",
    applymode = "calonly",
    use_multiscale=True,
    multiscale_scales="",
    weight="briggs",
    robust=0.0,
    threshold=3,
    mask_threshold=5,
    niter=100000,
    minuv_l=-1,
    continue_imaging=False,
    ncpu=-1,
    mem=-1,
):
    """
    Perform a single self-calibration round
    Parameters
    ----------
    msname : str
        Name of the measurement set
    selfcaldir : str
        Name of the base directory for selfcal
    iteration : int
        Self-cal iteration number
    calmode : str
        Calibration mode
    gaintype : str
        Gain type for calibration
    solint : str
        Solution interval
    applymode : str
        Solution apply mode
    use_multiscale : bool
        Whether to use multiscale cleaning or not
    multiscale_scales : str
        Multiscale scales seperated by comma
    weight : str
        Image weighting
    robust : float
        Briggs weighting robust parameter
    threshold : float
        CLEAN threshold
    mask_threshold : float
        Auto mask threshold
    niter : int
        Numbers of clean iteration
    minuv_l : float
        Minimum UV-lambda
    continue_imaging : bool
        Continue imaging using previous model
    ncpu : int
        Number of CPUs to use
    mem : float
        Amount of memory in GB to use
    Returns
    -------
    int
        Success message
    float
        Dynamic range of final round image
    str
        Caltable name
    """
    if os.path.exists(selfcaldir + "/selfcal_" + str(iteration))==False:
        os.makedirs(selfcaldir + "/selfcal_" + str(iteration))
    else:
        os.system("rm -rf "+selfcaldir + "/selfcal_" + str(iteration)+"/*")
    print("#####################")
    print("Selfcal iteration: " + str(iteration))
    print("#####################")
    print("Doing imaging ...")
    delmod(vis=msname, scr=True)
    msg, imagedir, imageprefix = perform_spectrotemporal_imaging(
        msname,
        freqres=1280,
        ntime=1,
        imagedir=selfcaldir + "/selfcal_" + str(iteration),
        use_multiscale=use_multiscale,
        multiscale_scales=multiscale_scales,
        weight=weight,
        robust=robust,
        pol="I",
        FWHM=True,
        threshold=threshold,
        mask_threshold=mask_threshold,
        niter=niter,
        minuv_l=minuv_l,
        savemodel=True,
        ncpu=ncpu,
        mem=mem,
        continue_imaging=continue_imaging,
    )
    mfs_imagename = imagedir + "/" + imageprefix + "-MFS-image.fits"
    mfs_resname = imagedir + "/" + imageprefix + "-MFS-residual.fits"
    maxval = imstat(imagename=mfs_imagename)["max"][0]
    rms = imstat(imagename=mfs_resname)["rms"][0]
    DR = round(maxval / rms, 2)
    caltable_name = (
        selfcaldir + "/selfcal_" + str(iteration) + "/selfcal_" + str(iteration) + ".gcal"
    )
    print("Doing gaincal ...")
    if minuv_l > 0:
        uvrange = ">" + str(minuv_l) + "lambda"
    else:
        uvrange = get_calibration_uvrange(msname)
    gaincal(
        vis=msname,
        caltable=caltable_name,
        uvrange=uvrange,
        calmode=calmode,
        gaintype=gaintype,
        solnorm=True,
        solmode="R",
        solint=solint,
        rmsthresh=[10, 7, 5, 3.5],
    )
    print("Applying solutions ...")
    applycal(vis=msname, gaintable=[caltable_name], applymode=applymode, calwt=[True])
    print("Dynamic range ", DR)
    return 0, DR, caltable_name


def run_selfcal_iterations(
    msname,
    basedir,
    modify_final_datacolumn=False,
    max_iteration=5,
    solint="inf",
    gaintype="T",
    applymode = "calonly",
    use_multiscale=True,
    multiscale_scales="",
    weight="briggs",
    robust=0.0,
    niter=100000,
    minuv_l=-1,
    ncpu=-1,
    mem=-1,
    keep_selfcal_rounds=False,
):
    """
    Perform self-calibration
    Parameters
    ----------
    msname : str
        Name of the measurement set
    basedir : str
        Name of the base directory
    modify_final_datacolumn : bool
        Whether modify the datacolumn at the end with the self-calibrated data or not
    max_iteration : int
        Maximum numbers of self-cal iterations
    solint : str
        Solution interval
    applymode : str
        Solution apply mode
    gaintype : str
        Gain type for calibration
    use_multiscale : bool
        Whether to use multiscale cleaning or not
    multiscale_scales : str
        Multiscale scales seperated by comma
    weight : str
        Image weighting
    robust : float
        Briggs weighting robust parameter
    niter : int
        Numbers of clean iteration
    minuv_l : float
        Minimum UV-lambda
    ncpu : int
        Number of CPUs to use
    mem : float
        Amount of memory in GB to use
    keep_selfcal_rounds : bool
        Keep selfcal round images and calibration tables
    Returns
    -------
    int
        Success message
    float
        Dynamic range of the final image
    str
        Final self-calibrated msname
    """
    try:
        continue_imaging=False
        selfcal_basedir = (
            basedir + "/selfcal_" + os.path.basename(msname).split(".ms")[0]
        )
        tb = table()
        tb.open(msname)
        colnames = tb.colnames()
        tb.close()
        outputms = basedir+'/'+os.path.basename(msname).split(".ms")[0] + "_selfcal.ms"
        if os.path.exists(outputms) and os.path.exists(outputms+"/.selfcal") and force_selfcal==False:
            print("Selfcal is done on ms: ", msname)
            return 0,outputms
        else:    
            os.system("rm -rf "+outputms)
        if "CORRECTED_DATA" in colnames:
            if os.path.exists(msname.split(".ms")[0] + "_selfcal.ms") == False:
                print("Spliting corrected column of ms: ", os.path.basename(msname))
                split(
                    vis=msname,
                    outputvis=outputms,
                    datacolumn="corrected",
                )
        else:
            os.system("cp -r " + msname + " " + outputms)
        msname = outputms
        start_threshold = 5
        selfcal_iter = 0
        DR0 = 0
        if float(minuv_l) < 0:
            uvrange = get_calibration_uvrange(msname)
            minuv_l = float(uvrange.split("~")[0])
        if os.path.exists(selfcal_basedir) == False:
            os.makedirs(selfcal_basedir)
        else:
            os.system("rm -rf "+selfcal_basedir+"/*")    
        while selfcal_iter < max_iteration:
            msg, DR1, caltable_name = perform_single_selfcal(
                msname,
                selfcal_basedir,
                iteration=selfcal_iter,
                calmode="ap",
                solint=solint,
                applymode=applymode,
                gaintype=gaintype,
                use_multiscale=use_multiscale,
                multiscale_scales=multiscale_scales,
                weight=weight,
                robust=robust,
                threshold=3,
                mask_threshold=start_threshold,
                continue_imaging=continue_imaging,
                niter=niter,
                minuv_l=minuv_l,
                ncpu=ncpu,
                mem=mem,
            )
            if selfcal_iter > 0:
                if DR1 > DR0 and (DR1 - DR0) / DR0 < 0.2:
                    if start_threshold > 3:
                        start_threshold -= 1
                        print("Reducing CLEAN threshold to: ", start_threshold)
                        continue_imaging=True
                    else:
                        print("Self-cal has converged.")
                        break
                elif DR1 < DR0 and (DR0 - DR1) / DR0 > 0.2:
                    print("Dynamic range decreases.")
                    continue_imaging=False
                    break
            DR0 = DR1
            selfcal_iter += 1
        if modify_final_datacolumn:  # Modify datacolumn with the self-calibrated data
            tb.open(msname, nomodify=False)
            cor_data = tb.getcol("CORRECTED_DATA")
            tb.putcol("DATA", cor_data)
            tb.flush()
            tb.close()
        gc.collect()
        os.system("touch "+msname+"/.selfcal")
        if keep_selfcal_rounds==False and os.path.exists(selfcal_basedir):
            os.system("rm -rf "+selfcal_basedir)
        return 0, DR1, msname
    except Exception as e:
        if keep_selfcal_rounds==False and os.path.exists(selfcal_basedir):
            os.system("rm -rf "+selfcal_basedir)
        traceback.print_exc()
        gc.collect()
        return 1, None, None


################################
def main():
    usage = "Perform self-calibartion"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
        metavar="String",
    )
    parser.add_option(
        "--basedir",
        dest="basedir",
        default=None,
        help="Name of the base directory",
        metavar="String",
    )
    parser.add_option(
        "--modify_datacolumn",
        dest="modify_datacolumn",
        default=False,
        help="Modify data column with self-calibrated data or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--solint",
        dest="solint",
        default="inf",
        help="Solution interval",
        metavar="String",
    )
    parser.add_option(
        "--applymode",
        dest="applymode",
        default="calonly",
        help="Solution apply mode",
        metavar="String",
    )
    parser.add_option(
        "--gaintype",
        dest="gaintype",
        default="T",
        help="Gain type for calibration (G or T)",
        metavar="String",
    )
    parser.add_option(
        "--max_iter",
        dest="max_iter",
        default=5,
        help="Maximum self-calibration iteration",
        metavar="Integer",
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
        help="Multiscale scales seperated in comma",
        metavar="String",
    )
    parser.add_option(
        "--weight",
        dest="weight",
        default="briggs",
        help="Image weighting scheme",
        metavar="String",
    )
    parser.add_option(
        "--robust",
        dest="robust",
        default=0.0,
        help="Briggs weighting robust value",
        metavar="Float",
    )
    parser.add_option(
        "--niter",
        dest="niter",
        default=5000,
        help="Number of clean iteration",
        metavar="Integer",
    )
    parser.add_option(
        "--minuv_l",
        dest="minuv_l",
        default=-1,
        help="Minimum UV in lambda",
        metavar="Float",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Number of CPU threads to use (default : -1, determine automatically)",
        metavar="Integer",
    )
    parser.add_option(
        "--absmem",
        dest="absmem",
        default=-1,
        help="Amount of memory to use in GB (default : -1, determine automatically)",
        metavar="Float",
    )
    parser.add_option(
        "--keep_selfcal_rounds",
        dest="keep_selfcal_rounds",
        default=False,
        help="Keep selfcal rounds",
        metavar="Boolean",
    )
    (options, args) = parser.parse_args()
    if options.msname == None or os.path.exists(options.msname) == False:
        print("Please provide correct measurement set name.\n")
        return 1
    if options.basedir == None or os.path.exists(options.basedir) == False:
        print("Please provide correct base directory name.\n")
        return
    msg, DR, final_msname = run_selfcal_iterations(
        options.msname,
        options.basedir,
        modify_final_datacolumn=eval(str(options.modify_datacolumn)),
        max_iteration=int(options.max_iter),
        solint=options.solint,
        applymode=options.applymode,
        gaintype=options.gaintype,
        use_multiscale=eval(str(options.use_multiscale)),
        multiscale_scales=options.multiscale_scales,
        weight=options.weight,
        robust=float(options.robust),
        niter=int(options.niter),
        minuv_l=float(options.minuv_l),
        ncpu=int(options.ncpu),
        mem=float(options.absmem),
        keep_selfcal_rounds=eval(str(options.keep_selfcal_rounds)),
    )
    if msg == 0:
        print("Self-calibration is completed successfully.")
        print("Final image dynamic range : ", DR)
        print("Final msname: ", final_msname)
    else:
        print("Self-calibration is completed unsuccessfully. Issues occured.")
    gc.collect()
    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)
