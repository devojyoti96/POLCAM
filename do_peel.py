from do_imaging import *
from basic_func import *
from casatasks import delmod, uvsub
import gc, traceback
from optparse import OptionParser


def peel_source(
    msname,
    metafits,
    basedir,
    source_name,
    ra,
    dec,
    use_multiscale=True,
    multiscale_scales="",
    weight="briggs",
    robust=0.0,
    threshold=3,
    niter=100000,
    minuv_l=-1,
    ncpu=-1,
    mem=-1,
):
    """
    Peel a single source
    Parameters
    ----------
    msname : str
        Name of the measurement set
    metafits : str
        Name of the metafits file
    basedir : str
        Base directory
    source_name : str
        Source name to be peeled
    ra : str
        Source RA to be peeled in hms format
    dec : str
        Source DEC to be peeled in dms format
    use_multiscale : bool
        Use multiscale cleaning or not
    multiscale_scales : str
        Multiscale scales in pixels unit seperated by comma
    weight : str
        Image weighting scheme
    robust : float
        Briggs weighting robust parameter
    threshold : float
        Cleaning threshold
    niter : int
        Number of cleaning iterations
    minuv_l : float
        Minimum UV-lambda
    ncpu : int
        Number of CPU-threads to be used
    mem : float
        Amount of memory in GB to be used
    Returns
    -------
    int
        Success message
    """
    print("############################")
    print("Peeling source: ", source_name)
    header = fits.getheader(metafits)
    try:
        org_ra_deg = float(header["RAPHASE"])
        org_dec_deg = float(header["DECPHASE"])
    except:
        print(
            "Phase center information is not present. Hence, choosing pointing center as phase center."
        )
        org_ra_deg = float(header["RA"])
        org_dec_deg = float(header["DEC"])
    org_ra, org_dec = ra_dec_to_hms_dms(org_ra_deg, org_dec_deg)
    print("Shifting phase center to the source.")
    os.system(
        "chgcentre " + msname + " " + ra + " " + dec + " > chgcentre.log 2>/dev/null"
    )
    os.system("rm -rf chgcentre.log")
    delmod(vis=msname, otf=False, scr=True)
    if source_name.lower() == "sun":
        imsize = 3
        multiscale_scales = ",".join(
            [str(i) for i in calc_multiscale_scales(msname, 3, max_scale=16)]
        )
        use_multiscale = True
    else:
        imsize = 1
    msg, imagedir, imageprefix = perform_spectrotemporal_imaging(
        msname,
        freqres=1280,
        ntime=1,
        imagedir=basedir + "/peel_" + source_name,
        use_multiscale=use_multiscale,
        multiscale_scales=multiscale_scales,
        weight=weight,
        robust=robust,
        pol="IQUV",
        imsize=imsize,
        threshold=threshold,
        niter=niter,
        minuv_l=minuv_l,
        savemodel=True,
        ncpu=ncpu,
        mem=mem,
    )
    if msg == 0:
        print("Performing UV-subtraction.")
        uvsub(vis=msname, reverse=False)
        print("Shifting phase center to original phasecenter.")
        os.system(
            "chgcentre "
            + msname
            + " "
            + org_ra
            + " "
            + org_dec
            + " > chgcentre.log 2>/dev/null"
        )
        os.system("rm -rf chgcentre.log")
        gc.collect()
        if os.path.exists(imagedir):
            os.system("rm -rf " + imagedir)
        return 0
    else:
        print("Error in imaging and peeling.")
        print("Shifting phase center to original phasecenter.")
        os.system(
            "chgcentre "
            + msname
            + " "
            + org_ra
            + " "
            + org_dec
            + " > chgcentre.log 2>/dev/null"
        )
        os.system("rm -rf chgcentre.log")
        gc.collect()
        if os.path.exists(imagedir):
            os.system("rm -rf " + imagedir)
        return 1


def run_peel(
    msname,
    metafits,
    basedir,
    MWA_PB_file,
    sweet_spot_file,
    min_beamgain=0.001,
    threshold_flux=0.0,
    use_multiscale=True,
    multiscale_scales="",
    weight="briggs",
    robust=0.0,
    threshold=3,
    niter=100000,
    minuv_l=-1,
    modify_final_datacolumn=False,
    ncpu=-1,
    mem=-1,
    force_peel=False,
):
    """
    Peel all a-team sources and Sun in the measurement set
    Parameters
    ----------
    msname : str
        Name of the measurement set
    metafits : str
        Name of the metafits file
    basedir : str
        Base directory
    MWA_PB_file : str
        MWA primary beam file
    sweet_spot_file : str
        MWA sweetspot file
    min_beamgain : float
        Minimum beam gain at the source to be considered for peeling
    threshold_flux : float
        Apprent flux density threshold for the source to be considered for peeling
    use_multiscale : bool
        Use multiscale cleaning or not
    multiscale_scales : str
        Multiscale scales in pixels unit seperated by comma
    weight : str
        Image weighting scheme
    robust : float
        Briggs weighting robust parameter
    threshold : float
        Cleaning threshold
    niter : int
        Number of cleaning iterations
    minuv_l : float
        Minimum UV-lambda
    modify_final_datacolumn : bool
        Whether modify the datacolumn at the end with the self-calibrated data or not
    ncpu : int
        Number of CPU-threads to be used
    mem : float
        Amount of memory in GB to be used
    force_peel : bool
        Force peeling even it is aleady peeled
    Returns
    -------
    int
        Success message
    """
    if os.path.exists(msname + "/.peeled") and force_peel == False:
        print("Peeling is done on ms: ", msname)
        return 0
    peel_source_dic = get_ateam_sources(
        msname,
        metafits,
        MWA_PB_file,
        sweet_spot_file,
        min_beam_threshold=min_beamgain,
        threshold_flux=threshold_flux,
    )
    if len(peel_source_dic) == 0:
        print("No source to peel.")
        return 0
    else:
        source_names = list(peel_source_dic.keys())
        for source in source_names:
            source_ra, source_dec, alt, az, appflux = peel_source_dic[source]
            peel_source(
                msname,
                metafits,
                basedir,
                source,
                source_ra,
                source_dec,
                use_multiscale=use_multiscale,
                multiscale_scales=multiscale_scales,
                weight=weight,
                robust=robust,
                threshold=threshold,
                niter=niter,
                minuv_l=minuv_l,
                ncpu=ncpu,
                mem=mem,
            )
        if modify_final_datacolumn:  # Modify datacolumn with the self-calibrated data
            tb.open(msname, nomodify=False)
            cor_data = tb.getcol("CORRECTED_DATA")
            tb.putcol("DATA", cor_data)
            tb.flush()
            tb.close()
        os.system("touch " + msname + "/.peeled")
        return 0


################################
def main():
    usage = "Perform source peeling"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
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
        "--basedir",
        dest="basedir",
        default=None,
        help="Name of the base directory",
        metavar="String",
    )
    parser.add_option(
        "--MWA_PB_file",
        dest="MWA_PB_file",
        default="mwa_full_embedded_element_pattern.h5",
        help="Name of the MWA primary beam file",
        metavar="String",
    )
    parser.add_option(
        "--sweet_spot_file",
        dest="sweet_spot_file",
        default="MWA_sweet_spots.npy",
        help="Name of the MWA sweet spot file",
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
        "--min_beamgain",
        dest="min_beamgain",
        default=0.001,
        help="Minimum primary beam gain to consider the source to peel",
        metavar="Float",
    )
    parser.add_option(
        "--threshold_flux",
        dest="threshold_flux",
        default=0.0,
        help="Minimum apparent flux density threshold to consider the source to peel",
        metavar="Float",
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
        help="Image weighiting scheme",
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
        default=100000,
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
        "--threshold",
        dest="threshold",
        default=3.0,
        help="Cleaning threshold",
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
        "--force_peel",
        dest="force_peel",
        default=False,
        help="Force peeling even ms is saying it is already peeled",
        metavar="Boolean",
    )
    (options, args) = parser.parse_args()
    if options.msname == None or os.path.exists(options.msname) == False:
        print("Please provide correct measurement set name.\n")
        return 1
    if options.metafits == None or os.path.exists(options.metafits) == False:
        print("Please provide correct metafits name.\n")
        return 1
    if options.basedir == None or os.path.exists(options.basedir) == False:
        print("Please provide correct base directory name.\n")
        return 1
    msg = run_peel(
        options.msname,
        options.metafits,
        options.basedir,
        options.MWA_PB_file,
        options.sweet_spot_file,
        min_beamgain=float(options.min_beamgain),
        threshold_flux=float(options.threshold_flux),
        use_multiscale=eval(str(options.use_multiscale)),
        multiscale_scales=options.multiscale_scales,
        weight=options.weight,
        robust=float(options.robust),
        threshold=float(options.threshold),
        niter=int(options.niter),
        minuv_l=float(options.minuv_l),
        modify_final_datacolumn=eval(str(options.modify_datacolumn)),
        ncpu=int(options.ncpu),
        mem=float(options.absmem),
        force_peel=eval(str(options.force_peel)),
    )
    if msg == 0:
        print("Source peeling is completed successfully.")
    else:
        print("Source peeling is completed unsuccessfully. Issues occured.")
    gc.collect()
    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)
