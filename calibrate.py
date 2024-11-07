from casatasks import bandpass, flagdata
from casatools import table
from calibrate_crossphase import crossphasecal
from optparse import OptionParser
from basic_func import *
import os

os.system("rm -rf casa*log")


def do_flag_cal(msname, refant, uvrange=""):
    """
    Parameters
    ----------
    msname : str
        Name of the measurement set
    refant : str
        Reference antenna index or name
    uvrange : str
        UV-range to be used for calibration
    Returns
    -------
    int
        Success or failure message
    str
        Bandpass caltable
    str
        Crosshand phase caltable
    """
    try:
        print("Flagging: " + msname)
        flagdata(vis=msname, mode="tfcrop")
        tb = table()
        tb.open(msname + "/SPECTRAL_WINDOW")
        freq = tb.getcol("CHAN_FREQ")
        tb.close()
        start_coarse_chan = freq_to_MWA_coarse(freq[0] / 10**6)
        end_coarse_chan = freq_to_MWA_coarse(freq[-1] / 10**6)
        caltable_prefix = (
            msname.split(".ms")[0]
            + "_ref_"
            + str(refant)
            + "_ch_"
            + str(start_coarse_chan)
            + "_"
            + str(end_coarse_chan)
        )
        print(
            "################\nCalibrating ms :"
            + msname
            + "\n#######################\n"
        )
        if uvrange=='':
            uvrange=get_calibration_uvrange(msname)   
        bandpass(
            vis=msname,
            caltable=caltable_prefix + ".bcal",
            refant=str(refant),
            solint="inf",
            uvrange=uvrange,
        )
        print("Estimating crosshand phase...\n")
        crossphase_caltable = crossphasecal(
            msname,
            uvrange=uvrange,
            caltable=caltable_prefix + ".kcross",
            gaintable=[caltable_prefix + ".bcal"],
        )
        return 0, caltable_prefix + ".bcal", crossphase_caltable
    except Exception as e:
        print ('Exception: ',e)
        return 1, None, None


################################
def main():
    usage = "Flag and calibrate"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
        metavar="String",
    )
    parser.add_option(
        "--refant",
        dest="refant",
        default="1",
        help="Reference antenna",
        metavar="String",
    )
    parser.add_option(
        "--uvrange",
        dest="uvrange",
        default="",
        help="UV-range for calibration",
        metavar="String",
    )
    parser.add_option(
        "--caldir",
        dest="caldir",
        default=None,
        help="Caltable directory",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    if options.msname == None:
        print("Please provide the measurement set name.\n")
        return 1
    if options.caldir == None:
        caldir = os.path.dirname(options.msname) + "/caltables"
    else:
        caldir = options.caldir
    if os.path.exists(caldir) == False:
        os.makedirs(caldir)
    msg, bcal, kcrosscal = do_flag_cal(
        options.msname, options.refant, uvrange=str(options.uvrange)
    )
    if msg == 0:
        os.system('rm -rf '+caldir+'/'+os.path.basename(bcal))
        os.system("mv " + bcal + " " + caldir)
        os.system('rm -rf '+caldir+'/'+os.path.basename(kcrosscal))
        os.system("mv " + kcrosscal + " " + caldir)
        print("Caltable names: " + str(bcal) + "," + str(kcrosscal))
    else:
        print ("Issues occured")    
    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)
