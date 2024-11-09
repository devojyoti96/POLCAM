from casatasks import applycal, flagdata
from calibrate_crossphase import apply_crossphasecal
from optparse import OptionParser
import os

os.system("rm -rf casa*log")


def apply_sol(
    msname, bandpass_table, kcross_table, applymode="calflag", flagbackup=True
):
    """
    Apply bandpass and crosshand phase solutions
    Parameters
    ----------
    msname : str
        Name of the measurement set
    bandpass_table : str
        Bandpass table name
    kcross_table : str
        Crosshand phase table
    applymode : str
        Solution apply and flag
    flagbackup : bool
        Keep flag backup before applying solutions
    """
    try:
        print("Applying bandpass solutons...\n")
        applycal(
            vis=msname,
            gaintable=[bandpass_table],
            applymode=applymode,
            flagbackup=flagbackup,
        )
        print("Applying crosshand phase solutions...\n")
        apply_crossphasecal(
            msname,
            gaintable=kcross_table,
            datacolumn="CORRECTED_DATA",
            applymode=applymode,
            flagbackup=flagbackup,
        )
        print ('Calibration solutions applied.\n')
        return 0
    except Exception as e:
        print("Exception: ", e)
        return 1


################################
def main():
    usage = "Apply calibration solutions"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
        metavar="String",
    )
    parser.add_option(
        "--bandpass_table",
        dest="bandpass_table",
        default=None,
        help="Name of the bandpass table",
        metavar="String",
    )
    parser.add_option(
        "--kcross_table",
        dest="kcross_table",
        default=None,
        help="Crosshand phase table",
        metavar="String",
    )
    parser.add_option(
        "--applymode",
        dest="applymode",
        default="calflag",
        help="Solution apply and flag",
        metavar="String",
    )
    parser.add_option(
        "--flagbackup",
        dest="flagbackup",
        default=True,
        help="Keep flag backup before applying solutions or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--do_flag",
        dest="do_flag",
        default=True,
        help="Perform flagging or not",
        metavar="Boolean",
    )
    (options, args) = parser.parse_args()
    if options.msname == None:
        print("Please provide the measurement set name.\n")
        return 1
    if options.bandpass_table == None:
        print("Please provide the bandpass table name.\n")
        return 1
    if options.bandpass_table == None:
        print("Please provide the crosshand phase table name.\n")
        return 1
    if eval(options.do_flag):
        print("Flagging: " + options.msname)
        flagdata(vis=options.msname, mode="tfcrop")    
    msg = apply_sol(
        options.msname,
        options.bandpass_table,
        options.kcross_table,
        applymode=options.applymode,
        flagbackup=eval(str(options.flagbackup)),
    )
    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)
