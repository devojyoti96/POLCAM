from casatasks import applycal
from calibrate_crossphase import apply_crossphasecal
from optparse import OptionParser
import os
os.system("rm -rf casa*log")

########
# Inputs
########
msname = input("Name of the target measurement set:")
bandpass_table = input("Path of the bandpass table:")
kcross_table = input("Path of the crosshand phase table:")

def apply_sol(msname,bandpass_table,kcross_table):
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
    """
    try:            
        print("Applying bandpass solutons...\n")
        applycal(vis=msname, gaintable=[bandpass_table], applymode="calflag", flagbackup=True)
        print("Applying crosshand phase solutions...\n")
        apply_crossphasecal(msname, gaintable=kcross_table, datacolumn="CORRECTED_DATA")
        return 0
    except Exception as e:
        print ('Exception: ',e)
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
    if options.msname==None:
        print ('Please provide the measurement set name.\n')
        return 1
    if options.bandpass_table==None:
        print ('Please provide the bandpass table name.\n')
        return 1
    if options.bandpass_table==None:
        print ('Please provide the crosshand phase table name.\n')
        return 1    
    msg=apply_sol(options.msname,options.bandpass_table,options.kcross_table)   
    return msg

if __name__ == "__main__":
    result=main()
    os._exit(result)            
